from pathlib import Path
import importlib
import sys

from .base import MetricBase
from .utils import optional_import, MetricSkip


def _import_local_nisqa():
    base_dir = Path(__file__).resolve().parents[2]
    local_root = base_dir / "libs" / "NISQA"
    if not local_root.exists():
        raise MetricSkip("Local NISQA toolkit not found at audiotoolkits/libs/NISQA.")
    local_root_str = str(local_root)
    if local_root_str not in sys.path:
        sys.path.insert(0, local_root_str)
    for key in list(sys.modules.keys()):
        if key == "nisqa" or key.startswith("nisqa."):
            del sys.modules[key]
    try:
        nisqa_lib = importlib.import_module("nisqa.NISQA_lib")
        nisqa_model = importlib.import_module("nisqa.NISQA_model")
    except ImportError as exc:
        raise MetricSkip("Failed to import local NISQA toolkit from audiotoolkits/libs/NISQA.") from exc
    return nisqa_lib, nisqa_model


def _build_local_model_cls(nisqa_model):
    class _LocalNISQAModel(nisqa_model.nisqaModel):
        def _loadModel(self):
            super()._loadModel()
            if "ms_channel" not in self.args:
                self.args["ms_channel"] = None

    return _LocalNISQAModel


def _get_audio_samplerate(audio_path):
    try:
        import librosa
    except ImportError:
        return None
    try:
        return int(librosa.get_samplerate(str(audio_path)))
    except Exception:
        return None


class _LocalNISQAWrapper:
    def __init__(
        self,
        model_cls,
        nisqa_lib,
        model_path,
        device,
        batch_size,
        num_workers,
        args_overrides=None,
        default_fmax=20000,
    ):
        self.model_cls = model_cls
        self.nisqa_lib = nisqa_lib
        self.model_path = str(model_path)
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args_overrides = args_overrides or {}
        self.default_fmax = default_fmax
        self.model = None

    def _ensure_model(self, audio_path):
        if isinstance(audio_path, (list, tuple)):
            if not audio_path:
                return None
            audio_path = audio_path[0]
        dynamic_args = dict(self.args_overrides)
        file_sr = None
        if "ms_sr" not in dynamic_args or "ms_fmax" not in dynamic_args:
            file_sr = _get_audio_samplerate(audio_path)
        if "ms_sr" not in dynamic_args and file_sr:
            dynamic_args["ms_sr"] = file_sr
        if "ms_fmax" not in dynamic_args and file_sr:
            safe_fmax = min(self.default_fmax, (file_sr / 2) - 1)
            if safe_fmax > 0:
                dynamic_args["ms_fmax"] = safe_fmax
        if self.model is None:
            args = {
                "mode": "predict_file",
                "deg": str(audio_path),
                "pretrained_model": self.model_path,
                "tr_bs_val": self.batch_size,
                "tr_num_workers": self.num_workers,
                "tr_device": self.device,
            }
            args.update(dynamic_args)
            self.model = self.model_cls(args)
        else:
            self.model.args.update(dynamic_args)
            self.model.args["deg"] = str(audio_path)
            self.model._loadDatasetsFile()
        if self.model.args.get("double_ended"):
            raise MetricSkip("NISQA double-ended model requires reference audio.")
        return self.model

    def score(self, audio_path):
        model = self._ensure_model(audio_path)
        if model is None:
            return None
        if model.args.get("dim"):
            self.nisqa_lib.predict_dim(
                model.model,
                model.ds_val,
                self.batch_size,
                model.dev,
                num_workers=self.num_workers,
            )
        else:
            self.nisqa_lib.predict_mos(
                model.model,
                model.ds_val,
                self.batch_size,
                model.dev,
                num_workers=self.num_workers,
            )
        try:
            return float(model.ds_val.df["mos_pred"].iloc[0])
        except Exception:
            return None


class NISQAMetric(MetricBase):
    name = "nisqa"
    supports_src = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = None
        self.device = None
        self.cache_dir = None

    def prepare(self, context):
        torch = optional_import("torch")
        nisqa_lib, nisqa_model = _import_local_nisqa()
        model_cls = _build_local_model_cls(nisqa_model)
        device = context.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, str) and device.startswith("cuda"):
            device = "cuda"
        else:
            device = "cpu"
        self.device = device
        model_path = self.cfg.get("model_path") or self.cfg.get("model_name_or_path")
        if not model_path:
            model_path = Path(__file__).resolve().parents[2] / "libs" / "NISQA" / "weights" / "nisqa_tts.tar"
        model_path = Path(model_path).expanduser()
        if not model_path.is_absolute():
            model_path = (Path.cwd() / model_path).resolve()
        if not model_path.exists():
            raise MetricSkip(f"NISQA checkpoint not found: {model_path}")
        batch_size = int(self.cfg.get("batch_size") or self.cfg.get("bs") or 1)
        num_workers = int(self.cfg.get("num_workers") or 0)
        args_overrides = {}
        for key in (
            "ms_sr",
            "ms_fmax",
            "ms_n_mels",
            "ms_n_fft",
            "ms_hop_length",
            "ms_win_length",
            "ms_seg_length",
            "ms_seg_hop_length",
            "ms_max_segments",
            "ms_channel",
        ):
            if key in self.cfg and self.cfg[key] is not None:
                args_overrides[key] = self.cfg[key]
        self.model = _LocalNISQAWrapper(
            model_cls=model_cls,
            nisqa_lib=nisqa_lib,
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            args_overrides=args_overrides,
            default_fmax=20000,
        )

    def _extract_score(self, result):
        if isinstance(result, dict):
            for key in ["mos", "mos_pred", "score"]:
                if key in result:
                    return float(result[key])
        try:
            return float(result)
        except Exception:
            return None

    def _score(self, audio_path):
        if self.model is None:
            raise RuntimeError("NISQA model not initialized.")
        for name in ["predict", "score", "__call__"]:
            if hasattr(self.model, name):
                method = getattr(self.model, name)
                try:
                    result = method([str(audio_path)])
                except TypeError:
                    result = method(str(audio_path))
                if hasattr(result, "to_dict"):
                    try:
                        result = result.to_dict(orient="records")[0]
                    except Exception:
                        pass
                score = self._extract_score(result)
                if score is not None:
                    return score
        return None

    def compute(self, item, context, role="gen"):
        if self.model is None:
            self.prepare(context)
        audio_path = item.gen_path if role == "gen" else item.src_path
        if audio_path is None:
            return {}
        score = self._score(audio_path)
        if score is None:
            return {}
        key = self.name if role == "gen" else f"{self.name}_src"
        return {key: score}
