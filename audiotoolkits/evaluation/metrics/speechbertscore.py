from pathlib import Path

from .base import MetricBase
from .utils import optional_import, ensure_dir, load_audio, MetricSkip


def _import_speechbertscore_backend():
    try:
        import speechbertscore as sbs
        return "speechbertscore", sbs
    except ImportError:
        pass
    try:
        import discrete_speech_metrics as dsm
        return "discrete_speech_metrics", dsm
    except ImportError as exc:
        raise MetricSkip(
            "Missing dependency: speechbertscore or discrete_speech_metrics. "
            "Install one to use SpeechBERTScore."
        ) from exc


def _normalize_bool(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _normalize_score_type(value):
    if value is None:
        return "precision"
    value = str(value).strip().lower()
    return value or "precision"


class SpeechBERTScoreMetric(MetricBase):
    name = "speechbertscore"
    supports_src = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = None
        self.device = None
        self.cache_dir = None
        self.backend = None
        self.sample_rate = None
        self.score_type = _normalize_score_type(cfg.get("score_type"))
        self.ref_type = cfg.get("ref_type", "text")
        if self.ref_type == "text":
            self.requires_ref_text = True
        elif self.ref_type == "audio":
            self.requires_gt_audio = True

    def prepare(self, context):
        torch = optional_import("torch")
        backend, module = _import_speechbertscore_backend()
        self.backend = backend
        device = context.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        cache_dir = Path(context.model_cache_dir) / "speechbertscore"
        ensure_dir(cache_dir)
        model_path = self.cfg.get("model_path") or self.cfg.get("model_name_or_path")

        if backend == "speechbertscore":
            model = None
            if hasattr(module, "SpeechBERTScore"):
                try:
                    model = module.SpeechBERTScore(model_path=model_path, device=device, cache_dir=str(cache_dir))
                except TypeError:
                    model = module.SpeechBERTScore(model_path=model_path, device=device)
            else:
                try:
                    from speechbertscore.speechbertscore import SpeechBERTScore
                    model = SpeechBERTScore(model_path=model_path, device=device)
                except Exception as exc:  # pragma: no cover - fallback
                    raise RuntimeError("Unsupported speechbertscore API; provide a compatible version.") from exc
            self.model = model
            return

        if self.ref_type != "audio":
            raise MetricSkip(
                "DiscreteSpeechMetrics SpeechBERTScore only supports audio reference; "
                "set ref_type: audio or install speechbertscore."
            )
        self.requires_ref_text = False
        self.requires_gt_audio = True
        model_type = self.cfg.get("model_type") or model_path or "wavlm-large"
        sr = int(self.cfg.get("sr") or self.cfg.get("sample_rate") or 16000)
        self.sample_rate = sr
        use_gpu = _normalize_bool(self.cfg.get("use_gpu"))
        if use_gpu is None:
            use_gpu = device != "cpu"
        kwargs = {"sr": sr, "model_type": model_type, "use_gpu": bool(use_gpu)}
        layer = self.cfg.get("layer")
        if layer is not None:
            try:
                kwargs["layer"] = int(layer)
            except (TypeError, ValueError) as exc:
                raise MetricSkip(f"Invalid layer for SpeechBERTScore: {layer}") from exc
        try:
            self.model = module.SpeechBERTScore(**kwargs)
        except TypeError:
            try:
                self.model = module.SpeechBERTScore(sr=sr, model_type=model_type)
            except TypeError:
                self.model = module.SpeechBERTScore(sr=sr)

    def _score(self, audio_path, ref):
        if self.model is None:
            raise RuntimeError("SpeechBERTScore model not initialized.")
        if self.backend == "discrete_speech_metrics":
            target_sr = self.sample_rate or 16000
            ref_wav, _ = load_audio(ref, target_sr=target_sr)
            gen_wav, _ = load_audio(audio_path, target_sr=target_sr)
            precision, recall, f1 = self.model.score(
                ref_wav.cpu().numpy(),
                gen_wav.cpu().numpy(),
            )
            score_type = self.score_type
            if score_type in ("recall", "r"):
                return float(recall)
            if score_type in ("f1", "f1_score", "f"):
                return float(f1)
            return float(precision)
        for name in ["score", "predict", "__call__"]:
            if hasattr(self.model, name):
                method = getattr(self.model, name)
                try:
                    result = method(str(audio_path), ref)
                except TypeError:
                    result = method(audio_path, ref)
                if isinstance(result, dict):
                    for key in ["score", "sbs", "speechbertscore"]:
                        if key in result:
                            return float(result[key])
                try:
                    return float(result)
                except Exception:
                    return None
        return None

    def compute(self, item, context, role="gen"):
        if self.model is None:
            self.prepare(context)
        audio_path = item.gen_path if role == "gen" else item.src_path
        if audio_path is None:
            return {}
        if self.ref_type == "text":
            if not item.ref_text:
                return {}
            ref = item.ref_text
        else:
            if item.gt_path is None:
                return {}
            ref = str(item.gt_path)
        score = self._score(audio_path, ref)
        if score is None:
            return {}
        key = self.name if role == "gen" else f"{self.name}_src"
        return {key: score}
