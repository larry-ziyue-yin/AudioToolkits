from pathlib import Path
import importlib.util
import urllib.request

from .base import MetricBase
from .utils import optional_import, ensure_dir, MetricSkip


class WVMOSMetric(MetricBase):
    name = "wvmos"
    supports_src = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = None
        self.device = None
        self.cache_dir = None

    def _install_hf_local_patch(self, wvmos, context):
        hf_model_dir = self.cfg.get("hf_model_dir") or self.cfg.get("hf_model_name_or_path")
        if not hf_model_dir:
            return lambda: None
        hf_model_dir = Path(hf_model_dir).expanduser()
        if not hf_model_dir.is_absolute():
            hf_model_dir = (Path.cwd() / hf_model_dir).resolve()
        required = ("config.json", "preprocessor_config.json", "pytorch_model.bin")
        missing = [name for name in required if not (hf_model_dir / name).exists()]
        if missing:
            raise RuntimeError(f"WVMOS hf_model_dir is missing files {missing}: {hf_model_dir}")

        restore_items = []

        def patch_from_pretrained(cls_name):
            cls = getattr(wvmos, cls_name, None)
            if cls is None or not hasattr(cls, "from_pretrained"):
                return
            original = cls.from_pretrained
            restore_items.append((cls, original))

            def _from_pretrained(inner_cls, name_or_path, *args, **kwargs):
                if str(name_or_path) == "facebook/wav2vec2-base":
                    name_or_path = str(hf_model_dir)
                    kwargs.setdefault("local_files_only", True)
                return original(name_or_path, *args, **kwargs)

            cls.from_pretrained = classmethod(_from_pretrained)

        patch_from_pretrained("Wav2Vec2Model")
        patch_from_pretrained("Wav2Vec2Processor")
        if context and context.logger:
            context.logger.info("WVMOS HF base model local dir: %s", hf_model_dir)

        def _restore():
            for cls, original in restore_items:
                cls.from_pretrained = original

        return _restore

    def prepare(self, context):
        torch = optional_import("torch")
        spec = importlib.util.find_spec("wvmos")
        if spec is None or not spec.origin:
            raise MetricSkip("Missing dependency: wvmos. Install wvmos to use WVMOS.")
        wv_mos_path = Path(spec.origin).parent / "wv_mos.py"
        if not wv_mos_path.exists():
            raise MetricSkip("wvmos package is missing wv_mos.py.")
        module_spec = importlib.util.spec_from_file_location("_audiotoolkits_wv_mos", str(wv_mos_path))
        if module_spec is None or module_spec.loader is None:
            raise MetricSkip("Failed to load wvmos module.")
        wvmos = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(wvmos)
        device = context.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        cache_dir = Path(context.model_cache_dir) / "wvmos"
        ensure_dir(cache_dir)
        model_path = self.cfg.get("model_path") or self.cfg.get("model_name_or_path")
        download_url = self.cfg.get("download_url") or "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1"

        def _download_ckpt(dest_path):
            dest_path = Path(dest_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
            context.logger.info("Downloading WVMOS checkpoint to %s", dest_path)
            urllib.request.urlretrieve(download_url, str(tmp_path))
            tmp_path.replace(dest_path)

        model_cls = getattr(wvmos, "Wav2Vec2MOS", None)
        if model_cls is None:
            raise RuntimeError("wvmos package does not expose Wav2Vec2MOS.")
        use_cuda = device != "cpu"
        ckpt_path = None
        try:
            if model_path:
                ckpt_path = Path(model_path)
                if not ckpt_path.exists():
                    raise RuntimeError(f"WVMOS checkpoint not found: {ckpt_path}")
            else:
                ckpt_path = cache_dir / "wv_mos.ckpt"
                if not ckpt_path.exists():
                    _download_ckpt(ckpt_path)
            restore_hf = self._install_hf_local_patch(wvmos, context)
            try:
                self.model = model_cls(path=str(ckpt_path), cuda=use_cuda)
            finally:
                restore_hf()
        except RuntimeError as exc:
            msg = str(exc)
            if ckpt_path and any(token in msg for token in ["unexpected EOF", "file might be corrupted", "PytorchStreamReader"]):
                raise RuntimeError(
                    "WVMOS checkpoint appears corrupted: "
                    f"{ckpt_path}. Delete it and re-download or set model_path to a valid ckpt."
                ) from exc
            raise

    def _score(self, audio_path):
        if self.model is None:
            raise RuntimeError("WVMOS model is not initialized.")
        for name in ["calculate_one", "predict", "score", "get_score", "__call__"]:
            if hasattr(self.model, name):
                method = getattr(self.model, name)
                try:
                    result = method(str(audio_path))
                except TypeError:
                    result = method(audio_path)
                if isinstance(result, dict):
                    for key in ["mos", "score", "wvmos"]:
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
        score = self._score(audio_path)
        if score is None:
            return {}
        key = self.name if role == "gen" else f"{self.name}_src"
        return {key: score}
