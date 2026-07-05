import importlib
from pathlib import Path

from .utils import optional_import, ensure_dir

from torch.nn.attention import sdpa_kernel, SDPBackend

class WhisperASR:
    def __init__(self, cfg, intermediate_dir, device, logger, save_intermediate=True):
        self.cfg = cfg
        self.intermediate_dir = Path(intermediate_dir)
        self.logger = logger
        self.save_intermediate = save_intermediate
        self.force_script_defaults = bool(cfg.get("force_script_defaults", True))
        self.convert_to_simplified = bool(cfg.get("convert_to_simplified", False))
        self._opencc_converter = None
        self._opencc_checked = False
        torch = optional_import("torch")
        whisper = optional_import("whisper", "Install openai-whisper to use ASR.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model_name = cfg.get("model_name", "turbo")
        if self.logger:
            self.logger.info("加载 Whisper 模型: name=%s, device=%s", model_name, device)
        self.model = whisper.load_model(model_name, device=device)
        if self.logger:
            self.logger.info("Whisper 模型就绪: name=%s", model_name)
        self.transcribe_kwargs = self._build_transcribe_kwargs()
        if self.convert_to_simplified:
            self._get_opencc_converter()
        self.output_paths = {
            "gen": self._resolve_output_path("gen"),
            "src": self._resolve_output_path("src"),
        }
        self.text_maps = {"gen": {}, "src": {}}
        self.pending_writes = {"gen": [], "src": []}
        self.write_flush_every = max(1, int(cfg.get("write_flush_every", 50)))
        if self.save_intermediate:
            for role in ("gen", "src"):
                self.text_maps[role] = self._load_text_map(self.output_paths[role])

    def _build_transcribe_kwargs(self):
        if self.force_script_defaults:
            kwargs = {
                "task": "transcribe",
                "fp16": True,
            }
        else:
            kwargs = {
                "task": self.cfg.get("task", "transcribe"),
                "fp16": bool(self.cfg.get("fp16", True)),
            }

        language = self.cfg.get("language")
        if language:
            kwargs["language"] = language

        beam_size = self.cfg.get("beam_size")
        if beam_size is not None:
            try:
                beam_size = int(beam_size)
            except (TypeError, ValueError):
                beam_size = None
            if beam_size and beam_size > 0:
                kwargs["beam_size"] = beam_size
        return kwargs

    def _get_opencc_converter(self):
        if self._opencc_checked:
            return self._opencc_converter
        self._opencc_checked = True
        try:
            opencc = importlib.import_module("opencc")
            self._opencc_converter = opencc.OpenCC("t2s")
        except Exception:
            self._opencc_converter = None
            if self.logger:
                self.logger.warning(
                    "convert_to_simplified=True 但未安装 opencc，跳过繁转简。"
                    "可执行: pip install opencc-python-reimplemented"
                )
        return self._opencc_converter

    def _normalize_text(self, text):
        text = (text or "").strip()
        if not text:
            return ""
        if self.convert_to_simplified:
            converter = self._get_opencc_converter()
            if converter is not None:
                text = converter.convert(text)
        return text

    def _resolve_output_path(self, role):
        key = "output_path" if role == "gen" else "output_path_src"
        path = self.cfg.get(key)
        if path:
            return Path(path)
        return self.intermediate_dir / "asr" / f"{role}_text.txt"

    def _load_text_map(self, path):
        mapping = {}
        if not path.exists():
            return mapping
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                utt_id = parts[0]
                text = " ".join(parts[1:])
                text = self._normalize_text(text)
                mapping[utt_id] = text
        return mapping

    def _append_text(self, role, utt_id, text):
        path = self.output_paths[role]
        ensure_dir(path.parent)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{utt_id} {text}\n")

    def _append_many(self, role, rows):
        if not rows:
            return
        path = self.output_paths[role]
        ensure_dir(path.parent)
        with open(path, "a", encoding="utf-8") as fh:
            for utt_id, text in rows:
                fh.write(f"{utt_id} {text}\n")

    def _cache_text(self, role, utt_id, text, persist=True):
        self.text_maps.setdefault(role, {})[utt_id] = text
        if not self.save_intermediate or not persist:
            return
        self.pending_writes.setdefault(role, []).append((utt_id, text))
        if len(self.pending_writes[role]) >= self.write_flush_every:
            self.flush(role=role)

    def update_texts(self, role, mapping, persist=True):
        rows = []
        role_map = self.text_maps.setdefault(role, {})
        for utt_id, text in mapping.items():
            if utt_id in role_map:
                continue
            role_map[utt_id] = text
            rows.append((utt_id, text))
        if not rows:
            return 0
        if self.save_intermediate and persist:
            self._append_many(role, rows)
        return len(rows)

    def flush(self, role=None):
        if not self.save_intermediate:
            return
        roles = (role,) if role else ("gen", "src")
        for r in roles:
            rows = self.pending_writes.get(r) or []
            if not rows:
                continue
            self._append_many(r, rows)
            self.pending_writes[r] = []

    def transcribe(self, item, role="gen", persist=True):
        cached = self.text_maps.get(role, {}).get(item.utt_id)
        if cached is not None:
            return cached.strip()

        audio_path = item.gen_path if role == "gen" else item.src_path
        if audio_path is None:
            return ""

        kwargs = self.transcribe_kwargs
        try:
            result = self.model.transcribe(str(audio_path), **kwargs)
        except Exception:
            with sdpa_kernel(SDPBackend.MATH):
                result = self.model.transcribe(str(audio_path), **kwargs)

        text = self._normalize_text(result.get("text"))
        self._cache_text(role, item.utt_id, text, persist=persist)
        return text


def build_asr(cfg, intermediate_dir, device, logger, save_intermediate=True):
    backend = str(cfg.get("backend", "whisper")).lower()
    if backend == "whisper":
        return WhisperASR(cfg, intermediate_dir, device, logger, save_intermediate=save_intermediate)
    raise RuntimeError(f"Unsupported ASR backend: {backend}")
