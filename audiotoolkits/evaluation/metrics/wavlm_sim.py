import os
from pathlib import Path

from .base import MetricBase
from .utils import optional_import, ensure_dir, load_audio


class WavLMSimMetric(MetricBase):
    name = "wavlm_sim"
    requires_gt_audio = True
    supports_src = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = None
        self.processor = None
        self.device = None
        self.cache_dir = None
        self.save_intermediate = True
        self.embedding_path = None
        self.embeddings = {}
        self.logger = None
        self.min_audio_samples = None

    def prepare(self, context):
        torch = optional_import("torch")
        transformers = optional_import("transformers", "Install transformers to use WavLM.")
        self.logger = context.logger
        model_name = self.cfg.get("model_name_or_path", "microsoft/wavlm-base-plus-sv")
        local_files_only = bool(self.cfg.get("local_files_only", False))
        revision = self.cfg.get("revision")
        token = self.cfg.get("hf_token") or self.cfg.get("token")
        hf_timeout = self.cfg.get("hf_timeout")
        cache_dir = Path(context.model_cache_dir) / "wavlm"
        ensure_dir(cache_dir)
        if hf_timeout is not None:
            try:
                timeout_s = max(1, int(hf_timeout))
            except (TypeError, ValueError):
                timeout_s = None
            if timeout_s is not None:
                os.environ["HF_HUB_ETAG_TIMEOUT"] = str(timeout_s)
                os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout_s)
        load_kwargs = {
            "cache_dir": str(cache_dir),
            "local_files_only": local_files_only,
        }
        if revision:
            load_kwargs["revision"] = revision
        if token:
            load_kwargs["token"] = token
        try:
            model_cls = transformers.AutoModelForAudioXVector
        except AttributeError:
            model_cls = transformers.AutoModel
        # Prefer a feature extractor to avoid tokenizer requirements for SV models.
        try:
            feature_extractor_cls = transformers.AutoFeatureExtractor
        except AttributeError:
            feature_extractor_cls = None
        if self.logger:
            self.logger.info(
                "加载 WavLM 处理器: model=%s, local_files_only=%s, cache_dir=%s",
                model_name,
                local_files_only,
                cache_dir,
            )
        if feature_extractor_cls is not None:
            try:
                self.processor = self._from_pretrained(feature_extractor_cls, model_name, load_kwargs)
            except Exception:
                self.processor = self._from_pretrained(transformers.AutoProcessor, model_name, load_kwargs)
        else:
            self.processor = self._from_pretrained(transformers.AutoProcessor, model_name, load_kwargs)
        if self.logger:
            self.logger.info("WavLM 处理器就绪")
            self.logger.info(
                "加载 WavLM 模型: model=%s, local_files_only=%s, cache_dir=%s",
                model_name,
                local_files_only,
                cache_dir,
            )
        self.model = self._from_pretrained(model_cls, model_name, load_kwargs)
        device = context.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device).eval()
        if self.logger:
            self.logger.info("WavLM 模型就绪: device=%s", device)
        self.cache_dir = Path(context.intermediate_dir) / "embeddings" / "wavlm"
        ensure_dir(self.cache_dir)
        self.save_intermediate = context.save_intermediate
        default_path = self.cache_dir / "wavlm_embeds.pt"
        self.embedding_path = Path(self.cfg.get("embedding_path") or default_path).expanduser()
        ensure_dir(self.embedding_path.parent)
        if self.save_intermediate and self.embedding_path.exists():
            try:
                self.embeddings = torch.load(self.embedding_path, weights_only=False)
            except Exception as exc:
                if self.logger:
                    self.logger.warning("读取 WavLM Embedding 文件失败: %s", exc)
                self.embeddings = {}
        self.min_audio_samples = self._infer_min_audio_samples()
        if self.min_audio_samples is None and hasattr(self.model, "xvector"):
            self.min_audio_samples = 16000

    @staticmethod
    def _from_pretrained(cls, model_name, kwargs):
        try:
            return cls.from_pretrained(model_name, **kwargs)
        except TypeError:
            legacy_kwargs = dict(kwargs)
            token = legacy_kwargs.pop("token", None)
            if token is not None:
                legacy_kwargs["use_auth_token"] = token
            return cls.from_pretrained(model_name, **legacy_kwargs)

    def _infer_min_audio_samples(self):
        min_frames = self._infer_min_frame_length()
        if min_frames is None:
            return None
        conv_kernel = getattr(self.model.config, "conv_kernel", None)
        conv_stride = getattr(self.model.config, "conv_stride", None)
        if not conv_kernel or not conv_stride or len(conv_kernel) != len(conv_stride):
            return None
        length = int(min_frames)
        for kernel, stride in zip(reversed(conv_kernel), reversed(conv_stride)):
            length = (length - 1) * int(stride) + int(kernel)
        return length

    def _infer_min_frame_length(self):
        tdnn_kernel = getattr(self.model.config, "tdnn_kernel", None)
        if not tdnn_kernel:
            return None
        tdnn_dilation = getattr(self.model.config, "tdnn_dilation", None)
        if tdnn_dilation and len(tdnn_dilation) == len(tdnn_kernel):
            # X-vector statistics pooling computes an unbiased standard
            # deviation, so at least two frames must survive all TDNN layers.
            return 2 + sum(
                int(d) * (int(k) - 1)
                for k, d in zip(tdnn_kernel, tdnn_dilation)
            )
        return 2 + sum(int(k) - 1 for k in tdnn_kernel)

    def _encode(self, audio_path, utt_id, role):
        torch = optional_import("torch")
        cache_key = f"{utt_id}_{role}"
        if cache_key in self.embeddings:
            return self.embeddings[cache_key]
        wav, sr = load_audio(audio_path, target_sr=16000)
        if self.min_audio_samples and wav.numel() < self.min_audio_samples:
            pad_len = self.min_audio_samples - wav.numel()
            wav = torch.nn.functional.pad(wav, (0, pad_len))
        inputs = self.processor(wav.numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, "embeddings"):
                emb = outputs.embeddings
            elif hasattr(outputs, "xvector"):
                emb = outputs.xvector
            elif hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state.mean(dim=1)
            else:
                raise RuntimeError("Unsupported WavLM output format.")
        emb = emb.squeeze(0).detach().cpu()
        self.embeddings[cache_key] = emb
        if self.save_intermediate:
            try:
                torch.save(self.embeddings, self.embedding_path)
            except Exception as exc:
                if self.logger:
                    self.logger.warning("保存 WavLM Embedding 文件失败: %s", exc)
        return emb

    def compute(self, item, context, role="gen"):
        if self.model is None:
            self.prepare(context)
        torch = optional_import("torch")
        if item.gt_path is None:
            return {}
        audio_path = item.gen_path if role == "gen" else item.src_path
        if audio_path is None:
            return {}
        emb_a = self._encode(audio_path, item.utt_id, role)
        emb_b = self._encode(item.gt_path, item.utt_id, "gt")
        sim = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        key = self.name if role == "gen" else f"{self.name}_src"
        return {key: sim}
