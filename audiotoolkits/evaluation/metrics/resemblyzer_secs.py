import os
from pathlib import Path

from .base import MetricBase
from .utils import optional_import, ensure_dir, MetricSkip


class ResemblyzerSECSMetric(MetricBase):
    name = "secs"
    requires_gt_audio = True
    supports_src = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.encoder = None
        self.save_intermediate = True
        self.embedding_path = None
        self.embeddings = {}
        self.logger = None
        self.pair_with = str(cfg.get("pair_with", "gt")).lower()
        self.ref_token = cfg.get("ref_token", "_ref_")
        self.ref_index = {}
        self.ref_index_built = False
        if self.pair_with not in ("gt", "ref", "tgt"):
            raise ValueError(f"SECS pair_with must be 'gt' or 'ref' or 'tgt', got: {self.pair_with}")
        self.requires_gt_audio = self.pair_with == "gt"

    def prepare(self, context):
        torch = optional_import("torch")
        resemblyzer = optional_import("resemblyzer", "Install resemblyzer to use SECS.")
        self.logger = context.logger
        device = context.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.encoder = resemblyzer.VoiceEncoder(device=device)
        except TypeError:
            self.encoder = resemblyzer.VoiceEncoder()
        self.preprocess_wav = resemblyzer.preprocess_wav
        self.save_intermediate = context.save_intermediate
        if self.pair_with == "ref":
            if not self.ref_token:
                raise ValueError("SECS pair_with=ref requires ref_token, e.g. _ref_")
            self._build_ref_index(context)
        default_path = Path(context.intermediate_dir) / "embeddings" / "resemblyzer" / "resemblyzer_embeds.pt"
        self.embedding_path = Path(self.cfg.get("embedding_path") or default_path).expanduser()
        ensure_dir(self.embedding_path.parent)
        if self.save_intermediate and self.embedding_path.exists():
            try:
                self.embeddings = torch.load(self.embedding_path, weights_only=False)
            except Exception as exc:
                if self.logger:
                    self.logger.warning("读取 Resemblyzer Embedding 文件失败: %s", exc)
                self.embeddings = {}

    def _embed_key(self, audio_path):
        return Path(audio_path).stem

    def _extract_embedding(self, audio_path, utt_id, role):
        torch = optional_import("torch")
        key = self._embed_key(audio_path)
        if key in self.embeddings:
            return self.embeddings[key]
        wav = self.preprocess_wav(str(audio_path))
        emb = self.encoder.embed_utterance(wav)
        if self.save_intermediate:
            self.embeddings[key] = emb
            try:
                torch.save(self.embeddings, self.embedding_path)
            except Exception as exc:
                if self.logger:
                    self.logger.warning("保存 Resemblyzer Embedding 文件失败: %s", exc)
        return emb

    def _build_ref_index(self, context):
        if self.ref_index_built:
            return
        root = Path(context.cfg["data"]["root"])
        exts = context.cfg["data"]["audio_extensions"]
        if self.logger:
            self.logger.info("SECS: 扫描 ref 文件 (root=%s, token=%s)", root, self.ref_token)
        total = 0
        for dirpath, _, filenames in os.walk(root, followlinks=False):
            for filename in filenames:
                if self.ref_token not in filename:
                    continue
                if not any(filename.endswith(ext) for ext in exts):
                    continue
                stem = Path(filename).stem
                if self.ref_token not in stem:
                    continue
                src_id, ref_id = stem.split(self.ref_token, 1)
                if not src_id or not ref_id:
                    continue
                path = Path(dirpath) / filename
                self.ref_index.setdefault(src_id, []).append(path)
                total += 1
        self.ref_index_built = True
        if self.logger:
            self.logger.info("SECS: ref 文件数=%d, 关联 src_id=%d", total, len(self.ref_index))

    def _select_ref_path(self, item, audio_path, context):
        if not self.ref_index_built:
            self._build_ref_index(context)
        candidates = self.ref_index.get(item.utt_id)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        root = Path(context.cfg["data"]["root"])
        rel_audio_dir = None
        try:
            rel_audio_dir = audio_path.parent.relative_to(root)
        except ValueError:
            rel_audio_dir = None
        if rel_audio_dir:
            for path in candidates:
                try:
                    if path.parent.relative_to(root) == rel_audio_dir:
                        return path
                except ValueError:
                    continue
        if self.logger:
            self.logger.warning("SECS: %s 有多个 ref 候选(%d)，使用 %s", item.utt_id, len(candidates), candidates[0])
        return candidates[0]

    def compute(self, item, context, role="gen"):
        if self.encoder is None:
            self.prepare(context)
        torch = optional_import("torch")
        audio_path = item.gen_path if role == "gen" else item.src_path
        if audio_path is None:
            raise MetricSkip(f"缺少 {role} 音频")
        if self.pair_with == "gt":
            target_path = item.gt_path
            if target_path is None:
                raise MetricSkip("缺少 gt 音频")
        else:
            target_path = self._select_ref_path(item, audio_path, context)
            if target_path is None:
                raise MetricSkip("缺少 ref 音频")
        emb_a = self._extract_embedding(audio_path, item.utt_id, role)
        emb_b = self._extract_embedding(target_path, item.utt_id, "ref" if self.pair_with == "ref" else "gt")
        if not isinstance(emb_a, torch.Tensor):
            emb_a = torch.from_numpy(emb_a)
        if not isinstance(emb_b, torch.Tensor):
            emb_b = torch.from_numpy(emb_b)
        sim = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        key = self.name if role == "gen" else f"{self.name}_src"
        return {key: sim}
