import os
from pathlib import Path

from .base import MetricBase
from .utils import optional_import, ensure_dir, load_audio, MetricSkip


class WeSpeakerSimMetric(MetricBase):
    name = "wespeaker_sim"
    requires_gt_audio = True
    supports_src = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = None
        self.device = None
        self.cache_dir = None
        self.save_intermediate = True
        self.embedding_path = None
        self.embeddings = {}
        self.logger = None
        pair_with = str(cfg.get("pair_with", "gt")).strip().lower()
        self.pair_with = pair_with
        self.ref_token = cfg.get("ref_token", "_ref_")
        self.ref_index = {}
        self.ref_index_built = False
        if self.pair_with not in ("gt", "src", "ref"):
            raise ValueError(
                f"WeSpeaker pair_with must be 'gt', 'src' or 'ref', got: {self.pair_with}"
            )
        self.requires_gt_audio = self.pair_with == "gt"
        if self.pair_with == "src":
            # pair_with=src 只输出 gen vs src，避免出现 src vs src
            self.supports_src = False

    def _move_model_to_device(self, device):
        torch = optional_import("torch")
        if hasattr(self.model, "set_device"):
            try:
                self.model.set_device(device)
            except Exception:
                pass
        for module in (self.model, getattr(self.model, "model", None)):
            if not isinstance(module, torch.nn.Module):
                continue
            try:
                module.to(device)
                module.eval()
            except Exception:
                pass

    def _extract_with_named_methods(self, audio_path):
        for name in ["extract_embedding", "get_embedding", "embedding"]:
            if not hasattr(self.model, name):
                continue
            method = getattr(self.model, name)
            try:
                return method(str(audio_path))
            except TypeError:
                continue
        return None

    def _extract_with_forward(self, audio_path):
        if not hasattr(self.model, "forward"):
            return None
        wav, _ = load_audio(audio_path, target_sr=16000)
        if self.device and str(self.device) != "cpu":
            wav = wav.to(self.device)
        return self.model(wav.unsqueeze(0))

    def prepare(self, context):
        torch = optional_import("torch")
        wespeaker = optional_import("wespeaker", "Install wespeaker to use speaker similarity.")
        self.logger = context.logger
        model_path = self.cfg.get("model_path")
        if not model_path:
            raise RuntimeError("WeSpeaker model_path is required.")
        model_path_str = str(model_path)
        model_path_obj = Path(model_path_str)
        model_dir = None
        model_id = None
        looks_like_path = (
            "/" in model_path_str or "\\" in model_path_str
            or model_path_obj.suffix in (".pt", ".pth")
        )
        if model_path_obj.exists():
            model_dir = model_path_obj if model_path_obj.is_dir() else model_path_obj.parent
        else:
            try:
                from wespeaker.cli.hub import Hub
                hub_assets = set(Hub.Assets.keys())
            except Exception:
                hub_assets = None
            if hub_assets and model_path_str in hub_assets:
                model_id = model_path_str
            elif looks_like_path:
                raise RuntimeError(f"WeSpeaker model_path not found: {model_path_obj}")
            elif hub_assets:
                supported = ", ".join(sorted(hub_assets))
                raise RuntimeError(f"Unsupported WeSpeaker model id: {model_path_str}. Supported: {supported}")
            else:
                model_id = model_path_str
        device = context.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if hasattr(wespeaker, "load_model"):
            try:
                if model_dir is not None:
                    self.model = wespeaker.load_model(model_dir=str(model_dir))
                else:
                    self.model = wespeaker.load_model(model_id=model_id)
            except TypeError:
                try:
                    self.model = wespeaker.load_model(model_path_str, device=device)
                except TypeError:
                    self.model = wespeaker.load_model(model_path_str)
        else:
            try:
                from wespeaker.cli.speaker import Speaker
                if model_dir is None:
                    from wespeaker.cli.hub import Hub
                    if model_id not in Hub.Assets:
                        supported = ", ".join(sorted(Hub.Assets.keys()))
                        raise RuntimeError(
                            f"Unsupported WeSpeaker model id: {model_id}. Supported: {supported}"
                        )
                    model_dir = Hub.get_model(model_id)
                self.model = Speaker(str(model_dir))
            except Exception as exc:  # pragma: no cover - fallback path
                raise RuntimeError("Unsupported wespeaker API; provide a compatible version.") from exc
        self._move_model_to_device(device)
        self.cache_dir = Path(context.intermediate_dir) / "embeddings" / "wespeaker"
        ensure_dir(self.cache_dir)
        self.save_intermediate = context.save_intermediate
        default_path = self.cache_dir / "wespeaker_embeds.pt"
        self.embedding_path = Path(self.cfg.get("embedding_path") or default_path).expanduser()
        ensure_dir(self.embedding_path.parent)
        if self.save_intermediate and self.embedding_path.exists():
            try:
                self.embeddings = torch.load(self.embedding_path, weights_only=False)
            except Exception as exc:
                if self.logger:
                    self.logger.warning("读取 WeSpeaker Embedding 文件失败: %s", exc)
                self.embeddings = {}
        if self.pair_with == "ref":
            if not self.ref_token:
                raise ValueError("WeSpeaker pair_with=ref requires ref_token, e.g. _ref_")
            self._build_ref_index(context)

    def _build_ref_index(self, context):
        if self.ref_index_built:
            return
        root = Path(context.cfg["data"]["root"])
        exts = context.cfg["data"]["audio_extensions"]
        if self.logger:
            self.logger.info("WeSpeaker: 扫描 ref 文件 (root=%s, token=%s)", root, self.ref_token)
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
            self.logger.info("WeSpeaker: ref 文件数=%d, 关联 src_id=%d", total, len(self.ref_index))

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
            self.logger.warning(
                "WeSpeaker: %s 有多个 ref 候选(%d)，使用 %s",
                item.utt_id,
                len(candidates),
                candidates[0],
            )
        return candidates[0]

    def _extract_embedding(self, audio_path, utt_id, role):
        torch = optional_import("torch")
        cache_key = str(Path(audio_path).resolve())
        if cache_key in self.embeddings:
            return self.embeddings[cache_key]
        emb = self._extract_with_named_methods(audio_path)
        if emb is None:
            emb = self._extract_with_forward(audio_path)
        if emb is None:
            raise RuntimeError("Could not extract embedding from wespeaker model.")
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        emb = emb.squeeze().detach().cpu()
        self.embeddings[cache_key] = emb
        if self.save_intermediate:
            try:
                torch.save(self.embeddings, self.embedding_path)
            except Exception as exc:
                if self.logger:
                    self.logger.warning("保存 WeSpeaker Embedding 文件失败: %s", exc)
        return emb

    def compute(self, item, context, role="gen"):
        if self.model is None:
            self.prepare(context)
        torch = optional_import("torch")
        audio_path = item.gen_path if role == "gen" else item.src_path
        if audio_path is None:
            raise MetricSkip(f"缺少 {role} 音频")
        if self.pair_with == "gt":
            target_path = item.gt_path
            target_role = "gt"
            if target_path is None:
                raise MetricSkip("缺少 gt 音频")
        elif self.pair_with == "src":
            if role == "src":
                raise MetricSkip("pair_with=src 时不计算 src 角色")
            target_path = item.src_path
            target_role = "src_ref"
            if target_path is None:
                raise MetricSkip("缺少 src 音频")
        else:
            target_path = self._select_ref_path(item, audio_path, context)
            target_role = "ref"
            if target_path is None:
                raise MetricSkip("缺少 ref 音频")
        emb_a = self._extract_embedding(audio_path, item.utt_id, role)
        emb_b = self._extract_embedding(target_path, item.utt_id, target_role)
        sim = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        key = self.name if role == "gen" else f"{self.name}_src"
        return {key: sim}
