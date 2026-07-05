from pathlib import Path

from .base import MetricBase
from .utils import optional_import


class UTMOSMetric(MetricBase):
    name = "utmos"
    supports_src = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.scorer = None
        self.device = None

    def prepare(self, context):
        torch = optional_import("torch")
        from audiotoolkits.libs.mos_kits.utmos.score import Score

        base_dir = Path(__file__).resolve().parents[2]
        ckpt_default = base_dir / "libs" / "mos_kits" / "utmos" / "epoch=3-step=7459.ckpt"
        ckpt_path = Path(self.cfg.get("ckpt_path") or ckpt_default)
        device = context.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.scorer = Score(ckpt_path=str(ckpt_path), input_sample_rate=16000, device=device)

    def compute(self, item, context, role="gen"):
        if self.scorer is None:
            self.prepare(context)
        torchaudio = optional_import("torchaudio")
        audio_path = item.gen_path if role == "gen" else item.src_path
        if audio_path is None:
            return {}
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)
        score = self.scorer.score(wav.to(self.device))
        key = self.name if role == "gen" else f"{self.name}_src"
        return {key: float(score[0]) if hasattr(score, "__len__") else float(score)}
