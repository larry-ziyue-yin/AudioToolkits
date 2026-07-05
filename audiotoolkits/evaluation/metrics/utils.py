from pathlib import Path
import importlib


class MetricSkip(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def optional_import(module_name, hint=None):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        msg = f"Missing dependency: {module_name}."
        if hint:
            msg += f" {hint}"
        raise MetricSkip(msg) from exc


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_audio(path, target_sr=16000):
    torchaudio = optional_import("torchaudio", "Install torchaudio to load audio.")
    torch = optional_import("torch")
    wav, sr = torchaudio.load(path)
    if wav.dim() > 1 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
        sr = target_sr
    wav = wav.squeeze(0).to(torch.float32)
    return wav, sr
