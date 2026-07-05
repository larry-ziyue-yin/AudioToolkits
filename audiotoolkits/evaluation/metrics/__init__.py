from .base import MetricBase
from .dnsmos import DNSMOSMetric
from .utmos import UTMOSMetric
from .wvmos import WVMOSMetric
from .nisqa import NISQAMetric
from .speechbertscore import SpeechBERTScoreMetric
from .wavlm_sim import WavLMSimMetric
from .wespeaker_sim import WeSpeakerSimMetric
from .resemblyzer_secs import ResemblyzerSECSMetric
from .wer import WerCerMetric


METRIC_REGISTRY = {
    "dnsmos": DNSMOSMetric,
    "utmos": UTMOSMetric,
    "wvmos": WVMOSMetric,
    "nisqa": NISQAMetric,
    "speechbertscore": SpeechBERTScoreMetric,
    "wavlm_sim": WavLMSimMetric,
    "wespeaker_sim": WeSpeakerSimMetric,
    "secs": ResemblyzerSECSMetric,
    "resemblyzer_secs": ResemblyzerSECSMetric,
}


def build_metrics(metric_cfgs):
    metrics = []
    for cfg in metric_cfgs or []:
        if not cfg:
            continue
        if cfg.get("enabled") is False:
            continue
        name = str(cfg.get("name", "")).strip().lower()
        if not name:
            continue
        if name == "wer":
            metrics.append(WerCerMetric(cfg, char_level=False))
        elif name == "cer":
            metrics.append(WerCerMetric(cfg, char_level=True))
        else:
            cls = METRIC_REGISTRY.get(name)
            if not cls:
                raise ValueError(f"Unknown metric: {name}")
            metrics.append(cls(cfg))
    return metrics
