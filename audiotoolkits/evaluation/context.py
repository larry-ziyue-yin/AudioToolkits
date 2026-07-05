from dataclasses import dataclass, field
from pathlib import Path
import logging


@dataclass
class EvalContext:
    cfg: dict
    output_dir: Path
    model_cache_dir: Path
    device: str
    logger: logging.Logger
    resources: dict = field(default_factory=dict)

    @property
    def save_intermediate(self):
        return bool(self.cfg.get("output", {}).get("save_intermediate", True))

    @property
    def intermediate_dir(self):
        return self.output_dir / "intermediate"
