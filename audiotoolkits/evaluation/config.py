import os
from copy import deepcopy


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _expand_path(path):
    if path is None:
        return None
    return os.path.expandvars(os.path.expanduser(str(path)))


def _normalize_extensions(exts):
    if not exts:
        return [".wav"]
    out = []
    for ext in exts:
        ext = str(ext)
        if not ext.startswith("."):
            ext = "." + ext
        out.append(ext)
    return out


def _as_positive_int(value, default):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


DEFAULT_CONFIG = {
    "data": {
        "root": ".",
        "gt_root": None,
        "src_root": None,
        "recursive": True,
        "audio_extensions": [".wav"],
        "gen_suffix": "_gen",
        "gt_suffix": "_gt",
        "src_suffix": "_whisper",
        "eval_src": False,
        "manifest": None,
        "gt_text_path": None,
        "missing_policy": "skip",
        "scan_progress": False,
    },
    "asr": {
        "enabled": True,
        "backend": "whisper",
        "model_name": "turbo",
        "language": None,
        "convert_to_simplified": False,
        "beam_size": None,
        "task": "transcribe",
        "fp16": True,
        "force_script_defaults": True,
        "write_flush_every": 50,
        "output_path": None,
        "output_path_src": None,
        "device": "auto",
    },
    "output": {
        "output_dir": None,
        "results_csv": "results.csv",
        "summary_csv": "summary.csv",
        "cache_dir": "~/.cache/audiotoolkits",
        "save_intermediate": True,
        "overwrite": True,
        "max_error_logs": 3,
        "max_skip_samples": 3,
    },
    "parallel": {
        "enabled": False,
        "devices": "auto",
        "workers_per_device": 1,
        "chunk_size": 8,
        "precompute_asr": True,
    },
    "metrics": [],
}


def normalize_config(cfg):
    cfg["data"]["root"] = _expand_path(cfg["data"].get("root") or ".")
    cfg["data"]["gt_root"] = _expand_path(cfg["data"].get("gt_root")) or cfg["data"]["root"]
    cfg["data"]["src_root"] = _expand_path(cfg["data"].get("src_root")) or cfg["data"]["root"]
    cfg["data"]["manifest"] = _expand_path(cfg["data"].get("manifest"))
    cfg["data"]["gt_text_path"] = _expand_path(cfg["data"].get("gt_text_path"))
    cfg["data"]["audio_extensions"] = _normalize_extensions(cfg["data"].get("audio_extensions"))

    cfg["output"]["output_dir"] = _expand_path(cfg["output"].get("output_dir"))
    cfg["output"]["results_csv"] = _expand_path(cfg["output"].get("results_csv") or "results.csv")
    cfg["output"]["summary_csv"] = _expand_path(cfg["output"].get("summary_csv") or "summary.csv")
    cfg["output"]["cache_dir"] = _expand_path(cfg["output"].get("cache_dir") or "~/.cache/audiotoolkits")

    cfg["asr"]["output_path"] = _expand_path(cfg["asr"].get("output_path"))
    cfg["asr"]["output_path_src"] = _expand_path(cfg["asr"].get("output_path_src"))
    cfg["parallel"]["workers_per_device"] = _as_positive_int(
        cfg["parallel"].get("workers_per_device"), 1
    )
    cfg["parallel"]["chunk_size"] = _as_positive_int(cfg["parallel"].get("chunk_size"), 8)

    return cfg


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load config files. Install with `pip install pyyaml`.") from exc

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    cfg = deepcopy(DEFAULT_CONFIG)
    _deep_update(cfg, data)
    return normalize_config(cfg)
