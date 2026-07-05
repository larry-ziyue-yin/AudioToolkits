from dataclasses import dataclass
import os
from pathlib import Path
import csv
import json
from typing import Optional


@dataclass
class EvalItem:
    utt_id: str
    gen_path: Path
    gt_path: Optional[Path]
    src_path: Optional[Path]
    ref_text: Optional[str]


def _load_text_map(path):
    if not path:
        return {}
    mapping = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip("\n")
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            utt_id = parts[0]
            text = " ".join(parts[1:])
            mapping[utt_id] = text
    return mapping


def _suffix_includes_ext(suffix, exts):
    for ext in exts:
        if suffix.endswith(ext):
            return True
    return False


def _extract_utt_id(path, suffix, exts):
    name = path.name
    if _suffix_includes_ext(suffix, exts):
        if name.endswith(suffix):
            return name[: -len(suffix)]
        return None
    stem = path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return None


def _resolve_pair_path(root, utt_id, suffix, exts, prefer_ext=None):
    root = Path(root)
    if _suffix_includes_ext(suffix, exts):
        candidate = root / f"{utt_id}{suffix}"
        return candidate if candidate.exists() else None
    if prefer_ext and prefer_ext in exts:
        candidate = root / f"{utt_id}{suffix}{prefer_ext}"
        if candidate.exists():
            return candidate
    for ext in exts:
        candidate = root / f"{utt_id}{suffix}{ext}"
        if candidate.exists():
            return candidate
    return None


def _iter_audio_files(root, recursive, exts):
    root = Path(root)
    if recursive:
        for ext in exts:
            yield from root.rglob(f"*{ext}")
    else:
        for ext in exts:
            yield from root.glob(f"*{ext}")


def _iter_suffix_files(root, suffix, exts, recursive):
    root = Path(root)
    if not suffix:
        yield from _iter_audio_files(root, recursive, exts)
        return
    if recursive:
        yield from _iter_suffix_files_walk(root, suffix, exts)
        return
    patterns = []
    if _suffix_includes_ext(suffix, exts):
        patterns.append(f"*{suffix}")
    else:
        for ext in exts:
            patterns.append(f"*{suffix}{ext}")
    for pattern in patterns:
        if recursive:
            yield from root.rglob(pattern)
        else:
            yield from root.glob(pattern)


def _match_suffix(filename, suffix, exts):
    if not suffix:
        return any(filename.endswith(ext) for ext in exts)
    if _suffix_includes_ext(suffix, exts):
        return filename.endswith(suffix)
    for ext in exts:
        if filename.endswith(f"{suffix}{ext}"):
            return True
    return False


def _iter_suffix_files_walk(root, suffix, exts):
    for dirpath, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            if _match_suffix(filename, suffix, exts):
                yield Path(dirpath) / filename


def _build_suffix_index(root, suffix, exts, recursive):
    index = {}
    for path in _iter_suffix_files(root, suffix, exts, recursive):
        utt_id = _extract_utt_id(path, suffix, exts)
        if not utt_id:
            continue
        if utt_id not in index:
            index[utt_id] = path
    return index


def load_manifest(manifest_path, root):
    manifest_path = Path(manifest_path)
    rows = []
    if manifest_path.suffix.lower() == ".csv":
        with open(manifest_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    elif manifest_path.suffix.lower() in {".jsonl", ".json"}:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            if manifest_path.suffix.lower() == ".json":
                rows = json.load(fh)
            else:
                rows = [json.loads(line) for line in fh if line.strip()]
    else:
        raise ValueError("Unsupported manifest format. Use .csv, .jsonl, or .json")

    root = Path(root)
    items = []
    for row in rows:
        if not row:
            continue
        utt_id = row.get("utt_id") or row.get("id") or row.get("utt")
        gen_path = row.get("gen_path") or row.get("audio_path") or row.get("path")
        if not utt_id or not gen_path:
            raise ValueError("Manifest row missing utt_id or gen_path")
        gen_path = Path(gen_path)
        if not gen_path.is_absolute():
            gen_path = (manifest_path.parent / gen_path).resolve()
        gt_path = row.get("gt_path")
        src_path = row.get("src_path")
        ref_text = row.get("ref_text")
        if gt_path:
            gt_path = Path(gt_path)
            if not gt_path.is_absolute():
                gt_path = (manifest_path.parent / gt_path).resolve()
        if src_path:
            src_path = Path(src_path)
            if not src_path.is_absolute():
                src_path = (manifest_path.parent / src_path).resolve()
        items.append(EvalItem(utt_id=utt_id, gen_path=gen_path, gt_path=gt_path, src_path=src_path, ref_text=ref_text))
    return items


def scan_items(cfg):
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    gt_root = Path(data_cfg.get("gt_root") or root)
    src_root = Path(data_cfg.get("src_root") or root)
    exts = data_cfg["audio_extensions"]
    text_map = _load_text_map(data_cfg.get("gt_text_path"))

    if data_cfg.get("manifest"):
        items = load_manifest(data_cfg["manifest"], root)
        out = []
        for item in items:
            ref_text = item.ref_text or text_map.get(item.utt_id)
            out.append(EvalItem(
                utt_id=item.utt_id,
                gen_path=item.gen_path,
                gt_path=item.gt_path,
                src_path=item.src_path,
                ref_text=ref_text,
            ))
        return out

    gen_suffix = data_cfg["gen_suffix"]
    gt_suffix = data_cfg["gt_suffix"]
    src_suffix = data_cfg.get("src_suffix")

    items_by_id = {}
    gt_index = {}
    src_index = {}
    gt_index_built = False
    src_index_built = False
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    gen_iter = _iter_suffix_files(root, gen_suffix, exts, data_cfg["recursive"])
    if tqdm and data_cfg.get("scan_progress", False):
        gen_iter = tqdm(gen_iter, desc="扫描音频")
    for path in gen_iter:
        utt_id = _extract_utt_id(path, gen_suffix, exts)
        if not utt_id:
            continue
        if utt_id in items_by_id:
            continue
        rel_dir = None
        try:
            rel_dir = path.parent.relative_to(root)
        except ValueError:
            rel_dir = None
        gt_search_root = gt_root / rel_dir if rel_dir else gt_root
        src_search_root = src_root / rel_dir if rel_dir else src_root
        gt_path = _resolve_pair_path(gt_search_root, utt_id, gt_suffix, exts, prefer_ext=path.suffix)
        if gt_path is None and gt_search_root != gt_root:
            gt_path = _resolve_pair_path(gt_root, utt_id, gt_suffix, exts, prefer_ext=path.suffix)
        if gt_path is None:
            if not gt_index_built:
                gt_index = _build_suffix_index(gt_root, gt_suffix, exts, data_cfg["recursive"])
                gt_index_built = True
            gt_path = gt_index.get(utt_id)
        src_path = None
        if src_suffix:
            src_path = _resolve_pair_path(src_search_root, utt_id, src_suffix, exts, prefer_ext=path.suffix)
            if src_path is None and src_search_root != src_root:
                src_path = _resolve_pair_path(src_root, utt_id, src_suffix, exts, prefer_ext=path.suffix)
            if src_path is None:
                if not src_index_built:
                    src_index = _build_suffix_index(src_root, src_suffix, exts, data_cfg["recursive"])
                    src_index_built = True
                src_path = src_index.get(utt_id)
        ref_text = text_map.get(utt_id)
        items_by_id[utt_id] = EvalItem(
            utt_id=utt_id,
            gen_path=path,
            gt_path=gt_path,
            src_path=src_path,
            ref_text=ref_text,
        )

    return list(items_by_id.values())
