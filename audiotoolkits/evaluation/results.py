import csv
import statistics
from pathlib import Path


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def write_results_csv(rows, path):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    base_fields = ["utt_id", "gen_path", "gt_path", "src_path", "ref_text"]
    all_fields = set()
    for row in rows:
        all_fields.update(row.keys())
    other_fields = [f for f in sorted(all_fields) if f not in base_fields]
    fieldnames = [f for f in base_fields if f in all_fields] + other_fields

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows, extra_summary=None):
    values = {}
    for row in rows:
        for key, value in row.items():
            if _is_number(value):
                values.setdefault(key, []).append(float(value))
    summary = []
    for key, vals in values.items():
        if not vals:
            continue
        summary.append({
            "metric": key,
            "count": len(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        })
    if extra_summary:
        for key, value in extra_summary.items():
            summary.append({
                "metric": key,
                "count": 1,
                "mean": float(value),
                "median": float(value),
                "std": 0.0,
                "min": float(value),
                "max": float(value),
            })
    return summary


def write_summary_csv(rows, path):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = ["metric", "count", "mean", "median", "std", "min", "max"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
