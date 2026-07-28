#!/usr/bin/env python3
"""Batch-run AudioToolkits eval over whispervc output directories.

Discovers leaf directories containing *_gen.wav under egs/output and runs
`python -m audiotoolkits.evaluation.run` sequentially for each directory.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_OUTPUT_ROOT = Path("/SMIIPdata2/yinzy/whispervc/egs/output")
DEFAULT_CONFIG = Path("/home/yinzy/AudioToolkits/w2n_eval_metrics_offline.yaml")
DEFAULT_REPO_ROOT = Path("/home/yinzy/AudioToolkits")
DEFAULT_RESULT_NAME = "metrics_result_offline"
EXCLUDED_DIR_NAMES = {
    "intermediate",
    "cer_analysis",
    "_eval_tmux_logs",
    "_batch_eval_logs",
}


@dataclass(frozen=True)
class Task:
    root: Path
    rel: str
    output_dir: Path
    gen_count: int
    gt_count: int
    src_count: int

    @property
    def slug(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "__", self.rel.strip("/"))
        return slug[:220] or "root"


def _is_excluded_path(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDED_DIR_NAMES:
            return True
        if part.startswith("metrics_result"):
            return True
    return False


def _count_suffix(directory: Path, suffix: str) -> int:
    return sum(1 for _ in directory.glob(f"*{suffix}.wav"))


def _compile_patterns(values: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(value) for value in values]


def _matches_any(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def discover_tasks(
    output_root: Path,
    result_name: str,
    include: list[re.Pattern[str]],
    exclude: list[re.Pattern[str]],
) -> list[Task]:
    dirs: set[Path] = set()
    for wav in output_root.rglob("*_gen.wav"):
        if _is_excluded_path(wav):
            continue
        dirs.add(wav.parent)

    tasks: list[Task] = []
    for directory in sorted(dirs, key=lambda p: str(p)):
        rel = str(directory.relative_to(output_root))
        if include and not _matches_any(include, rel):
            continue
        if exclude and _matches_any(exclude, rel):
            continue
        gen_count = _count_suffix(directory, "_gen")
        gt_count = _count_suffix(directory, "_gt")
        src_count = _count_suffix(directory, "_whisper")
        if gen_count <= 0:
            continue
        if gt_count <= 0:
            print(f"[WARN] skip no-gt directory: {directory}", file=sys.stderr)
            continue
        tasks.append(
            Task(
                root=directory,
                rel=rel,
                output_dir=directory / result_name,
                gen_count=gen_count,
                gt_count=gt_count,
                src_count=src_count,
            )
        )
    return tasks


def result_status(task: Task) -> tuple[str, str]:
    results_csv = task.output_dir / "results.csv"
    summary_csv = task.output_dir / "summary.csv"
    if not task.output_dir.exists():
        return "missing", "output_dir_missing"
    if not results_csv.exists() or not summary_csv.exists():
        return "incomplete", "missing_results_or_summary"
    try:
        with results_csv.open("r", encoding="utf-8", newline="") as fh:
            line_count = sum(1 for _ in fh)
    except OSError as exc:
        return "incomplete", f"cannot_read_results:{exc}"
    expected = task.gen_count + 1
    if line_count != expected:
        return "incomplete", f"results_lines={line_count},expected={expected}"
    return "complete", f"results_lines={line_count}"


def metric_coverage(results_csv: Path, metrics: list[str]) -> dict[str, tuple[int, int]]:
    coverage: dict[str, tuple[int, int]] = {}
    if not results_csv.exists():
        return coverage
    with results_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    total = len(rows)
    for metric in metrics:
        if metric not in (reader.fieldnames or []):
            continue
        non_empty = sum(1 for row in rows if row.get(metric) not in (None, "", "nan", "NaN"))
        coverage[metric] = (non_empty, total)
    return coverage


def build_command(python_exe: str, config: Path, task: Task) -> list[str]:
    return [
        python_exe,
        "-m",
        "audiotoolkits.evaluation.run",
        "--config",
        str(config),
        "--root",
        str(task.root),
        "--output-dir",
        str(task.output_dir),
    ]


def run_task(
    task: Task,
    args: argparse.Namespace,
    log_dir: Path,
    task_index: int,
    task_total: int,
) -> int:
    status, detail = result_status(task)
    if status == "complete" and not args.force:
        print(f"[SKIP] {task_index}/{task_total} {task.rel} ({detail})")
        return 0
    if status == "incomplete" and not args.force:
        print(
            f"[ERROR] Refusing to overwrite incomplete result without --force: {task.output_dir} ({detail})",
            file=sys.stderr,
        )
        return 2

    task.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task_index:03d}_{task.slug}.log"
    cmd = build_command(args.python, args.config, task)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.force:
        # The eval config has overwrite:false. Remove only final CSVs in this output dir
        # so intermediate caches can still be reused.
        for name in ("results.csv", "summary.csv"):
            target = task.output_dir / name
            if target.exists():
                target.unlink()

    print(f"[RUN ] {task_index}/{task_total} {task.rel}")
    print(f"       root: {task.root}")
    print(f"       out : {task.output_dir}")
    print(f"       log : {log_path}")
    print(f"       cmd : {' '.join(shlex.quote(part) for part in cmd)}")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"# task {task_index}/{task_total}: {task.rel}\n")
        log_fh.write(f"# root: {task.root}\n")
        log_fh.write(f"# output: {task.output_dir}\n")
        log_fh.write(f"# command: {' '.join(shlex.quote(part) for part in cmd)}\n\n")
        log_fh.flush()
        proc = subprocess.run(cmd, cwd=args.repo_root, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    if proc.returncode != 0:
        print(f"[FAIL] {task.rel} returncode={proc.returncode} elapsed={elapsed:.1f}s log={log_path}", file=sys.stderr)
        return proc.returncode

    status, detail = result_status(task)
    if status != "complete":
        print(f"[FAIL] {task.rel} finished but result is {status}: {detail}", file=sys.stderr)
        return 3

    coverage = metric_coverage(task.output_dir / "results.csv", args.check_metrics)
    coverage_text = ", ".join(f"{k}={v[0]}/{v[1]}" for k, v in coverage.items())
    if coverage_text:
        print(f"[DONE] {task.rel} elapsed={elapsed:.1f}s {coverage_text}")
    else:
        print(f"[DONE] {task.rel} elapsed={elapsed:.1f}s {detail}")

    if args.strict_metrics:
        bad = [f"{k}={v[0]}/{v[1]}" for k, v in coverage.items() if v[0] != v[1]]
        allowed = set(args.allow_partial_metric)
        bad = [item for item in bad if item.split("=", 1)[0] not in allowed]
        if bad:
            print(f"[FAIL] incomplete metric coverage: {', '.join(bad)}", file=sys.stderr)
            return 4
    return 0


def write_manifest(tasks: list[Task], log_dir: Path) -> Path:
    manifest = log_dir / "tasks.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["index", "rel", "root", "output_dir", "gen_count", "gt_count", "src_count", "status", "detail"])
        for idx, task in enumerate(tasks, 1):
            status, detail = result_status(task)
            writer.writerow([idx, task.rel, task.root, task.output_dir, task.gen_count, task.gt_count, task.src_count, status, detail])
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--result-name", default=DEFAULT_RESULT_NAME)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--include", action="append", default=[], help="Regex over path relative to output root. May repeat.")
    parser.add_argument("--exclude", action="append", default=[], help="Regex over path relative to output root. May repeat.")
    parser.add_argument("--start-at", type=int, default=1, help="1-based task index to start from.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of tasks to run after filtering/start-at.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if final CSVs exist; removes only results.csv and summary.csv in the target output dir.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES for every eval subprocess. Leave unset when config uses physical cuda:N ids.")
    parser.add_argument("--check-metrics", nargs="*", default=["cer", "utmos", "wvmos", "wavlm_sim", "nisqa", "speechbertscore"])
    parser.add_argument("--strict-metrics", action="store_true", help="Fail a task if any checked metric is not complete.")
    parser.add_argument("--allow-partial-metric", action="append", default=["wavlm_sim"], help="Metric allowed to be partial under --strict-metrics. May repeat.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.config = args.config.resolve()
    args.repo_root = args.repo_root.resolve()
    if args.log_dir is None:
        args.log_dir = args.output_root / "_batch_eval_logs" / args.result_name
    args.log_dir.mkdir(parents=True, exist_ok=True)

    include = _compile_patterns(args.include)
    exclude = _compile_patterns(args.exclude)
    tasks = discover_tasks(args.output_root, args.result_name, include, exclude)
    if args.start_at < 1:
        print("[ERROR] --start-at must be >= 1", file=sys.stderr)
        return 2
    if args.start_at > 1:
        tasks = tasks[args.start_at - 1:]
    if args.limit is not None:
        tasks = tasks[:args.limit]

    manifest = write_manifest(tasks, args.log_dir)
    print(f"[INFO] tasks: {len(tasks)}")
    print(f"[INFO] manifest: {manifest}")
    print(f"[INFO] config: {args.config}")
    print(f"[INFO] result_name: {args.result_name}")
    if args.cuda_visible_devices:
        print(f"[INFO] CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
    print("[INFO] PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    for idx, task in enumerate(tasks, args.start_at):
        status, detail = result_status(task)
        print(
            f"[TASK] {idx}: {task.rel} gen={task.gen_count} gt={task.gt_count} "
            f"src={task.src_count} status={status} ({detail})"
        )
    if args.dry_run:
        print("[INFO] dry-run only; no eval command executed.")
        return 0

    total = args.start_at + len(tasks) - 1
    for idx, task in enumerate(tasks, args.start_at):
        code = run_task(task, args, args.log_dir, idx, total)
        if code != 0:
            print(f"[STOP] stopped at task {idx}: {task.rel}", file=sys.stderr)
            return code
    print("[ALL DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
