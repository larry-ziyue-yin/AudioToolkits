#!/usr/bin/env python3
"""Build denoising-line evaluation tables, including raw baseline outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


SUMMARY_METRICS = [
    "cer",
    "overall_cer",
    "wer",
    "overall_wer",
    "dnsmos_ovrl",
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_p808",
    "secs",
    "utmos",
    "wvmos",
    "wespeaker_sim",
    "wavlm_sim",
    "nisqa",
    "speechbertscore",
]

AUDIO_METRICS = [
    "cer",
    "wer",
    "dnsmos_ovrl",
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_p808",
    "secs",
    "utmos",
    "wvmos",
    "wespeaker_sim",
    "wavlm_sim",
    "nisqa",
    "speechbertscore",
]

LOWER_IS_BETTER = {"cer", "overall_cer", "wer", "overall_wer"}


@dataclass(frozen=True)
class EvalItem:
    label: str
    experiment: str
    variant: str
    condition: str
    snr_db: str
    condition_path: Path
    summary_path: Path
    results_path: Path


def metric_direction(metric: str) -> str:
    return "lower" if metric in LOWER_IS_BETTER else "higher"


def parse_raw_condition(name: str) -> tuple[str, str] | None:
    if "audioOnly_full" in name and "real_VC" in name:
        return "clean", ""
    markers = {
        "snr0_": ("snr_0", "0"),
        "snr10_": ("snr_10", "10"),
        "snr20_": ("snr_20", "20"),
        "snr_m5_": ("snr_m5", "-5"),
    }
    for marker, value in markers.items():
        if marker in name:
            return value
    return None


def parse_snr(condition: str) -> str:
    if condition == "clean":
        return ""
    if condition.startswith("snr_m"):
        return "-" + condition.removeprefix("snr_m")
    if condition.startswith("snr_"):
        return condition.removeprefix("snr_")
    return ""


def discover_items(flow_root: Path, denoised_root: Path) -> list[EvalItem]:
    items: list[EvalItem] = []

    for summary_path in sorted(flow_root.glob("AISHELL_6_*/metrics_result_offline/summary.csv")):
        condition_info = parse_raw_condition(summary_path.parents[1].name)
        if condition_info is None:
            continue
        condition, snr_db = condition_info
        results_path = summary_path.with_name("results.csv")
        if not results_path.exists():
            continue
        items.append(
            EvalItem(
                label="raw_baseline",
                experiment="raw_baseline",
                variant="raw_baseline",
                condition=condition,
                snr_db=snr_db,
                condition_path=summary_path.parents[1],
                summary_path=summary_path,
                results_path=results_path,
            )
        )

    for summary_path in sorted(denoised_root.glob("*/*/metrics_result_offline/summary.csv")):
        if "_batch_eval_logs" in summary_path.parts:
            continue
        results_path = summary_path.with_name("results.csv")
        if not results_path.exists():
            continue
        condition_path = summary_path.parents[1]
        condition = condition_path.name
        label = condition_path.parent.name
        items.append(
            EvalItem(
                label=label,
                experiment=label,
                variant=label,
                condition=condition,
                snr_db=parse_snr(condition),
                condition_path=condition_path,
                summary_path=summary_path,
                results_path=results_path,
            )
        )

    return sorted(items, key=lambda x: (x.condition, x.label))


def read_summary(item: EvalItem) -> dict[str, object]:
    df = pd.read_csv(item.summary_path)
    by_metric = {str(row["metric"]): row for _, row in df.iterrows()}
    results_rows = len(pd.read_csv(item.results_path, usecols=["utt_id"]))
    row: dict[str, object] = {
        "denoise_model": item.label,
        "experiment": item.experiment,
        "variant": item.variant,
        "condition": item.condition,
        "snr_db": item.snr_db,
        "condition_path": str(item.condition_path),
        "summary_path": str(item.summary_path),
        "results_path": str(item.results_path),
        "has_summary_csv": "yes",
        "has_results_csv": "yes",
        "result_rows": results_rows,
    }
    for metric in SUMMARY_METRICS:
        metric_row = by_metric.get(metric)
        for stat in ["count", "mean", "median", "std", "min", "max"]:
            row[f"{metric}_{stat}"] = metric_row.get(stat, pd.NA) if metric_row is not None else pd.NA
    return row


def add_condition_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in SUMMARY_METRICS:
        col = f"{metric}_mean"
        if col not in out.columns:
            continue
        ascending = metric_direction(metric) == "lower"
        out[f"{metric}_rank"] = (
            out.groupby("condition")[col]
            .rank(method="min", ascending=ascending, na_option="bottom")
            .astype("Int64")
        )
    return out


def build_metric_rows(summary_df: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in summary_df.iterrows():
        for metric in SUMMARY_METRICS:
            rows.append(
                {
                    "snapshot_time": snapshot,
                    "condition": row["condition"],
                    "metric": metric,
                    "direction": metric_direction(metric),
                    "rank": row.get(f"{metric}_rank", pd.NA),
                    "value_mean": row.get(f"{metric}_mean", pd.NA),
                    "value_median": row.get(f"{metric}_median", pd.NA),
                    "value_std": row.get(f"{metric}_std", pd.NA),
                    "value_count": row.get(f"{metric}_count", pd.NA),
                    "denoise_model": row["denoise_model"],
                    "experiment": row["experiment"],
                    "variant": row["variant"],
                    "condition_path": row["condition_path"],
                    "summary_path": row["summary_path"],
                }
            )
    return pd.DataFrame(rows).sort_values(["condition", "metric", "rank", "denoise_model"])


def build_audio_rows(items: list[EvalItem], snapshot: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    usecols = ["utt_id", "gen_path", "gt_path", "src_path", "ref_text"] + AUDIO_METRICS
    for item in items:
        df = pd.read_csv(item.results_path)
        cols = [col for col in usecols if col in df.columns]
        df = df[cols].copy()
        df.insert(0, "snapshot_time", snapshot)
        df.insert(1, "condition", item.condition)
        df.insert(3, "speaker", df["utt_id"].astype(str).str.extract(r"^(S\d+)", expand=False))
        df.insert(4, "denoise_model", item.label)
        df.insert(5, "experiment", item.experiment)
        df.insert(6, "variant", item.variant)
        for metric in AUDIO_METRICS:
            if metric not in df.columns:
                df[metric] = pd.NA
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    ordered = [
        "snapshot_time",
        "condition",
        "utt_id",
        "speaker",
        "denoise_model",
        "experiment",
        "variant",
        "gen_path",
        "gt_path",
        "src_path",
        "ref_text",
    ] + AUDIO_METRICS
    return pd.concat(frames, ignore_index=True)[ordered]


def build_audio_rank_rows(audio_df: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base_cols = [
        "condition",
        "utt_id",
        "speaker",
        "denoise_model",
        "experiment",
        "variant",
        "gen_path",
        "gt_path",
        "src_path",
    ]
    for metric in AUDIO_METRICS:
        if metric not in audio_df.columns:
            continue
        sub = audio_df[base_cols + [metric]].dropna(subset=[metric]).copy()
        if sub.empty:
            continue
        direction = metric_direction(metric)
        sub["snapshot_time"] = snapshot
        sub["metric"] = metric
        sub["direction"] = direction
        sub["value"] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.dropna(subset=["value"])
        sub["rank"] = (
            sub.groupby(["condition", "utt_id"])["value"]
            .rank(method="min", ascending=(direction == "lower"))
            .astype("Int64")
        )
        rows.append(
            sub[
                [
                    "snapshot_time",
                    "condition",
                    "utt_id",
                    "speaker",
                    "metric",
                    "direction",
                    "rank",
                    "value",
                    "denoise_model",
                    "experiment",
                    "variant",
                    "gen_path",
                    "gt_path",
                    "src_path",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["condition", "utt_id", "metric", "rank", "denoise_model"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow-root",
        type=Path,
        default=Path("/SMIIPdata2/yinzy/whispervc/egs/output/flow_matching_based"),
    )
    parser.add_argument(
        "--denoised-root",
        type=Path,
        default=Path("/SMIIPdata2/yinzy/whispervc/egs/output/flow_matching_based/denoised_vc_eval"),
    )
    args = parser.parse_args()

    snapshot = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    items = discover_items(args.flow_root, args.denoised_root)
    summary_df = add_condition_ranks(pd.DataFrame([read_summary(item) for item in items]))
    summary_df.insert(0, "snapshot_time", snapshot)
    metric_df = build_metric_rows(summary_df, snapshot)
    audio_df = build_audio_rows(items, snapshot)
    audio_rank_df = build_audio_rank_rows(audio_df, snapshot)

    args.denoised_root.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.denoised_root / "summary_comparison_by_condition.csv", index=False)
    metric_df.to_csv(args.denoised_root / "summary_comparison_by_metric.csv", index=False)
    metric_df.to_csv(args.denoised_root / "metric_ranking_by_condition.csv", index=False)
    audio_df.to_csv(args.denoised_root / "raw_inference_comparison_by_utt.csv", index=False)
    audio_rank_df.to_csv(args.denoised_root / "audio_metric_ranking_by_utt.csv", index=False)

    print(f"items={len(items)}")
    print(f"summary_rows={len(summary_df)}")
    print(f"metric_rows={len(metric_df)}")
    print(f"audio_rows={len(audio_df)}")
    print(f"audio_rank_rows={len(audio_rank_df)}")
    for item in items:
        print(f"{item.condition}\t{item.label}\t{item.results_path}")


if __name__ == "__main__":
    main()
