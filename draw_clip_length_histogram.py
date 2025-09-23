from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import polars as pl

DEFAULT_SNAPSHOT = (
    "s3://synthesia-rnd-dataops-prd-datalake/dataset-snapshots/data/snapshot_89eeaaec_3ccd_4436_b48f_d54c05a83634"
)


def choose_duration_column(column_names: List[str]) -> Optional[str]:
    """Pick the most likely clip-duration column from available names."""
    preferred_order = [
        "c_duration_s",
        "duration_s",
        "clip_duration_s",
        "clip_length_s",
        "clip_len_s",
        "clip_len",
        "clip_length",
        "clip_duration",
        "duration",
        "length",
    ]

    lowered_to_original: Dict[str, str] = {name.lower(): name for name in column_names}
    for candidate in preferred_order:
        if candidate in lowered_to_original:
            return lowered_to_original[candidate]
    # Fallback: any column containing 'duration' or 'length'
    for name in column_names:
        lower = name.lower()
        if "duration" in lower or "length" in lower:
            return name
    return None


def load_duration_lazyframe(snapshot_path: str) -> tuple[pl.LazyFrame, str]:
    """Create a LazyFrame with a single column 'duration_s' from snapshot."""
    glob_path = f"{snapshot_path}/**/*.parquet"
    lf = pl.scan_parquet(glob_path, hive_partitioning=True)

    schema = lf.schema
    duration_col = choose_duration_column(list(schema.keys()))
    if duration_col is None:
        raise RuntimeError(f"Could not infer a duration/length column from snapshot schema: {list(schema.keys())}")

    duration_expr = pl.col(duration_col).cast(pl.Float64).alias("duration_s")
    lf_duration = lf.select(duration_expr).filter(pl.col("duration_s").is_not_null())
    return lf_duration, duration_col


def compute_total_hours(lf_duration: pl.LazyFrame) -> float:
    """Compute total hours across all clips in the LazyFrame."""
    result = lf_duration.select(pl.sum("duration_s").alias("total_seconds")).collect()
    total_seconds = result[0, 0] or 0.0
    return float(total_seconds) / 3600.0


def plot_histogram(
    durations: List[float],
    bins: int,
    title: str,
    max_duration: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    if max_duration is not None:
        durations = [d for d in durations if d <= max_duration]

    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=bins, color="#4C78A8", edgecolor="white")
    plt.xlabel("Clip length (seconds)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a dataset snapshot with Polars and plot clip length histogram.")
    parser.add_argument(
        "--snapshot",
        type=str,
        default=DEFAULT_SNAPSHOT,
        help="S3 or local path to snapshot root (containing parquet files).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional: filter out durations greater than this many seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=("Optional: limit number of rows sampled for plotting to avoid large memory use."),
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional: path to save the histogram image instead of showing it.",
    )
    parser.add_argument(
        "--print-hours",
        action="store_true",
        help="Print total hours across all clips (ignores --limit).",
    )
    parser.add_argument(
        "--hours-only",
        action="store_true",
        help="Only compute and print total hours, do not plot.",
    )

    args = parser.parse_args()

    lf_duration, chosen_col = load_duration_lazyframe(args.snapshot)

    # Compute on the full dataset if requested
    if args.print_hours or args.hours_only:
        total_hours = compute_total_hours(lf_duration)
        print(f"Total clip hours: {total_hours:.2f}")
        if args.hours_only:
            return

    if args.limit is not None and args.limit > 0:
        lf_duration = lf_duration.limit(args.limit)

    df = lf_duration.collect()
    durations = df.get_column("duration_s").to_list()

    title = f"Clip length histogram ({chosen_col})\n{args.snapshot}"
    plot_histogram(
        durations=durations,
        bins=args.bins,
        title=title,
        max_duration=args.max_duration,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
