import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _read_eval_rows(csv_path):
    with open(csv_path, mode="r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")
    return rows


def _column_stats(rows, column_name):
    values = np.array([_to_float(row.get(column_name)) for row in rows], dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError(f"Column '{column_name}' contains no numeric values")
    return float(values.mean())


def summarize_eval_csv(csv_path):
    rows = _read_eval_rows(csv_path)
    fieldnames = rows[0].keys()
    required = [
        "steps",
        "steps_in_target_lane",
        "avg_speed_diff",
        "speed_deviation_ratio",
        "fail_mode",
    ]
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

    steps = _column_stats(rows, "steps")
    steps_in_target_lane = _column_stats(rows, "steps_in_target_lane")
    avg_speed_diff = _column_stats(rows, "avg_speed_diff")
    speed_deviation_ratio = _column_stats(rows, "speed_deviation_ratio")
    fail_modes = np.array([_to_int(row.get("fail_mode")) for row in rows], dtype=np.int64)
    total = len(fail_modes)
    success_pct = float(np.mean(fail_modes == 0) * 100.0)
    collision_pct = float(np.mean(fail_modes == 1) * 100.0)
    oob_pct = float(np.mean(fail_modes == 2) * 100.0)

    return {
        "name": Path(csv_path).stem,
        "steps": steps,
        "steps_in_target_lane": steps_in_target_lane,
        "avg_speed_diff": avg_speed_diff,
        "speed_deviation_ratio": speed_deviation_ratio,
        "success_pct": success_pct,
        "collision_pct": collision_pct,
        "oob_pct": oob_pct,
        "count": total,
    }


def _write_tracker(results, tracker_path):
    lines = []
    for result in results:
        lines.append(
            f"{result['name']}:\n"
            f"avg steps: {result['steps']:.4f} ; "
            f"avg steps in lane: {result['steps_in_target_lane']:.4f} ; "
            f"avg speed dev: {result['avg_speed_diff']:.4f} ; "
            f"avg steps speed dev: {result['speed_deviation_ratio']:.4f} ; "
            f"success: {result['success_pct']:.2f}% / col: {result['collision_pct']:.2f}% / oob: {result['oob_pct']:.2f}%"
        )
    tracker_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bar_chart(results, key, title, ylabel, out_path):
    names = [result["name"] for result in results]
    values = [result[key] for result in results]

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.6), 5))
    ax.bar(names, values, color="#4472C4")
    ax.set_title(title)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _bar_chart_failures(results, out_path):
    names = [result["name"] for result in results]
    x = np.arange(len(names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.8), 5))
    ax.bar(x - width, [result["success_pct"] for result in results], width=width, label="Success")
    ax.bar(x, [result["collision_pct"] for result in results], width=width, label="Collision")
    ax.bar(x + width, [result["oob_pct"] for result in results], width=width, label="OOB")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_title("Success / Failure Breakdown")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Percentage")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize and compare evaluation CSV logs.")
    parser.add_argument("csvs", nargs="+", help="One or more evaluation CSV files.")
    parser.add_argument("--comp", action="store_true", help="Generate comparison charts across all provided CSVs.")
    parser.add_argument("--outdir", type=Path, default=Path("eval_plots"), help="Directory for comparison charts.")
    parser.add_argument("--tracker", type=Path, default=Path("eval_tracker.txt"), help="Output text file for summary lines.")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_paths = [Path(csv_path + ".csv") for csv_path in args.csvs]
    for csv_path in csv_paths:
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    results = [summarize_eval_csv(csv_path) for csv_path in csv_paths]
    _write_tracker(results, args.tracker)

    for result in results:
        print(
            f"{result['name']}: avg steps={result['steps']:.4f}, "
            f"avg lane steps={result['steps_in_target_lane']:.4f}, "
            f"avg speed dev={result['avg_speed_diff']:.4f}, "
            f"avg steps speed dev={result['speed_deviation_ratio']:.4f}, "
            f"success={result['success_pct']:.2f}%, col={result['collision_pct']:.2f}%, oob={result['oob_pct']:.2f}%"
        )

    if args.comp and len(results) > 1:
        args.outdir.mkdir(parents=True, exist_ok=True)
        _bar_chart(results, "steps", "Average Steps Taken", "Steps", args.outdir / "avg_steps.png")
        _bar_chart(results, "steps_in_target_lane", "Average Steps in Target Lane", "Steps in Target Lane", args.outdir / "avg_steps_in_target_lane.png")
        _bar_chart(results, "avg_speed_diff", "Average Speed Deviation", "Speed Deviation", args.outdir / "avg_speed_deviation.png")
        _bar_chart(results, "speed_deviation_ratio", "Average Steps Deviating", "Deviation Ratio", args.outdir / "avg_steps_deviating.png")
        _bar_chart_failures(results, args.outdir / "success_collision_oob.png")


if __name__ == "__main__":
    main()