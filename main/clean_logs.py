#!/usr/bin/env python3
"""
Remove rows that are out of order with respect to neighboring rows.

A row is removed if its selected order key is strictly smaller than both
the previous and next row values for that same key.

Duplicate keys are also removed: only the first appearance of each selected
order key is kept.

Key selection:
- episode logs -> episode
- training logs -> step (or steps)

Example:
    python clean_logs.py --input clamped_td3_episode_log.csv --output cleaned.csv
"""

import argparse
import csv
from pathlib import Path


def is_strict_local_min(prev_val, curr_val, next_val):
    return curr_val < prev_val and curr_val < next_val


def parse_int_like(value):
    return int(float(value))


def pick_order_column(input_path: Path, fieldnames, by_mode: str):
    has_episode = "episode" in fieldnames
    step_col = "step" if "step" in fieldnames else ("steps" if "steps" in fieldnames else None)

    if by_mode == "episode":
        if not has_episode:
            raise ValueError("Requested --by episode but CSV has no 'episode' column.")
        return "episode"

    if by_mode == "step":
        if step_col is None:
            raise ValueError("Requested --by step but CSV has neither 'step' nor 'steps' column.")
        return step_col

    # auto mode: prefer filename intent first, then fallback to available columns.
    stem = input_path.stem.lower()
    if "episode" in stem and has_episode:
        return "episode"
    if "training" in stem and step_col is not None:
        return step_col
    if has_episode and step_col is None:
        return "episode"
    if step_col is not None and not has_episode:
        return step_col
    if has_episode:
        return "episode"

    raise ValueError("Could not infer order key. Need 'episode' or 'step/steps' column.")


def clean_csv(input_path: Path, output_path: Path, by_mode: str):
    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError("CSV has no header.")

    order_col = pick_order_column(input_path, fieldnames, by_mode)

    cleaned = []
    seen_keys = set()
    n = len(rows)
    local_min_removed = 0
    duplicate_removed = 0

    for i, row in enumerate(rows):
        curr_key = parse_int_like(row[order_col])

        # Keep only first appearance of each key.
        if curr_key in seen_keys:
            duplicate_removed += 1
            continue

        if i == 0 or i == n - 1:
            seen_keys.add(curr_key)
            cleaned.append(row)
            continue

        prev_row = rows[i - 1]
        next_row = rows[i + 1]

        prev_val = parse_int_like(prev_row[order_col])
        curr_val = curr_key
        next_val = parse_int_like(next_row[order_col])

        if is_strict_local_min(prev_val, curr_val, next_val):
            local_min_removed += 1
            continue

        seen_keys.add(curr_key)
        cleaned.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned)

    print(
        f"[{order_col}] Read {len(rows)} rows, wrote {len(cleaned)} rows to {output_path} "
        f"(removed {local_min_removed} local-min rows, {duplicate_removed} duplicates)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path for cleaned data")
    parser.add_argument(
        "--by",
        choices=["auto", "episode", "step"],
        default="auto",
        help="Order key to clean on (default: auto).",
    )
    args = parser.parse_args()

    clean_csv(Path(args.input), Path(args.output), args.by)


if __name__ == "__main__":
    main()
