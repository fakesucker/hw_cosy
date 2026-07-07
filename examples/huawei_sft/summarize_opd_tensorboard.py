#!/usr/bin/env python3
"""Summarize OPD distillation TensorBoard scalars.

The trainer writes OPD losses and overlap metrics as scalar tags such as
TRAIN/external_kd_loss and TRAIN/online_external_kd_topk_overlap. This helper
turns one or more TensorBoard event directories into TSV files that are easy to
diff across baseline, forced, online, and hybrid runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover - exercised by runtime environments
    raise SystemExit("tensorboard is required to read event files: {}".format(exc))


DEFAULT_INCLUDE = [
    "loss",
    "acc",
    "ce_loss",
    "speech_ce_loss",
    "kd_loss",
    "kd_weighted",
    "kd_top1_agree",
    "kd_topk_overlap",
    "kd_token_count",
    "online_sample_token_count",
    "online_sample_batch_size",
    "lr",
    "grad_norm",
]


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def find_event_dirs(paths: Sequence[Path]) -> List[Path]:
    event_dirs = []
    seen = set()
    for path in paths:
        if path.is_file() and path.name.startswith("events.out.tfevents."):
            directory = path.parent
            if directory not in seen:
                event_dirs.append(directory)
                seen.add(directory)
            continue
        if not path.exists():
            raise FileNotFoundError(str(path))
        if any(child.name.startswith("events.out.tfevents.") for child in path.iterdir() if child.is_file()):
            if path not in seen:
                event_dirs.append(path)
                seen.add(path)
            continue
        for event_file in path.rglob("events.out.tfevents.*"):
            directory = event_file.parent
            if directory not in seen:
                event_dirs.append(directory)
                seen.add(directory)
    return sorted(event_dirs)


def run_name(directory: Path, root_hint: Path | None) -> str:
    if root_hint is not None:
        try:
            return str(directory.relative_to(root_hint))
        except ValueError:
            pass
    return str(directory)


def load_scalars(directory: Path) -> Dict[str, list]:
    accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    return {tag: accumulator.Scalars(tag) for tag in tags}


def tag_selected(tag: str, include: Sequence[str], exclude: Sequence[str]) -> bool:
    if include and not any(pattern in tag for pattern in include):
        return False
    if exclude and any(pattern in tag for pattern in exclude):
        return False
    return True


def scalar_summary(events: list) -> Tuple[int, int, int, float, float, float, float, float, float, float]:
    steps = [int(event.step) for event in events]
    values = [float(event.value) for event in events]
    first_step, last_step = steps[0], steps[-1]
    first_value, last_value = values[0], values[-1]
    delta = last_value - first_value
    step_delta = max(last_step - first_step, 1)
    slope = delta / step_delta
    return (
        len(events),
        first_step,
        last_step,
        first_value,
        last_value,
        min(values),
        max(values),
        mean(values),
        delta,
        slope,
    )


def dedupe_events(events: list, mode: str) -> list:
    if mode == "none":
        return events
    if mode != "last":
        raise ValueError("unsupported dedupe mode {}".format(mode))
    latest_by_step = {}
    for event in events:
        previous = latest_by_step.get(int(event.step))
        if previous is None or float(event.wall_time) >= float(previous.wall_time):
            latest_by_step[int(event.step)] = event
    return [latest_by_step[step] for step in sorted(latest_by_step)]


def write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        fout.write("\t".join(header) + "\n")
        for row in rows:
            fout.write("\t".join(str(item) for item in row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OPD TensorBoard scalar metrics")
    parser.add_argument("--input", action="append", required=True, help="TensorBoard event file or directory")
    parser.add_argument("--output_dir", default="", help="Directory for summary.tsv and curves.tsv")
    parser.add_argument(
        "--include",
        default=",".join(DEFAULT_INCLUDE),
        help="Comma-separated tag substrings to include; empty includes all scalar tags",
    )
    parser.add_argument("--exclude", default="", help="Comma-separated tag substrings to exclude")
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="Required tag substring. Can be passed multiple times; missing substrings make the command fail.",
    )
    parser.add_argument(
        "--dedupe_steps",
        default="last",
        choices=["last", "none"],
        help="How to handle duplicate scalar points at the same global step in a run/tag.",
    )
    parser.add_argument("--curves", action="store_true", help="Write per-step scalar rows to curves.tsv")
    args = parser.parse_args()

    input_paths = [Path(item).resolve() for item in args.input]
    root_hint = input_paths[0] if len(input_paths) == 1 and input_paths[0].is_dir() else None
    include = parse_csv(args.include)
    exclude = parse_csv(args.exclude)
    required = [item for value in args.require for item in parse_csv(value)]

    event_dirs = find_event_dirs(input_paths)
    if not event_dirs:
        raise SystemExit("no TensorBoard event files found under {}".format(", ".join(map(str, input_paths))))

    summary_rows = []
    curve_rows = []
    seen_tags = set()
    for directory in event_dirs:
        run = run_name(directory, root_hint)
        scalars = load_scalars(directory)
        for tag, events in sorted(scalars.items()):
            if not events or not tag_selected(tag, include, exclude):
                continue
            events = dedupe_events(events, args.dedupe_steps)
            if not events:
                continue
            seen_tags.add(tag)
            summary_rows.append((run, tag, *scalar_summary(events)))
            if args.curves:
                for event in events:
                    curve_rows.append((run, tag, int(event.step), float(event.value), float(event.wall_time)))

    missing_required = [pattern for pattern in required if not any(pattern in tag for tag in seen_tags)]
    if missing_required:
        raise SystemExit("missing required scalar tag substring(s): {}".format(", ".join(missing_required)))

    header = [
        "run",
        "tag",
        "count",
        "first_step",
        "last_step",
        "first_value",
        "last_value",
        "min",
        "max",
        "mean",
        "delta",
        "slope_per_step",
    ]

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        summary_path = output_dir / "summary.tsv"
        write_tsv(summary_path, header, summary_rows)
        print("summary_tsv={}".format(summary_path))
        if args.curves:
            curves_path = output_dir / "curves.tsv"
            write_tsv(curves_path, ["run", "tag", "step", "value", "wall_time"], curve_rows)
            print("curves_tsv={}".format(curves_path))
    else:
        print("\t".join(header))
        for row in summary_rows:
            print("\t".join(str(item) for item in row))


if __name__ == "__main__":
    main()
