#!/usr/bin/env python3
"""Regenerate the OPSD listening comparison HTML from wav directories."""

from __future__ import annotations

import argparse
import html
from pathlib import Path


def column_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.name
    if name == "baseline_init":
        return (0, 0, name)
    prefix = "opsd_step"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            return (1, int(suffix), name)
    return (2, 0, name)


def load_rows(meta_file: Path, max_rows: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with meta_file.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            rows.append((parts[0], parts[3]))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    if not rows:
        raise RuntimeError(f"no rows loaded from {meta_file}")
    return rows


def discover_columns(compare_dir: Path) -> list[Path]:
    columns = [
        item
        for item in compare_dir.iterdir()
        if item.is_dir() or item.is_symlink()
    ]
    columns = [item for item in columns if item.name == "baseline_init" or item.name.startswith("opsd_step")]
    return sorted(columns, key=column_sort_key)


def step_summary(columns: list[Path]) -> str:
    labels: list[str] = []
    for column in columns:
        if column.name == "baseline_init":
            labels.append("baseline init")
        elif column.name.startswith("opsd_step"):
            labels.append("step" + column.name.removeprefix("opsd_step"))
        else:
            labels.append(column.name)
    return " vs ".join(labels)


def audio_cell(compare_dir: Path, column: Path, utt_id: str) -> str:
    rel_src = f"{column.name}/{utt_id}.wav"
    wav_path = compare_dir / rel_src
    if wav_path.exists():
        return f'<td><audio controls preload="none" src="{html.escape(rel_src)}"></audio></td>'
    return '<td class="missing">missing</td>'


def render_html(
    compare_dir: Path,
    meta_file: Path,
    output_file: Path,
    max_rows: int,
    prompt_note: str,
) -> str:
    rows = load_rows(meta_file, max_rows)
    columns = discover_columns(compare_dir)
    if not columns:
        raise RuntimeError(f"no listen columns found in {compare_dir}")

    title = f"OPSD Nonstream Listen Compare: {step_summary(columns)}"
    parts: list[str] = [
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>OPSD listen compare</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;}",
        "table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ddd;padding:8px;vertical-align:top;}",
        "th{background:#f5f5f5;position:sticky;top:0;}",
        "audio{width:240px;}",
        ".text{max-width:520px;line-height:1.4;}",
        ".utt{white-space:nowrap;font-family:monospace;}",
        ".meta{color:#444;}",
        ".missing{color:#a00;font-family:monospace;}",
        "</style>",
        "</head><body>",
        f"<h2>{html.escape(title)}</h2>",
        f'<p class="meta">{html.escape(prompt_note)}</p>',
        "<table><thead><tr><th>utt</th><th>target text</th>",
    ]
    for column in columns:
        parts.append(f"<th>{html.escape(column.name)}</th>")
    parts.append("</tr></thead><tbody>")

    for utt_id, text in rows:
        parts.append(
            f'<tr><td class="utt">{html.escape(utt_id)}</td>'
            f'<td class="text">{html.escape(text)}</td>'
        )
        for column in columns:
            parts.append(audio_cell(compare_dir, column, utt_id))
        parts.append("</tr>")

    parts.append("</tbody></table></body></html>")
    html_text = "\n".join(parts) + "\n"
    output_file.write_text(html_text, encoding="utf-8")
    return html_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-dir", type=Path, required=True)
    parser.add_argument("--meta-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument(
        "--prompt-note",
        default="Prompt: Achird_kefu_003.wav; mode: STREAM=0, BISTREAM_FIXED_RATIO=0; flow: epoch_4_whole.pt; eval rows: first 20.",
    )
    args = parser.parse_args()

    compare_dir = args.compare_dir.resolve()
    output_file = args.output_file or (compare_dir / "listen_compare.html")
    render_html(
        compare_dir=compare_dir,
        meta_file=args.meta_file.resolve(),
        output_file=output_file.resolve(),
        max_rows=args.max_rows,
        prompt_note=args.prompt_note,
    )
    print(f"[refresh-listen] wrote {output_file}")


if __name__ == "__main__":
    main()
