#!/usr/bin/env python3
"""Convert 测评对话.xlsx to CosyVoice meta.lst for inference.

Output format (same as kefu_0506_onlymale.lst):
  id|prompt_text|prompt_wav_path|tts_text|[caption]

Sheets:
  - 错词率评测集 / 评测集一 -> ceping_cer_wer.lst (Seed-TTS-Eval 单句，测试集1)
  - 错词率评测集 / 评测集二 -> ceping_cer_wer_set2.lst（总体评测样例中客服+用户全话术，测试集2）
  - 总体评测样例 -> ceping_mos_dialog.lst (主观 MOS，仅客服话术)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_PROMPT_TEXT = (
    "嗯非常抱歉给您带来麻烦了先那这样我这边马上安排骑手给您尽快配送"
    "嗯骑手接单情况下我会马上以短信的形式告知给您。"
)
DEFAULT_PROMPT_WAV = (
    "/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/"
    "huawei_streaming_cosyvoice/kefu_test/prompt_wav/03729.wav"
)

SKIP_WER_LABELS = {
    "评测集一",
    "Seed-TTS-Eval 中文",
    "总体测评集中，客服+用户",
}


def _norm_text(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_wer_texts(xlsx: Path) -> list[str]:
    df = pd.read_excel(xlsx, sheet_name="错词率评测集")
    texts: list[str] = []
    for raw in df["评测集一"].dropna():
        t = _norm_text(raw)
        if not t or t in SKIP_WER_LABELS:
            continue
        texts.append(t)
    return texts


def parse_dialog_turns(dialog: str) -> list[tuple[str, str]]:
    """Extract (role, text) turns from a dialog; role is 客服 or 用户."""
    dialog = str(dialog).strip()
    if not dialog or dialog.lower() == "nan":
        return []
    parts = re.split(r"(?=^(?:用户|客服)[：:])", dialog, flags=re.MULTILINE)
    turns: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(用户|客服)[：:]\s*(.+)$", part, flags=re.DOTALL)
        if not m:
            continue
        role = m.group(1)
        text = _norm_text(m.group(2))
        if text:
            turns.append((role, text))
    return turns


def parse_kefu_utterances(dialog: str) -> list[str]:
    return [text for role, text in parse_dialog_turns(dialog) if role == "客服"]


def load_mos_dialogs(xlsx: Path, kefu_only: bool = True) -> list[dict]:
    df = pd.read_excel(xlsx, sheet_name="总体评测样例")
    dialogs: list[dict] = []
    for _i, row in df.iterrows():
        sample = row.get("样例")
        if kefu_only:
            turns = [( "客服", t) for t in parse_kefu_utterances(sample)]
        else:
            turns = parse_dialog_turns(sample)
        if not turns:
            continue
        industry = row.get("行业")
        scene = row.get("场景")
        dim = row.get("维度")
        meta = []
        if pd.notna(industry):
            meta.append(f"行业={industry}")
        if pd.notna(scene):
            meta.append(f"场景={scene}")
        if pd.notna(dim):
            meta.append(f"维度={dim}")
        dialogs.append(
            {
                "dialog_idx": len(dialogs),
                "turns": turns,
                "caption_prefix": "；".join(meta) if meta else "总体评测样例",
            }
        )
    return dialogs


def write_lst(
    path: Path,
    rows: list[tuple[str, str, str, str, str | None]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample_id, prompt_text, prompt_wav, tts_text, caption in rows:
            line = f"{sample_id}|{prompt_text}|{prompt_wav}|{tts_text}"
            if caption:
                line += f"|{caption}"
            f.write(line + "\n")


def build_wer_set1_rows(
    texts: list[str],
    prompt_text: str,
    prompt_wav: str,
) -> list[tuple[str, str, str, str, str | None]]:
    rows = []
    for idx, tts_text in enumerate(texts):
        sample_id = f"cer_{idx:06d}"
        caption = "验收错词率评测集；测试集1；Seed-TTS-Eval中文"
        rows.append((sample_id, prompt_text, prompt_wav, tts_text, caption))
    return rows


def build_wer_set2_rows(
    dialogs: list[dict],
    prompt_text: str,
    prompt_wav: str,
) -> list[tuple[str, str, str, str, str | None]]:
    """测试集2：总体评测样例中客服+用户全部话术（与 Excel 评测集二一致）。"""
    rows = []
    for dlg in dialogs:
        d_idx = dlg["dialog_idx"]
        prefix = dlg["caption_prefix"]
        for u_idx, (role, tts_text) in enumerate(dlg["turns"]):
            sample_id = f"cer2_{d_idx:03d}_{u_idx:06d}"
            caption = f"{prefix}；测试集2；{role}话术"
            rows.append((sample_id, prompt_text, prompt_wav, tts_text, caption))
    return rows


def build_mos_rows(
    dialogs: list[dict],
    prompt_text: str,
    prompt_wav: str,
) -> list[tuple[str, str, str, str, str | None]]:
    rows = []
    for dlg in dialogs:
        d_idx = dlg["dialog_idx"]
        prefix = dlg["caption_prefix"]
        for u_idx, (role, tts_text) in enumerate(dlg["turns"]):
            if role != "客服":
                continue
            sample_id = f"mos_{d_idx:03d}_{u_idx:06d}"
            caption = f"{prefix}；客服话术"
            rows.append((sample_id, prompt_text, prompt_wav, tts_text, caption))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=Path(__file__).resolve().parent / "测评对话.xlsx",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--prompt-wav", default=DEFAULT_PROMPT_WAV)
    parser.add_argument(
        "--wer-only",
        action="store_true",
        help="Only write CER/WER test set 1 & 2 (skip MOS subjective lst)",
    )
    parser.add_argument(
        "--mos-only",
        action="store_true",
        help="Only write MOS subjective dialog lst",
    )
    args = parser.parse_args()

    if not args.prompt_wav or not Path(args.prompt_wav).is_file():
        raise SystemExit(f"prompt wav not found: {args.prompt_wav}")

    xlsx = args.xlsx.resolve()
    out_dir = args.out_dir.resolve()

    write_wer = not args.mos_only
    write_mos = not args.wer_only

    if write_wer:
        wer_texts = load_wer_texts(xlsx)
        wer_path = out_dir / "ceping_cer_wer.lst"
        wer_rows = build_wer_set1_rows(wer_texts, args.prompt_text, args.prompt_wav)
        write_lst(wer_path, wer_rows)
        print(f"Wrote {len(wer_rows)} lines (测试集1) -> {wer_path}")

        mos_all_dialogs = load_mos_dialogs(xlsx, kefu_only=False)
        wer2_path = out_dir / "ceping_cer_wer_set2.lst"
        wer2_rows = build_wer_set2_rows(
            mos_all_dialogs, args.prompt_text, args.prompt_wav
        )
        write_lst(wer2_path, wer2_rows)
        n_kefu = sum(1 for d in mos_all_dialogs for r, _ in d["turns"] if r == "客服")
        n_user = sum(1 for d in mos_all_dialogs for r, _ in d["turns"] if r == "用户")
        print(
            f"Wrote {len(wer2_rows)} lines (测试集2, 客服{n_kefu}+用户{n_user}, "
            f"{len(mos_all_dialogs)} dialogs) -> {wer2_path}"
        )

    if write_mos:
        mos_dialogs = load_mos_dialogs(xlsx, kefu_only=True)
        mos_path = out_dir / "ceping_mos_dialog.lst"
        mos_rows = build_mos_rows(mos_dialogs, args.prompt_text, args.prompt_wav)
        write_lst(mos_path, mos_rows)
        print(
            f"Wrote {len(mos_rows)} lines ({len(mos_dialogs)} dialogs) -> {mos_path}"
        )


if __name__ == "__main__":
    main()
