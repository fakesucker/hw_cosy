#!/usr/bin/env python3
"""
为 kefu.lst 第 4 列 tts_text 插入韵律标记（与训练侧 #2/#3/@ 约定一致）：

  #2  短停顿，一般放在较短意群边界前（常见：逗号前）
  #3  长停顿，一般放在句末或较长意群边界前（常见：句号/问号/叹号/分号前）
  @   重读：每个 @ 只作用于其左侧紧邻的一个字；多字词要整词重读时，需在该词每个字后各加一个 @（如 话@费@）

默认启发式（仅作初稿，务必抽检）：
  - 在 「，」「、」「：」 前插入 #2（若该位置尚无 #2/#3）
  - 在 「。」「！」「？」「；」 前插入 #3

用法:
  python annotate_tts_prosody.py --in kefu.lst --out kefu_with_prosody.lst
  python annotate_tts_prosody.py --in kefu.lst --out kefu_with_prosody.lst \\
      --emphasis-json emphasis.json
emphasis.json 示例（按 sample id 指定若干词；每个词会在其每个字后分别加 @）:
  {"001_000000_000000": ["话费", "短信"]}  → 话@费@  短@信@
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _out_ends_with_pause_mark(out: list[str]) -> bool:
    if len(out) < 2:
        return False
    return out[-2] == "#" and out[-1] in ("2", "3")


def heuristic_prosody(text: str) -> str:
    """在「，、：」前插 #2，在「。！？；」前插 #3；该标点前已有 #2/#3 则跳过。"""
    if not text:
        return text
    out: list[str] = []
    for c in text:
        if c in "，、：":
            if not _out_ends_with_pause_mark(out):
                out.extend(["#", "2"])
        elif c in "。！？；":
            if not _out_ends_with_pause_mark(out):
                out.extend(["#", "3"])
        out.append(c)
    return "".join(out)


# 客服/电信/保险场景常见「信息焦点」词（长词优先匹配）；每行最多匹配 max_spans 个词，每词内逐字加 @
_EMPHASIS_LEXICON = sorted(
    {
        "百万医疗险",
        "尊享e生",
        "青春畅想5G套餐",
        "理赔绿色通道",
        "预付赔款",
        "医保外用药责任险",
        "医保外用药",
        "定向流量",
        "通用流量",
        "双录视频",
        "犹豫期",
        "合约期",
        "免赔额",
        "三者险",
        "车损险",
        "交强险",
        "商业险",
        "报案号",
        "验证码",
        "营业厅",
        "客服热线",
        "话费",
        "短信",
        "余额",
        "套餐",
        "流量",
        "理赔",
        "报销",
        "保险",
        "扣费",
        "号卡",
        "宽带",
        "月租",
        "续保",
        "退款",
        "重疾险",
        "退保",
        "手续费",
        "保额",
        "保费",
        "保全",
        "复通",
        "挂失",
    },
    key=len,
    reverse=True,
)


def _insert_at_after_each_char_in_span(text: str, start: int, end: int) -> str:
    """在 text[start:end) 内每个字符右侧插入 @（若该字符右侧尚未是 @）。从右向左插入，避免下标错位。"""
    if start >= end or not text:
        return text
    end = min(end, len(text))
    start = max(0, start)
    for i in range(end - 1, start - 1, -1):
        if i + 1 < len(text) and text[i + 1] == "@":
            continue
        text = text[: i + 1] + "@" + text[i + 1 :]
    return text


def add_emphasis_lexicon(text: str, max_spans: int = 2) -> str:
    """在长词优先、互不重叠前提下，对若干关键词的每个字分别在其后插入 @。"""
    if not text or max_spans <= 0:
        return text
    placed: list[tuple[int, int]] = []
    for kw in _EMPHASIS_LEXICON:
        if len(placed) >= max_spans:
            break
        pos = text.find(kw)
        if pos < 0:
            continue
        s, e = pos, pos + len(kw)
        if any(s < e2 and e > s2 for s2, e2 in placed):
            continue
        placed.append((s, e))
    for s, e in sorted(placed, key=lambda x: -x[0]):
        text = _insert_at_after_each_char_in_span(text, s, e)
    return text


def add_emphasis_after_words(text: str, words: list[str]) -> str:
    """对每个指定词在文中的每次出现，在该词每个字后分别插入 @。"""
    if not words:
        return text
    for w in words:
        if not w:
            continue
        pos = 0
        while True:
            idx = text.find(w, pos)
            if idx < 0:
                break
            before_len = len(text)
            text = _insert_at_after_each_char_in_span(text, idx, idx + len(w))
            added = len(text) - before_len
            pos = idx + len(w) + added
    return text


def process_line(
    line: str,
    emphasis_map: dict[str, list[str]],
    use_heuristic: bool,
    auto_emphasis: bool = True,
    emphasis_max: int = 2,
) -> str:
    parts = line.rstrip("\n").split("|")
    if len(parts) < 4:
        return line
    sid = parts[0]
    tts = parts[3]
    if use_heuristic and not (("#2" in tts) or ("#3" in tts)):
        tts = heuristic_prosody(tts)
    if auto_emphasis and not ("@" in tts):
        tts = add_emphasis_lexicon(tts, max_spans=emphasis_max)
    if sid in emphasis_map:
        tts = add_emphasis_after_words(tts, emphasis_map[sid])
    parts[3] = tts
    return "|".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="为 kefu.lst 的 tts_text 插入 #2 #3 @")
    ap.add_argument("--in", dest="inp", required=True, help="输入 .lst")
    ap.add_argument(
        "--out",
        dest="out",
        default="",
        help="输出 .lst；与 --in-place 联用时可省略（等同覆盖输入）",
    )
    ap.add_argument(
        "--no-heuristic",
        action="store_true",
        help="不插入 #2/#3，仅应用 --emphasis-json 中的 @",
    )
    ap.add_argument(
        "--emphasis-json",
        type=str,
        default="",
        help='JSON: {"utt_id": ["词1","词2"]}，每个词内逐字后加 @',
    )
    ap.add_argument(
        "--no-auto-emphasis",
        action="store_true",
        help="不使用内置词表自动插入 @（仍可用 --emphasis-json）",
    )
    ap.add_argument(
        "--emphasis-max",
        type=int,
        default=2,
        help="每行从词表最多匹配几个关键词（每个词内逐字加 @，默认 2）",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="写回输入文件；会先复制为 .lst.bak",
    )
    args = ap.parse_args()

    emphasis_map: dict[str, list[str]] = {}
    if args.emphasis_json:
        p = Path(args.emphasis_json)
        if not p.is_file():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)
        emphasis_map = json.loads(p.read_text(encoding="utf-8"))

    inp = Path(args.inp)
    if args.in_place:
        import shutil

        bak = inp.with_suffix(".lst.bak")
        shutil.copy2(inp, bak)
        out = inp
        print(f"Backup: {bak}", file=sys.stderr)
    elif args.out:
        out = Path(args.out)
    else:
        print("Error: 请指定 --out，或使用 --in-place", file=sys.stderr)
        sys.exit(1)

    use_h = not args.no_heuristic
    auto_em = not args.no_auto_emphasis

    with inp.open("r", encoding="utf-8") as fin:
        lines = fin.readlines()
    out_lines = []
    for line in lines:
        if not line.strip():
            continue
        out_lines.append(
            process_line(
                line,
                emphasis_map,
                use_h,
                auto_emphasis=auto_em,
                emphasis_max=args.emphasis_max,
            )
        )
    with out.open("w", encoding="utf-8") as fout:
        fout.writelines(out_lines)

    print(f"Wrote {out} ({len(out_lines)} lines)", file=sys.stderr)


if __name__ == "__main__":
    main()
