import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
from collections import defaultdict
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm

# --- 核心配置 ---
os.environ["GEMINI_API_KEY"] = "sk-Eu7adosdYbKT9tgRjOyC4Ls5GHyWsTRF4DGeO5TnBpChkNN6"
os.environ["GEMINI_URL"] = "https://apim1tocn.cheapapi.ai"
"""
/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/out_grouped_v2_0428_3k10_mp                        
  请你自行读取我当前的kefu_dpo_pairs jsonl文件，其中有接近3000对的dpo数据对和其对应的分析正负样例逻辑理由，请你帮助我完全写一个网页前端，可以读取我的json  
  l然后帮助我展示每一对样例音频对比及其相应的信息，方便我听取当前打标正负样例效果。完全交给你完成，直到达到我的目标要求

"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use Gemini to rank kefu-style DPO audio candidates and export win/lose pairs."
    )
    parser.add_argument("--api_key", default=os.getenv("GEMINI_API_KEY", ""), help="Gemini API key")
    parser.add_argument("--url", default="https://apim1tocn.cheapapi.ai", help="Gemini proxy/base URL")
    parser.add_argument(
        "--model",
        default="gemini-3.1-pro-preview",
        help="Model name in endpoint path: /v1beta/models/<model>:generateContent",
    )
    parser.add_argument(
        "--input_jsonl",
        default="/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo/grouped_speech_tokens/dpo_rows_with_group_token_dedup.jsonl",
        help="Flat rows jsonl with fields like group_id/utt/token/text/wav_path",
    )
    parser.add_argument(
        "--output_jsonl",
        default="/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/kefu_dpo_pairs.jsonl",
        help="Output normalized DPO pair jsonl path",
    )
    parser.add_argument(
        "--discard_log",
        default="/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/kefu_dpo_pairs_discard.log",
        help="Discard/error log path",
    )
    parser.add_argument(
        "--summary_json",
        default="/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/kefu_dpo_pairs_summary.json",
        help="Summary json path",
    )
    parser.add_argument("--concurrent_groups", type=int, default=1, help="Parallel groups")
    parser.add_argument("--min_group_size", type=int, default=2, help="Minimum candidates per group")
    parser.add_argument("--max_retries", type=int, default=3, help="API retry count")
    parser.add_argument("--sleep_between_calls", type=float, default=0.3, help="Gap between API calls")
    return parser.parse_args()


def create_kefu_tournament_prompt(text, current_best, current_worst, candidate):
    """
    客服语音风格专用的评审 prompt。
    重点目标：筛选出“更像专业客服”的版本（chosen）和“明显不理想”的版本（rejected）。
    """
    return f"""
你是一位资深客服质检与语音体验评测专家。你的任务是从客服场景出发，比较三个音频版本，并更新当前最佳与最差版本。

目标文本：
"{text}"

版本说明：
- [current_best]: {current_best}
- [current_worst]: {current_worst}
- [candidate]: {candidate}

请重点按以下客服维度判断（重要性从高到低）：
1) 客服专业感与信任感（最重要）
   - 是否礼貌、稳重、让用户愿意继续沟通
   - 情绪是否克制，避免尖锐、生硬、敷衍
2) 清晰可懂度
   - 吐字是否清晰、句子是否完整、关键信息是否可辨识
3) 安抚与服务语气
   - 是否体现“先安抚再解决”的客服沟通感受
4) 自然度与连贯性
   - 是否有明显机械感、断裂感、突兀停顿
5) 交付可用性
   - 该版本是否适合直接用于客服系统上线语料

更新规则：
- 如果 candidate 在客服体验上明显优于 current_best，则 new_best_version="candidate"，否则 "current_best"
- 如果 candidate 在客服体验上明显劣于 current_worst，则 new_worst_version="candidate"，否则 "current_worst"
- 如果难以判断，请保持现状，不要随意变更

你必须只输出 JSON（不要输出其他文本）：
{{
  "new_best_version": "candidate" 或 "current_best",
  "new_worst_version": "candidate" 或 "current_worst",
  "reason": "简洁中文理由，指出客服维度差异"
}}
"""


def extract_json_from_text(s: str):
    if not s:
        return None
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def normalize_wav_path(item):
    return item.get("wav_path") or item.get("wavpath") or ""


def normalize_version_id(item):
    # prefer suffix_idx, fallback to utt tail
    if item.get("suffix_idx") is not None:
        return f"v{int(item['suffix_idx']):03d}"
    utt = str(item.get("utt", ""))
    m = re.search(r"_([0-9]{3})$", utt)
    if m:
        return f"v{int(m.group(1)):03d}"
    return f"v_{utt}"


async def analyze_audio_tournament(
    session,
    url,
    model,
    api_key,
    text,
    best_info,
    worst_info,
    candidate_info,
    max_retries=3,
):
    api_url = f"{url}/v1beta/models/{model}:generateContent?key={api_key}"
    prompt_text = create_kefu_tournament_prompt(text, best_info["version"], worst_info["version"], candidate_info["version"])

    parts = [{"text": prompt_text}]
    for item in [best_info, worst_info, candidate_info]:
        v_id = item["version"]
        path = item["wav_path"]
        try:
            audio_bytes = Path(path).read_bytes()
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type is None:
                mime_type = "audio/wav"
            parts.append({"text": f"下面是版本 {v_id}："})
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    }
                }
            )
        except Exception as e:
            return None, f"file_read_error: {e}"

    request_data = {"contents": [{"parts": parts}]}
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        try:
            async with session.post(api_url, headers=headers, json=request_data, timeout=180) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text_out = ""
                    try:
                        text_out = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    except Exception:
                        return None, f"invalid_response: {json.dumps(data, ensure_ascii=False)[:2000]}"
                    parsed = extract_json_from_text(text_out)
                    if parsed is not None:
                        return parsed, None
                    return None, f"json_parse_error: {text_out[:500]}"
                if resp.status == 429:
                    await asyncio.sleep(8 + 2 * attempt)
                else:
                    body = await resp.text()
                    await asyncio.sleep(2**attempt)
                    if attempt == max_retries:
                        return None, f"http_{resp.status}: {body[:500]}"
        except Exception as e:
            if attempt == max_retries:
                return None, f"request_exception: {e}"
            await asyncio.sleep(2**attempt)
    return None, "unknown_error"


def build_groups_from_flat_rows(input_jsonl, min_group_size=2):
    groups = defaultdict(list)
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            gid = obj.get("group_id")
            if not gid:
                # fallback: remove last _NNN
                utt = str(obj.get("utt", ""))
                m = re.match(r"^(.*)_([0-9]{3})$", utt)
                gid = m.group(1) if m else utt
                obj["group_id"] = gid
            obj["version"] = normalize_version_id(obj)
            obj["wav_path"] = normalize_wav_path(obj)
            groups[gid].append(obj)

    valid_groups = {}
    for gid, items in groups.items():
        items = [x for x in items if x.get("wav_path")]
        items.sort(key=lambda x: int(re.sub(r"\\D", "", x.get("version", "0")) or 0))
        if len(items) >= min_group_size:
            valid_groups[gid] = items
    return valid_groups


async def process_one_group(session, args, group_id, items):
    # initial champion / bottom
    current_best = items[0]
    current_worst = items[1]
    text = items[0].get("text", "")

    for i in range(2, len(items)):
        candidate = items[i]
        await asyncio.sleep(args.sleep_between_calls)
        result, error = await analyze_audio_tournament(
            session=session,
            url=args.url,
            model=args.model,
            api_key=args.api_key,
            text=text,
            best_info=current_best,
            worst_info=current_worst,
            candidate_info=candidate,
            max_retries=args.max_retries,
        )
        if result is None:
            return None, f"{group_id}: compare_failed_with_{candidate['version']} | {error}"

        if result.get("new_best_version") == "candidate":
            current_best = candidate
        if result.get("new_worst_version") == "candidate":
            current_worst = candidate

    if current_best["version"] == current_worst["version"]:
        return None, f"{group_id}: best_equals_worst"

    # 标准 DPO 样本格式（可直接用于后续训练转换）
    out_obj = {
        "group_id": group_id,
        "prompt": text,
        "utt": group_id,
        "chosen": {
            "version": current_best["version"],
            "utt": current_best.get("utt", ""),
            "text": current_best.get("text", ""),
            "token": current_best.get("token", []),
            "wav_path": current_best.get("wav_path", ""),
        },
        "rejected": {
            "version": current_worst["version"],
            "utt": current_worst.get("utt", ""),
            "text": current_worst.get("text", ""),
            "token": current_worst.get("token", []),
            "wav_path": current_worst.get("wav_path", ""),
        },
        "meta": {
            "num_candidates": len(items),
            "judge_model": args.model,
            "judge_task": "kefu_style_win_lose_selection",
        },
    }
    return out_obj, None


async def main():
    args = parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Please pass --api_key or set GEMINI_API_KEY.")
    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"input_jsonl not found: {args.input_jsonl}")

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.discard_log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    # clear old files
    open(args.output_jsonl, "w", encoding="utf-8").close()
    open(args.discard_log, "w", encoding="utf-8").close()

    groups = build_groups_from_flat_rows(args.input_jsonl, args.min_group_size)
    total = len(groups)
    if total == 0:
        raise ValueError("No valid groups found after min_group_size/wav_path filtering.")

    sem = asyncio.Semaphore(args.concurrent_groups)
    connector = aiohttp.TCPConnector(limit=args.concurrent_groups)
    success = 0
    failed = 0
    write_lock = asyncio.Lock()

    async with aiohttp.ClientSession(connector=connector) as session:
        async def run_one(gid, items):
            nonlocal success, failed
            async with sem:
                res, err = await process_one_group(session, args, gid, items)
                if res is not None:
                    async with write_lock:
                        with open(args.output_jsonl, "a", encoding="utf-8") as f:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    success += 1
                else:
                    async with write_lock:
                        with open(args.discard_log, "a", encoding="utf-8") as f:
                            f.write(err + "\n")
                    failed += 1

        tasks = [run_one(gid, items) for gid, items in groups.items()]
        for fut in tqdm(asyncio.as_completed(tasks), total=total, desc="Gemini kefu ranking"):
            await fut

    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "discard_log": args.discard_log,
        "groups_total": total,
        "groups_success": success,
        "groups_failed": failed,
        "model": args.model,
        "concurrent_groups": args.concurrent_groups,
        "min_group_size": args.min_group_size,
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")