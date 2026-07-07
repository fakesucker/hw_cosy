import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm

# # --- 核心配置 ---
# os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "你的API_KEY填在这里或通过环境变量传入")
# os.environ["GEMINI_URL"] = os.getenv("GEMINI_URL", "https://apim1tocn.cheapapi.ai")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use Gemini to rank kefu-style DPO audio candidates with Map-Reduce batching."
    )
    parser.add_argument("--api_key", default=os.getenv("GEMINI_API_KEY", ""), help="Gemini API key")
    parser.add_argument("--url", default=os.getenv("GEMINI_URL", ""), help="Gemini proxy/base URL")
    parser.add_argument(
        "--model",
        default="gemini-3.1-pro-preview",
        help="Model name in endpoint path",
    )
    parser.add_argument(
        "--input_jsonl",
        default="input_test.jsonl", # 替换为你自己的路径
        help="Input JSONL format: flat or grouped rows.",
    )
    parser.add_argument(
        "--out_dir",
        default="./dpo_output", # 替换为你自己的路径
        help="Directory to store all outputs.",
    )
    parser.add_argument("--concurrent_groups", type=int, default=2, help="Parallel groups")
    parser.add_argument("--min_group_size", type=int, default=2, help="Minimum candidates per group")
    parser.add_argument("--max_batch_size", type=int, default=4, help="每次请求大模型最多评估的音频数量，防止超长报错")
    parser.add_argument("--max_retries", type=int, default=3, help="API retry count")
    parser.add_argument("--sleep_between_calls", type=float, default=0.5, help="Gap between API calls")
    parser.add_argument(
        "--input_format",
        choices=["auto", "flat", "grouped"],
        default="auto",
    )
    parser.add_argument(
        "--save_win_lose_audio",
        action="store_true",
        help="Save chosen/rejected wav files into out_dir/win and out_dir/lose",
    )
    parser.add_argument(
        "--audio_save_mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to save win/lose audio files",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logger level",
    )
    return parser.parse_args()


def setup_logger(log_level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("kefu_dpo")


def create_kefu_scoring_prompt(text, candidates):
    version_list_str = "\n".join([f"- {c['version']}" for c in candidates])

    return f"""
你是一名【资深智能客服语音质检专家】。你的任务是盲测一组同文本的客服语音候选版本，选出表现最好的（best）和表现最差的（worst），用于训练 AI 偏好模型。

====================
【目标播报文本】
{text}

====================
【当前候选版本池】
{version_list_str}

====================
【评判标准与优先级】（从高到低）

⛔ 优先级 1：一票否决项（致命错误）
如果音频出现以下情况，必须作为 worst 候选：
- 严重合成瑕疵：电音、底噪、吞音、明显的拼接断裂感。
- 情绪异常：冷漠、不耐烦、带攻击性或机械生硬。

⭐ 优先级 2：客服业务核心维度（优胜标准）
在没有致命错误的前提下，对比以下维度选出 best：
1. 专业度：稳重、可信、像真实客服。
2. 亲和力：礼貌、温和、有耐心（带有服务感）。
3. 自然度：语调平稳连贯，重音准确，无机械感。

====================
【输出格式要求】
⚠️ 必须严格输出 JSON 格式。
⚠️ 必须先在 results 中详细写出对比分析理由（analysis），再输出最终决定。

{{
  "results": [
    {{
      "version": "v001",
      "analysis": "无电音瑕疵。语气温和有耐心，亲和力强，服务感明显优于其他版本。"
    }},
    {{
      "version": "v002",
      "analysis": "结尾处有明显电音（合成瑕疵），且语速过快，缺乏专业稳重感。"
    }}
  ],
  "ranking": ["v001", "v003", "v002"],
  "best": "v001",
  "worst": "v002"
}}
"""

def extract_json_from_text(s: str):
    if not s:
        return None
    # 暴力清理可能包含的 markdown 格式
    s = s.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
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
    if item.get("suffix_idx") is not None:
        return f"v{int(item['suffix_idx']):03d}"
    utt = str(item.get("utt", ""))
    m = re.search(r"_([0-9]{3})$", utt)
    if m:
        return f"v{int(m.group(1)):03d}"
    return f"v_{utt}"


def version_sort_key(version: str) -> int:
    if version is None:
        return 10**9
    digits = re.sub(r"\D", "", str(version))
    if not digits:
        return 10**9
    try:
        return int(digits)
    except Exception:
        return 10**9


async def analyze_audio_batch(session, url, model, api_key, text, candidates, max_retries=3, logger=None, group_id="unknown"):
    """
    将一组候选音频打包，向大模型发起单次打分请求
    """
    api_url = f"{url}/v1beta/models/{model}:generateContent?key={api_key}"
    prompt_text = create_kefu_scoring_prompt(text, candidates)

    parts = [{"text": prompt_text}]
    
    if logger:
        logger.info("[%s] prepare_batch candidates=%d model=%s", group_id, len(candidates), model)

    for item in candidates:
        v_id = item["version"]
        path = item["wav_path"]
        try:
            if logger:
                logger.debug("[%s] read_audio version=%s path=%s", group_id, v_id, path)
            audio_bytes = Path(path).read_bytes()
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type is None:
                mime_type = "audio/wav"
            parts.append({"text": f"\n[音频版本 ID]: {v_id}"})
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    }
                }
            )
        except Exception as e:
            return None, f"file_read_error on {v_id}: {e}"

    request_data = {"contents": [{"parts": parts}]}
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        try:
            if logger:
                logger.info(
                    "[%s] api_request_start attempt=%d/%d candidates=%d",
                    group_id,
                    attempt + 1,
                    max_retries + 1,
                    len(candidates),
                )
            async with session.post(api_url, headers=headers, json=request_data, timeout=180) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    try:
                        text_out = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    except Exception:
                        return None, f"invalid_response_structure: {json.dumps(data, ensure_ascii=False)[:500]}"
                    
                    parsed = extract_json_from_text(text_out)
                    if parsed is not None:
                        if logger:
                            logger.info("[%s] api_request_ok attempt=%d", group_id, attempt + 1)
                        return parsed, None
                    return None, f"json_parse_error: {text_out[:500]}"
                
                if resp.status == 429:
                    if logger:
                        logger.warning("[%s] api_429 attempt=%d, backing off", group_id, attempt + 1)
                    await asyncio.sleep(8 + 2 * attempt)
                else:
                    body = await resp.text()
                    if logger:
                        logger.warning(
                            "[%s] api_http_error status=%d attempt=%d body=%s",
                            group_id,
                            resp.status,
                            attempt + 1,
                            body[:200],
                        )
                    await asyncio.sleep(2**attempt)
                    if attempt == max_retries:
                        return None, f"http_{resp.status}: {body[:500]}"
        except Exception as e:
            if logger:
                logger.warning("[%s] api_exception attempt=%d err=%s", group_id, attempt + 1, e)
            if attempt == max_retries:
                return None, f"request_exception: {e}"
            await asyncio.sleep(2**attempt)
            
    return None, "unknown_error"


async def evaluate_candidates_hierarchically(session, args, text, candidates, logger=None, group_id="unknown", depth=0):
    """
    分治晋级算法 (Map-Reduce)：
    突破上下文窗口/Payload限制，自动切分长候选列表。
    """
    max_b = args.max_batch_size
    
    # 基础情况：数量在安全范围内，直接打分
    if len(candidates) <= max_b:
        if logger:
            logger.debug("[%s] depth=%d direct_eval candidates=%d", group_id, depth, len(candidates))
        result, error = await analyze_audio_batch(
            session=session,
            url=args.url,
            model=args.model,
            api_key=args.api_key,
            text=text,
            candidates=candidates,
            max_retries=args.max_retries,
            logger=logger,
            group_id=group_id,
        )
        if result is None:
            return None, None, f"batch_evaluate_failed | {error}", []
            
        best_ver = result.get("best")
        worst_ver = result.get("worst")
        
        c_best = next((x for x in candidates if x["version"] == best_ver), candidates[0])
        c_worst = next((x for x in candidates if x["version"] == worst_ver), candidates[-1])
        
        return c_best, c_worst, None, [result]

    # 递归情况：切分为多组 (Map)
    chunks = [candidates[i:i + max_b] for i in range(0, len(candidates), max_b)]
    promoted_candidates = []
    all_results = [] 
    
    for chunk in chunks:
        if len(chunk) == 1:
            promoted_candidates.append(chunk[0])
            continue
            
        await asyncio.sleep(args.sleep_between_calls)
        c_best, c_worst, err, c_results = await evaluate_candidates_hierarchically(
            session,
            args,
            text,
            chunk,
            logger=logger,
            group_id=group_id,
            depth=depth + 1,
        )
        
        if err:
            return None, None, err, []
            
        all_results.extend(c_results)
        
        promoted_candidates.append(c_best)
        if c_best["version"] != c_worst["version"]:
            promoted_candidates.append(c_worst)
            
    # 去重
    seen = set()
    unique_promoted = []
    for p in promoted_candidates:
        if p["version"] not in seen:
            seen.add(p["version"])
            unique_promoted.append(p)
            
    if len(unique_promoted) <= 1:
        return unique_promoted[0], unique_promoted[0], None, all_results

    # 总决赛 (Reduce)
    await asyncio.sleep(args.sleep_between_calls)
    f_best, f_worst, f_err, f_results = await evaluate_candidates_hierarchically(
        session,
        args,
        text,
        unique_promoted,
        logger=logger,
        group_id=group_id,
        depth=depth + 1,
    )
    all_results.extend(f_results)
    
    return f_best, f_worst, f_err, all_results


async def process_one_group(session, args, group_id, items, logger=None):
    if len(items) < 2:
        return None, f"{group_id}: insufficient_candidates({len(items)})"
        
    text = items[0].get("text", "")
    if logger:
        logger.info("[%s] group_start candidates=%d", group_id, len(items))

    # 执行分治锦标赛
    best_item, worst_item, error, meta_results = await evaluate_candidates_hierarchically(
        session, args, text, items, logger=logger, group_id=group_id
    )
    
    if error:
        return None, f"{group_id}: {error}"

    if best_item["version"] == worst_item["version"]:
        return None, f"{group_id}: best_equals_worst ({best_item['version']})"

    # 组装 DPO 标准格式
    out_obj = {
        "group_id": group_id,
        "prompt": text,
        "utt": group_id,
        "chosen": {
            "version": best_item["version"],
            "utt": best_item.get("utt", ""),
            "text": best_item.get("text", ""),
            "token": best_item.get("token", []),
            "wav_path": best_item.get("wav_path", ""),
        },
        "rejected": {
            "version": worst_item["version"],
            "utt": worst_item.get("utt", ""),
            "text": worst_item.get("text", ""),
            "token": worst_item.get("token", []),
            "wav_path": worst_item.get("wav_path", ""),
        },
        "meta": {
            "num_candidates": len(items),
            "judge_model": args.model,
            "judge_task": "kefu_style_batch_scoring",
            "gemini_eval_history": meta_results
        },
    }
    if logger:
        logger.info(
            "[%s] group_done best=%s worst=%s",
            group_id,
            best_item["version"],
            worst_item["version"],
        )
    return out_obj, None


# ==========================================
# 数据构建与解析模块 (无修改，保留原有逻辑)
# ==========================================

def build_groups_from_flat_rows(input_jsonl, min_group_size=2):
    groups = defaultdict(list)
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            obj = json.loads(s)
            gid = obj.get("group_id")
            if not gid:
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
        items.sort(key=lambda x: version_sort_key(x.get("version", "")))
        if len(items) >= min_group_size:
            valid_groups[gid] = items
    return valid_groups

def build_groups_from_grouped_rows(input_jsonl, min_group_size=2):
    groups = {}
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            obj = json.loads(s)
            gid = obj.get("group_id")
            if not gid: continue
            raw_items = obj.get("items", [])
            items = []
            for it in raw_items:
                x = dict(it)
                x["group_id"] = gid
                x["version"] = normalize_version_id(x)
                x["wav_path"] = normalize_wav_path(x)
                if x.get("wav_path"):
                    items.append(x)
            items.sort(key=lambda x: version_sort_key(x.get("version", "")))
            if len(items) >= min_group_size:
                groups[gid] = items
    return groups

def build_groups(input_jsonl, min_group_size=2, input_format="auto"):
    if input_format == "flat":
        return build_groups_from_flat_rows(input_jsonl, min_group_size)
    if input_format == "grouped":
        return build_groups_from_grouped_rows(input_jsonl, min_group_size)

    first_obj = None
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            first_obj = json.loads(s)
            break
    if first_obj is None: return {}
    if isinstance(first_obj.get("items"), list):
        return build_groups_from_grouped_rows(input_jsonl, min_group_size)
    return build_groups_from_flat_rows(input_jsonl, min_group_size)


async def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    logger.info(
        "run_start input=%s format=%s model=%s concurrent_groups=%d",
        args.input_jsonl,
        args.input_format,
        args.model,
        args.concurrent_groups,
    )

    if not args.api_key:
        raise ValueError("Missing API key. Please pass --api_key or set GEMINI_API_KEY.")
    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"input_jsonl not found: {args.input_jsonl}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_jsonl = str(out_dir / "kefu_dpo_pairs.jsonl")
    discard_log = str(out_dir / "kefu_dpo_pairs_discard.log")
    summary_json = str(out_dir / "kefu_dpo_pairs_summary.json")
    utt_win_lose_jsonl = out_dir / "utt_win_lose.jsonl"
    utt_win_lose_tsv = out_dir / "utt_win_lose.tsv"
    win_dir = out_dir / "win"
    lose_dir = out_dir / "lose"

    # 初始化文件
    open(output_jsonl, "w", encoding="utf-8").close()
    open(discard_log, "w", encoding="utf-8").close()
    open(utt_win_lose_jsonl, "w", encoding="utf-8").close()
    with open(utt_win_lose_tsv, "w", encoding="utf-8") as f:
        f.write("group_id\tprompt\twin_utt\twin_wav\tlose_utt\tlose_wav\n")
        
    if args.save_win_lose_audio:
        win_dir.mkdir(parents=True, exist_ok=True)
        lose_dir.mkdir(parents=True, exist_ok=True)

    logger.info("build_groups_start min_group_size=%d", args.min_group_size)
    groups = build_groups(args.input_jsonl, args.min_group_size, args.input_format)
    total = len(groups)
    logger.info("build_groups_done total_groups=%d", total)
    if total == 0:
        raise ValueError("No valid groups found after min_group_size/wav_path filtering.")

    sem = asyncio.Semaphore(args.concurrent_groups)
    connector = aiohttp.TCPConnector(limit=args.concurrent_groups)
    success = failed = 0
    write_lock = asyncio.Lock()

    async with aiohttp.ClientSession(connector=connector) as session:
        async def run_one(gid, items):
            nonlocal success, failed
            async with sem:
                try:
                    res, err = await process_one_group(session, args, gid, items, logger=logger)
                    if res is not None:
                        async with write_lock:
                            with open(output_jsonl, "a", encoding="utf-8") as f:
                                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                                
                            manifest_obj = {
                                "group_id": res["group_id"],
                                "prompt": res.get("prompt", ""),
                                "win_utt": res["chosen"].get("utt", ""),
                                "win_wav_path": res["chosen"].get("wav_path", ""),
                                "lose_utt": res["rejected"].get("utt", ""),
                                "lose_wav_path": res["rejected"].get("wav_path", ""),
                                "win_version": res["chosen"].get("version", ""),
                                "lose_version": res["rejected"].get("version", ""),
                            }
                            with open(utt_win_lose_jsonl, "a", encoding="utf-8") as f:
                                f.write(json.dumps(manifest_obj, ensure_ascii=False) + "\n")
                            with open(utt_win_lose_tsv, "a", encoding="utf-8") as f:
                                prompt_clean = manifest_obj["prompt"].replace("\t", " ").replace("\n", " ")
                                f.write(
                                    "{}\t{}\t{}\t{}\t{}\t{}\n".format(
                                        manifest_obj["group_id"],
                                        prompt_clean,
                                        manifest_obj["win_utt"],
                                        manifest_obj["win_wav_path"],
                                        manifest_obj["lose_utt"],
                                        manifest_obj["lose_wav_path"],
                                    )
                                )
                            
                            if args.save_win_lose_audio:
                                def _save_audio(src: str, dst: Path):
                                    if not src or not Path(src).exists(): return
                                    if dst.exists() or dst.is_symlink(): dst.unlink()
                                    if args.audio_save_mode == "copy":
                                        shutil.copy2(src, dst)
                                    else:
                                        dst.symlink_to(Path(src).resolve())

                                _save_audio(manifest_obj["win_wav_path"], win_dir / f"{manifest_obj['group_id']}__win__{manifest_obj['win_utt']}.wav")
                                _save_audio(manifest_obj["lose_wav_path"], lose_dir / f"{manifest_obj['group_id']}__lose__{manifest_obj['lose_utt']}.wav")
                        success += 1
                    else:
                        logger.warning("[%s] group_failed reason=%s", gid, err)
                        async with write_lock:
                            with open(discard_log, "a", encoding="utf-8") as f:
                                f.write(err + "\n")
                        failed += 1
                except Exception as e:
                    logger.exception("[%s] unexpected_exception", gid)
                    async with write_lock:
                        with open(discard_log, "a", encoding="utf-8") as f:
                            f.write(f"{gid}: unexpected_exception | {e}\n")
                    failed += 1

        tasks = [run_one(gid, items) for gid, items in groups.items()]
        for fut in tqdm(asyncio.as_completed(tasks), total=total, desc="Gemini DPO Processing"):
            await fut

    summary = {
        "groups_total": total,
        "groups_success": success,
        "groups_failed": failed,
        "model": args.model,
        "max_batch_size": args.max_batch_size,
        "concurrent_groups": args.concurrent_groups,
        "out_dir": str(out_dir),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n✅ Processing Complete! Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")