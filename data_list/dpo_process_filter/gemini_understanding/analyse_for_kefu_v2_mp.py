import argparse
import asyncio
import base64
import fcntl
import json
import logging
import mimetypes
import multiprocessing as mp
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    parser.add_argument("--num_processes", type=int, default=1, help="Number of worker processes")
    parser.add_argument(
        "--worker_concurrent_groups",
        type=int,
        default=0,
        help="Per-process async concurrency. 0 means using --concurrent_groups",
    )
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
    parser.add_argument(
        "--no_resume",
        action="store_false",
        dest="resume",
        help="Disable resume; re-process all groups (still incremental append unless --truncate_output)",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--truncate_output",
        action="store_true",
        help="Truncate output files and start fresh (ignores existing kefu_dpo_pairs.jsonl)",
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
            async with session.post(api_url, headers=headers, json=request_data, timeout=500) as resp:
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


def _chunk_groups(group_items, n_chunks):
    n_chunks = max(1, min(n_chunks, len(group_items)))
    chunks = [[] for _ in range(n_chunks)]
    for idx, item in enumerate(group_items):
        chunks[idx % n_chunks].append(item)
    return [c for c in chunks if c]


def summarize_error_reasons(errors):
    counts = defaultdict(int)
    for err in errors:
        if "batch_evaluate_failed" in err:
            key = "batch_evaluate_failed"
        elif "insufficient_candidates" in err:
            key = "insufficient_candidates"
        elif "best_equals_worst" in err:
            key = "best_equals_worst"
        elif "unexpected_exception" in err:
            key = "unexpected_exception"
        else:
            key = err.split(":", 1)[-1].strip().split(" ", 1)[0] if err else "unknown"
        counts[key] += 1
    return dict(sorted(counts.items()))


def build_manifest_from_result(res: dict) -> dict:
    return {
        "group_id": res["group_id"],
        "prompt": res.get("prompt", ""),
        "win_utt": res["chosen"].get("utt", ""),
        "win_wav_path": res["chosen"].get("wav_path", ""),
        "lose_utt": res["rejected"].get("utt", ""),
        "lose_wav_path": res["rejected"].get("wav_path", ""),
        "win_version": res["chosen"].get("version", ""),
        "lose_version": res["rejected"].get("version", ""),
    }


def _locked_append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line if line.endswith("\n") else line + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_completed_group_ids(out_dir: Path) -> set[str]:
    """Load group_id already written to kefu_dpo_pairs.jsonl for resume."""
    path = out_dir / "kefu_dpo_pairs.jsonl"
    done: set[str] = set()
    if not path.is_file():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            gid = obj.get("group_id")
            if gid:
                done.add(str(gid))
    return done


class IncrementalOutputWriter:
    """Append each group result immediately (safe for multi-process via flock)."""

    def __init__(self, args, out_dir: Path, *, truncate: bool = False):
        self.args = args
        self.out_dir = Path(out_dir)
        self.output_jsonl = self.out_dir / "kefu_dpo_pairs.jsonl"
        self.discard_log = self.out_dir / "kefu_dpo_pairs_discard.log"
        self.utt_win_lose_jsonl = self.out_dir / "utt_win_lose.jsonl"
        self.utt_win_lose_tsv = self.out_dir / "utt_win_lose.tsv"
        self.win_dir = self.out_dir / "win"
        self.lose_dir = self.out_dir / "lose"
        self._init_files(truncate)

    def _init_files(self, truncate: bool) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        mode = "w" if truncate else "a"
        if truncate or not self.output_jsonl.exists():
            open(self.output_jsonl, "w", encoding="utf-8").close()
        if truncate or not self.discard_log.exists():
            open(self.discard_log, "w", encoding="utf-8").close()
        if truncate or not self.utt_win_lose_jsonl.exists():
            open(self.utt_win_lose_jsonl, "w", encoding="utf-8").close()
        if truncate or not self.utt_win_lose_tsv.exists() or self.utt_win_lose_tsv.stat().st_size == 0:
            with open(self.utt_win_lose_tsv, mode, encoding="utf-8") as f:
                if mode == "w" or f.tell() == 0:
                    f.write("group_id\tprompt\twin_utt\twin_wav\tlose_utt\tlose_wav\n")
        if self.args.save_win_lose_audio:
            self.win_dir.mkdir(parents=True, exist_ok=True)
            self.lose_dir.mkdir(parents=True, exist_ok=True)

    def _save_audio(self, src: str, dst: Path) -> None:
        if not src or not Path(src).exists():
            return
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if self.args.audio_save_mode == "copy":
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(Path(src).resolve())

    def write_success(self, res: dict) -> None:
        manifest = build_manifest_from_result(res)
        _locked_append_line(self.output_jsonl, json.dumps(res, ensure_ascii=False))
        _locked_append_line(
            self.utt_win_lose_jsonl, json.dumps(manifest, ensure_ascii=False)
        )
        prompt_clean = manifest["prompt"].replace("\t", " ").replace("\n", " ")
        tsv_line = (
            f"{manifest['group_id']}\t{prompt_clean}\t{manifest['win_utt']}\t"
            f"{manifest['win_wav_path']}\t{manifest['lose_utt']}\t{manifest['lose_wav_path']}\n"
        )
        _locked_append_line(self.utt_win_lose_tsv, tsv_line)
        if self.args.save_win_lose_audio:
            self._save_audio(
                manifest["win_wav_path"],
                self.win_dir / f"{manifest['group_id']}__win__{manifest['win_utt']}.wav",
            )
            self._save_audio(
                manifest["lose_wav_path"],
                self.lose_dir / f"{manifest['group_id']}__lose__{manifest['lose_utt']}.wav",
            )

    def write_failure(self, err: str) -> None:
        _locked_append_line(self.discard_log, err)


async def run_groups_async(group_items, args, logger, writer: IncrementalOutputWriter | None = None):
    sem = asyncio.Semaphore(args.concurrent_groups)
    connector = aiohttp.TCPConnector(limit=args.concurrent_groups)
    success = failed = 0
    success_results = []
    failed_errors = []
    manifests = []
    incremental = writer is not None

    async with aiohttp.ClientSession(connector=connector) as session:
        async def run_one(gid, items):
            nonlocal success, failed
            async with sem:
                try:
                    res, err = await process_one_group(session, args, gid, items, logger=logger)
                    if res is not None:
                        success += 1
                        if incremental:
                            writer.write_success(res)
                        else:
                            success_results.append(res)
                            manifests.append(build_manifest_from_result(res))
                    else:
                        logger.warning("[%s] group_failed reason=%s", gid, err)
                        failed += 1
                        if incremental:
                            writer.write_failure(err)
                        else:
                            failed_errors.append(err)
                except Exception as e:
                    logger.exception("[%s] unexpected_exception", gid)
                    err_msg = f"{gid}: unexpected_exception | {e}"
                    failed += 1
                    if incremental:
                        writer.write_failure(err_msg)
                    else:
                        failed_errors.append(err_msg)

        tasks = [run_one(gid, items) for gid, items in group_items]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(group_items), desc="Gemini DPO Processing"):
            await fut

    return {
        "success": success,
        "failed": failed,
        "success_results": success_results,
        "failed_errors": failed_errors,
        "manifests": manifests,
    }


def process_chunk_worker(worker_id, group_items, args_dict):
    class _Args:
        pass

    args = _Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    logger = setup_logger(args.log_level)
    logger.info(
        "[worker-%d] start groups=%d concurrent_groups=%d",
        worker_id,
        len(group_items),
        args.concurrent_groups,
    )
    writer = IncrementalOutputWriter(args, Path(args.out_dir), truncate=False)
    result = asyncio.run(run_groups_async(group_items, args, logger, writer=writer))
    logger.info(
        "[worker-%d] done success=%d failed=%d",
        worker_id,
        result["success"],
        result["failed"],
    )
    return result


def write_outputs(args, out_dir, all_success_results, all_failed_errors, all_manifests):
    output_jsonl = str(out_dir / "kefu_dpo_pairs.jsonl")
    discard_log = str(out_dir / "kefu_dpo_pairs_discard.log")
    summary_json = str(out_dir / "kefu_dpo_pairs_summary.json")
    utt_win_lose_jsonl = out_dir / "utt_win_lose.jsonl"
    utt_win_lose_tsv = out_dir / "utt_win_lose.tsv"
    win_dir = out_dir / "win"
    lose_dir = out_dir / "lose"

    open(output_jsonl, "w", encoding="utf-8").close()
    open(discard_log, "w", encoding="utf-8").close()
    open(utt_win_lose_jsonl, "w", encoding="utf-8").close()
    with open(utt_win_lose_tsv, "w", encoding="utf-8") as f:
        f.write("group_id\tprompt\twin_utt\twin_wav\tlose_utt\tlose_wav\n")

    if args.save_win_lose_audio:
        win_dir.mkdir(parents=True, exist_ok=True)
        lose_dir.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "a", encoding="utf-8") as f:
        for res in all_success_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    with open(utt_win_lose_jsonl, "a", encoding="utf-8") as f_jsonl, open(
        utt_win_lose_tsv, "a", encoding="utf-8"
    ) as f_tsv:
        for manifest_obj in all_manifests:
            f_jsonl.write(json.dumps(manifest_obj, ensure_ascii=False) + "\n")
            prompt_clean = manifest_obj["prompt"].replace("\t", " ").replace("\n", " ")
            f_tsv.write(
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
                    if not src or not Path(src).exists():
                        return
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    if args.audio_save_mode == "copy":
                        shutil.copy2(src, dst)
                    else:
                        dst.symlink_to(Path(src).resolve())

                _save_audio(
                    manifest_obj["win_wav_path"],
                    win_dir / f"{manifest_obj['group_id']}__win__{manifest_obj['win_utt']}.wav",
                )
                _save_audio(
                    manifest_obj["lose_wav_path"],
                    lose_dir / f"{manifest_obj['group_id']}__lose__{manifest_obj['lose_utt']}.wav",
                )

    with open(discard_log, "a", encoding="utf-8") as f:
        for err in all_failed_errors:
            f.write(err + "\n")

    return summary_json


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    worker_concurrency = args.worker_concurrent_groups if args.worker_concurrent_groups > 0 else args.concurrent_groups
    logger.info(
        "run_start input=%s format=%s model=%s num_processes=%d worker_concurrent_groups=%d",
        args.input_jsonl,
        args.input_format,
        args.model,
        args.num_processes,
        worker_concurrency,
    )

    if not args.api_key:
        raise ValueError("Missing API key. Please pass --api_key or set GEMINI_API_KEY.")
    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"input_jsonl not found: {args.input_jsonl}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    do_resume = args.resume
    truncate = args.truncate_output
    if do_resume and not truncate:
        completed_ids = load_completed_group_ids(out_dir)
        if completed_ids:
            logger.info("resume_enabled completed_groups=%d", len(completed_ids))
    else:
        completed_ids = set()
        if truncate:
            logger.info("truncate_output enabled: fresh output files")

    IncrementalOutputWriter(args, out_dir, truncate=truncate)
    logger.info("incremental_write enabled out_dir=%s", out_dir)

    logger.info("build_groups_start min_group_size=%d", args.min_group_size)
    groups = build_groups(args.input_jsonl, args.min_group_size, args.input_format)
    total = len(groups)
    logger.info("build_groups_done total_groups=%d", total)
    if total == 0:
        raise ValueError("No valid groups found after min_group_size/wav_path filtering.")
    group_items = list(groups.items())
    pending_before = len(group_items)
    if completed_ids:
        group_items = [(gid, items) for gid, items in group_items if gid not in completed_ids]
        logger.info(
            "resume_filter pending=%d skipped_done=%d",
            len(group_items),
            pending_before - len(group_items),
        )
    if len(group_items) == 0:
        logger.info("all groups already done, writing summary only")

    process_count = max(1, min(args.num_processes, len(group_items) or 1))

    args_dict = vars(args).copy()
    args_dict["concurrent_groups"] = worker_concurrency
    args_dict["out_dir"] = str(out_dir)

    all_failed_errors = []
    success = failed = 0

    if len(group_items) > 0:
        if process_count == 1:
            logger.info("single_process_mode enabled")
            single_result = process_chunk_worker(0, group_items, args_dict)
            all_failed_errors.extend(single_result["failed_errors"])
            success = single_result["success"]
            failed = single_result["failed"]
        else:
            logger.info("multi_process_mode enabled processes=%d", process_count)
            chunks = _chunk_groups(group_items, process_count)
            with ProcessPoolExecutor(max_workers=process_count, mp_context=mp.get_context("spawn")) as ex:
                futures = [
                    ex.submit(process_chunk_worker, worker_id, chunk, args_dict)
                    for worker_id, chunk in enumerate(chunks)
                ]
                for fut in as_completed(futures):
                    r = fut.result()
                    success += r["success"]
                    failed += r["failed"]
                    all_failed_errors.extend(r["failed_errors"])

    discard_log = out_dir / "kefu_dpo_pairs_discard.log"
    if discard_log.is_file():
        with open(discard_log, "r", encoding="utf-8") as f:
            all_failed_errors = [ln.strip() for ln in f if ln.strip()]

    completed_now = len(load_completed_group_ids(out_dir))
    summary_json = str(out_dir / "kefu_dpo_pairs_summary.json")

    summary = {
        "groups_total": total,
        "groups_pending_this_run": len(group_items),
        "groups_skipped_resume": pending_before - len(group_items) if completed_ids else 0,
        "groups_success_this_run": success,
        "groups_failed_this_run": failed,
        "groups_success_total": completed_now,
        "failed_reason_counts": summarize_error_reasons(all_failed_errors),
        "model": args.model,
        "max_batch_size": args.max_batch_size,
        "concurrent_groups": worker_concurrency,
        "num_processes": process_count,
        "out_dir": str(out_dir),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n✅ Processing Complete! Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user. Incremental results are already saved under --out_dir.")