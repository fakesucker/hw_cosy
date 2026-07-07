import os
import asyncio
import aiohttp
import aiofiles
import base64
import json
import mimetypes
import re
from tqdm.asyncio import tqdm
from concurrent.futures import ProcessPoolExecutor

# --- 核心配置 ---
API_KEY = "sk-Eu7adosdYbKT9tgRjOyC4Ls5GHyWsTRF4DGeO5TnBpChkNN6"
URL = "https://apim1tocn.cheapapi.ai"
INPUT_PATH = "/home/A02_tmpdata1/wenhaoli/code/wh_tts/output_wavs/output_raokouling_dpo/part_001/merge_out.jsonl"
OUTPUT_PATH = "/home/A02_tmpdata1/wenhaoli/code/wh_tts/RL/dpo/output/output_data_raokouling/part1/output.jsonl"
DISCARD_PATH = "/home/A02_tmpdata1/wenhaoli/code/wh_tts/RL/dpo/output/output_data_raokouling/part1/error.log"

# 并发配置
CONCURRENT_WORKERS = 16  # 进程数（根据 CPU 核心数和 API 限制调整）
UTTS_PER_WORKER = 1     # 每个进程同时处理的 UTT 数量
# ----------------

def create_tournament_prompt(text, current_best, current_worst, candidate):
    """
    专门针对中文绕口令（Tongue Twister）筛选优化的提示词。
    重点考察：声母切换的灵活性、声调准确性、吐字归音的力度。
    """
    return f"""
你是一位顶级的播音主持评测专家，专门负责考核播音员在极端发音挑战（绕口令）下的吐字归音水平。你的任务是区分出“字正腔圆、颗粒感强”的高质量发音与“含混不清、咬字拌蒜”的低质量发音。

待测绕口令文本："{text}"

请分析以下三个音频版本，基于绕口令的专业维度识别出表现最好（BEST）和最差（WORST）的版本：

1. **吐字颗粒感 (Articulatory Precision - 最重要)**: 
   - 辅音（声母）是否清晰有力？特别是平翘舌（s/sh, z/zh, c/ch）、鼻音边音（n/l）等易混淆点的区分度。
   - 是否存在“吃字”或“粘连”现象？高质量的发音应“字字入耳”。

2. **发音灵活性 (Agility)**: 
   - 在高速切换相似发音部位时，是否有“拌蒜”或停顿？
   - 观察是否存在“舌头打结”导致的瞬时语音模糊。

3. **声调稳定性 (Tone Accuracy)**: 
   - 绕口令在快速朗读时，声调是否依然准确到位？是否存在因速度过快导致声调走样、变成平调的情况。

4. **气流控制 (Breath Control)**: 
   - 语音是否平稳，是否存在因换气不当导致的字音发虚。

待比较版本：
- [Current Best] (当前冠军): {current_best} （吐字最清晰、最稳健的版本）
- [Current Worst] (当前最差): {current_worst} （最含混、最易出错的版本）
- [Candidate] (新挑战者): {candidate} （待评估的新版本）

判定逻辑：
- **升级冠军 (Upgrade Best)**: 如果挑战者比 [Current Best] 表现出更强的颗粒感、更灵活的声母转换且完全没有停顿，则 new_best_version = "candidate"。
- **降级最差 (Downgrade Worst)**: 如果挑战者出现了明显的“拌蒜”、声母混淆、或比 [Current Worst] 更加含混，则 new_worst_version = "candidate"。
- **维持现状 (Maintain)**: 如果挑战者表现平平，介于两者之间，请保留原基准。

你必须【仅】返回一个 JSON 对象，严禁包含任何其他描述文字：
{{
  "new_best_version": "candidate" 或 "current_best",
  "new_worst_version": "candidate" or "current_worst",
  "analysis": "具体理由：例如'挑战者在平翘舌转换上极其丝滑'或'挑战者在处理n/l时出现了明显咬字不清'。"
}}
"""

async def analyze_audio_tournament(session, text, best_info, worst_info, candidate_info):
    api_url = f"{URL}/v1beta/models/gemini-3.1-pro-preview:generateContent?key={API_KEY}"
    prompt_text = create_tournament_prompt(text, best_info[0], worst_info[0], candidate_info[0])
    
    parts = [{"text": prompt_text}]
    for v_id, path in [best_info, worst_info, candidate_info]:
        try:
            async with aiofiles.open(path, "rb") as f:
                audio_bytes = await f.read()
            mime_type, _ = mimetypes.guess_type(path) or "audio/wav"
            parts.append({"text": f"这是版本 {v_id}:", "inline_data": {"mime_type": mime_type, "data": base64.b64encode(audio_bytes).decode('utf-8')}})
        except Exception as e:
            return None, f"IO Error: {e}"

    try:
        async with session.post(api_url, json={"contents": [{"parts": parts}]}, timeout=120) as resp:
            if resp.status == 200:
                data = await resp.json()
                inner_text = data["candidates"][0]["content"]["parts"][0]['text'].strip()
                match = re.search(r'\{.*\}', inner_text, re.DOTALL)
                return (json.loads(match.group()), None) if match else (None, "JSON Parse Error")
            return None, f"API Status {resp.status}"
    except Exception as e:
        return None, str(e)

async def process_one_utt(session, utt_data):
    prefix, text = utt_data['utt'], utt_data['text']
    audio_map, token_map = utt_data['audio_path'], utt_data['token_path']
    versions = sorted(list(audio_map.keys()), key=lambda x: int(re.sub(r'\D', '', x)))
    
    if len(versions) < 2: return None
    curr_best_v, curr_worst_v = versions[0], versions[1]

    # 锦标赛筛选
    for i in range(2, len(versions)):
        candidate_v = versions[i]
        await asyncio.sleep(0.5) # 防止频率过快
        result, _ = await analyze_audio_tournament(session, text, (curr_best_v, audio_map[curr_best_v]), (curr_worst_v, audio_map[curr_worst_v]), (candidate_v, audio_map[candidate_v]))
        if result:
            if result.get("new_best_version") == "candidate": curr_best_v = candidate_v
            if result.get("new_worst_version") == "candidate": curr_worst_v = candidate_v

    if curr_best_v == curr_worst_v: return None
    return {
        "utt": prefix, "text": text,
        "chosen": {"version": curr_best_v, "path": audio_map[curr_best_v], "token": token_map.get(curr_best_v, "")},
        "rejected": {"version": curr_worst_v, "path": audio_map[curr_worst_v], "token": token_map.get(curr_worst_v, "")}
    }

async def worker(queue):
    """单个进程的工作协程"""
    connector = aiohttp.TCPConnector(limit=UTTS_PER_WORKER)
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            line = await queue.get()
            if line is None: break
            
            utt_data = json.loads(line)
            res = await process_one_utt(session, utt_data)
            
            if res:
                async with aiofiles.open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                    await f.write(json.dumps(res, ensure_ascii=False) + '\n')
            else:
                async with aiofiles.open(DISCARD_PATH, 'a', encoding='utf-8') as f:
                    await f.write(f"Discarded: {utt_data['utt']}\n")
            queue.task_done()

async def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]

    print(f"🚀 启动多进程并行筛选 | 进程数: {CONCURRENT_WORKERS} | 总任务: {len(lines)}")
    
    queue = asyncio.Queue()
    for line in lines:
        await queue.put(line)
    for _ in range(CONCURRENT_WORKERS):
        await queue.put(None) # 结束标记

    # 创建多个并发 worker
    workers = [asyncio.create_task(worker(queue)) for _ in range(CONCURRENT_WORKERS)]
    
    # 使用 tqdm 监控进度（估算）
    with tqdm(total=len(lines)) as pbar:
        while not queue.empty():
            await asyncio.sleep(1)
            pbar.n = len(lines) - queue.qsize() + CONCURRENT_WORKERS
            pbar.refresh()
        await asyncio.gather(*workers)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 停止。")