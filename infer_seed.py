import argparse
import fcntl
import json
import os
import random
import re
import sys
import tempfile
import time
from datetime import datetime

import torch
import torchaudio
from tqdm import tqdm
import numpy as np

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.streaming_text import iter_stream_text_chunks, split_text_by_punctuation

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def parse_meta_lst_first_valid(meta_file: str):
    """First valid row of meta.lst (for register / auto SFT spk); does not load the whole file into a list."""
    meta_dir = os.path.dirname(meta_file)
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 4:
                continue
            sample_id, prompt_text, prompt_wav_path, tts_text = parts[:4]
            prompt_wav_full_path = os.path.join(meta_dir, prompt_wav_path)
            if not os.path.exists(prompt_wav_full_path):
                continue
            return {
                'id': sample_id,
                'prompt_text': prompt_text,
                'prompt_wav': prompt_wav_full_path,
                'tts_text': tts_text,
            }
    return None


def parse_meta_lst(meta_file: str, shard_index: int = 0, num_shards: int = 1):
    """Parse meta.lst: id|prompt_text|prompt_wav_path|tts_text|[caption].

    When num_shards > 1, only keep valid rows whose global valid-line index satisfies
    idx % num_shards == shard_index (same file, disjoint subsets for parallel workers).
    """
    data_list = []
    meta_dir = os.path.dirname(meta_file)
    global_idx = -1
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 4:
                continue
            sample_id, prompt_text, prompt_wav_path, tts_text = parts[:4]
            prompt_wav_full_path = os.path.join(meta_dir, prompt_wav_path)
            if not os.path.exists(prompt_wav_full_path):
                continue
            global_idx += 1
            if num_shards > 1 and (global_idx % num_shards) != shard_index:
                continue
            data_list.append({
                'id': sample_id,
                'prompt_text': prompt_text,
                'prompt_wav': prompt_wav_full_path,
                'tts_text': tts_text,
            })
    return data_list


def count_meta_valid_samples(meta_file: str) -> int:
    """与 parse_meta_lst 相同的「有效行」计数，不构建列表（大 lst 友好）。"""
    meta_dir = os.path.dirname(meta_file)
    n = 0
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 4:
                continue
            _sid, _pt, prompt_wav_path, _tts = parts[:4]
            prompt_wav_full_path = os.path.join(meta_dir, prompt_wav_path)
            if not os.path.exists(prompt_wav_full_path):
                continue
            n += 1
    return n


def append_speech_token_jsonl(jsonl_path: str, utt: str, tokens: list, text: str, wavpath: str) -> None:
    """每行 {utt, token, text, wavpath}；flock 便于多分片共写同一文件。"""
    rec = {'utt': utt, 'token': tokens, 'text': text, 'wavpath': wavpath}
    line = json.dumps(rec, ensure_ascii=False) + '\n'
    parent = os.path.dirname(os.path.abspath(jsonl_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _tsv_escape(value) -> str:
    if value is None:
        return ''
    s = str(value)
    return s.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')


def init_latency_tsv(tsv_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(tsv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    header = '\t'.join([
        'utt',
        'status',
        'infer_mode',
        'text_input_mode',
        'stream_audio',
        'text_chunk_count',
        'audio_yield_count',
        'first_chunk_latency',
        'first_audio_latency',
        'utterance_done_latency',
        'audio_duration_sec',
        'wav_path',
        'error',
    ]) + '\n'
    with open(tsv_path, 'a+', encoding='utf-8') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                f.write(header)
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_latency_tsv(
    tsv_path: str,
    utt: str,
    status: str,
    infer_mode: str,
    text_input_mode: str,
    stream_audio: bool,
    text_chunk_count: int,
    audio_yield_count: int,
    first_chunk_latency,
    first_audio_latency,
    utterance_done_latency,
    audio_duration_sec,
    wav_path: str,
    error: str = '',
) -> None:
    def _fmt_float(x):
        if x is None:
            return ''
        return f'{float(x):.6f}'

    row = '\t'.join([
        _tsv_escape(utt),
        _tsv_escape(status),
        _tsv_escape(infer_mode),
        _tsv_escape(text_input_mode),
        '1' if stream_audio else '0',
        _tsv_escape(text_chunk_count),
        _tsv_escape(audio_yield_count),
        _fmt_float(first_chunk_latency),
        _fmt_float(first_audio_latency),
        _fmt_float(utterance_done_latency),
        _fmt_float(audio_duration_sec),
        _tsv_escape(wav_path),
        _tsv_escape(error),
    ]) + '\n'
    with open(tsv_path, 'a', encoding='utf-8') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _write_shard_progress(path: str, value: int) -> None:
    if not path:
        return
    tmp = f'{path}.tmp'
    with open(tmp, 'w', encoding='ascii') as f:
        f.write(str(int(value)))
    os.replace(tmp, path)


def reseed_runtime_unfixed() -> int:
    """
    Override yaml-level fixed seeds with a fresh runtime seed.
    """
    seed64 = int.from_bytes(os.urandom(8), byteorder='big', signed=False)
    seed = (seed64 ^ int(time.time_ns()) ^ os.getpid()) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _has_model_yaml(model_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(model_dir, "cosyvoice.yaml"))
        or os.path.exists(os.path.join(model_dir, "cosyvoice2.yaml"))
        or os.path.exists(os.path.join(model_dir, "cosyvoice3.yaml"))
    )


def _build_runtime_model_dir(base_model_dir: str, checkpoint_pt: str) -> str:
    """
    Build a runtime model dir by linking all assets from base_model_dir and
    replacing llm.pt with checkpoint_pt.
    """
    runtime_dir = tempfile.mkdtemp(prefix="cosyvoice_runtime_")
    for name in os.listdir(base_model_dir):
        src = os.path.join(base_model_dir, name)
        dst = os.path.join(runtime_dir, name)
        if name == "llm.pt":
            continue
        try:
            os.symlink(src, dst)
        except Exception:
            # Fallback when symlink is not available for some filesystems.
            if os.path.isdir(src):
                continue
            import shutil
            shutil.copy2(src, dst)
    os.symlink(checkpoint_pt, os.path.join(runtime_dir, "llm.pt"))
    return runtime_dir


def register_spk_from_meta(cosyvoice, meta_file: str, register_spk_id: str):
    ref = parse_meta_lst_first_valid(meta_file)
    if ref is None:
        raise ValueError(f"No valid samples found in meta_file: {meta_file}")
    cosyvoice.add_zero_shot_spk(ref['prompt_text'], ref['prompt_wav'], register_spk_id)
    cosyvoice.save_spkinfo()
    print(f"Registered zero-shot speaker '{register_spk_id}' from first sample in {meta_file}")


def prepare_sft_spk_from_meta(cosyvoice, meta_file: str, sft_spk_id: str):
    """
    Ensure sft_spk_id exists for frontend_sft:
    - register from meta first sample
    - save spk2info.pt
    - reload spk2info.pt into memory
    """
    register_spk_from_meta(cosyvoice, meta_file, sft_spk_id)
    # frontend_sft expects spk2info[spk_id]['embedding'].
    # add_zero_shot_spk may only provide llm_embedding/flow_embedding, so adapt here.
    spk_info = cosyvoice.frontend.spk2info.get(sft_spk_id, {})
    if 'embedding' not in spk_info:
        if 'flow_embedding' in spk_info:
            spk_info['embedding'] = spk_info['flow_embedding']
        elif 'llm_embedding' in spk_info:
            spk_info['embedding'] = spk_info['llm_embedding']
        else:
            raise ValueError(
                f"Registered speaker '{sft_spk_id}' has no embedding keys. "
                f"keys={list(spk_info.keys())}"
            )
        cosyvoice.frontend.spk2info[sft_spk_id] = spk_info
        cosyvoice.save_spkinfo()
        print(f"Adapted spk2info['{sft_spk_id}'] to include 'embedding' for SFT.")

    spk2info_path = os.path.join(cosyvoice.model_dir, "spk2info.pt")
    if os.path.exists(spk2info_path):
        cosyvoice.frontend.spk2info = torch.load(
            spk2info_path, map_location=cosyvoice.frontend.device, weights_only=True
        )
        print(f"Reloaded speaker bank from: {spk2info_path}")


def inference_seed_testset(
    cosyvoice,
    meta_file,
    output_dir,
    infer_mode="zero_shot",
    sft_spk_id="",
    registered_spk_id="",
    is_use_spk_tag=False,
    spk_tag="<|spk_1|>",
    stream=False,
    speed=1.0,
    shard_index: int = 0,
    num_shards: int = 1,
    shard_progress_file: str = '',
    shard_progress_flush_interval: int = 25,
    speech_token_jsonl: str = '',
    stream_text_input: bool = False,
    stream_text_min_tokens: int = 5,
    stream_text_max_tokens: int = 20,
    stream_text_first_chunk_tokens: int = 5,
    stream_text_force_chunk_tokens: int = 30,
    stream_text_debug: bool = False,
    bistream_fixed_ratio: bool = False,
    bistream_text_chunk_tokens: int = 0,
    bistream_first_block_text_tokens: int = 0,
    bistream_first_chunk_text_tokens: int = 0,
    bistream_fixed_ratio_debug: bool = False,
    sentence_pseudo_stream: bool = False,
    sentence_pseudo_stream_debug: bool = False,
    latency_tsv: str = '',
):
    print(f"Loading test set from: {meta_file}")
    data_list = parse_meta_lst(meta_file, shard_index=shard_index, num_shards=num_shards)
    if num_shards > 1:
        print(f"Shard {shard_index}/{num_shards}, local samples: {len(data_list)}")
    else:
        print(f"Total samples: {len(data_list)}")
    os.makedirs(output_dir, exist_ok=True)
    if latency_tsv:
        init_latency_tsv(latency_tsv)
    if bistream_fixed_ratio and hasattr(cosyvoice.model, 'bistream_first_block_text_tokens'):
        cosyvoice.model.bistream_first_block_text_tokens = bistream_first_block_text_tokens

    success_count = 0
    fail_count = 0
    pbar_desc = f"Inference[s={shard_index}/{num_shards}]" if num_shards > 1 else "Inference"
    tqdm_kw = {"mininterval": 0.5}
    if num_shards > 1:
        tqdm_kw = {"mininterval": 2.0, "maxinterval": 10.0}
    use_outer_tqdm = not shard_progress_file
    if shard_progress_file:
        _write_shard_progress(shard_progress_file, 0)
    it = tqdm(data_list, desc=pbar_desc, **tqdm_kw) if use_outer_tqdm else data_list
    for sample in it:
        sample_start_time = time.perf_counter()
        first_chunk_time = None
        first_audio_time = None
        utterance_done_time = None
        text_chunk_count = 0
        audio_yield_count = 0
        audio_duration_sec = None
        wav_path = ''
        text_input_mode = 'whole_text'
        try:
            output_file = os.path.join(output_dir, f"{sample['id']}.wav")
            wav_path = os.path.abspath(output_file)
            tts_text = sample['tts_text']
            debug_chunks = []

            def _normalize_stream_chunk(text: str) -> str:
                return cosyvoice.frontend.text_normalize(text, split=False, text_frontend=True)

            def _build_infer_gen(tts_payload):
                if infer_mode == "zero_shot":
                    return cosyvoice.inference_zero_shot(
                        tts_payload,
                        sample['prompt_text'],
                        sample['prompt_wav'],
                        stream=stream,
                        speed=speed,
                    )
                if infer_mode == "sft":
                    return cosyvoice.inference_sft(
                        tts_payload,
                        sft_spk_id,
                        stream=stream,
                        speed=speed,
                    )
                if infer_mode == "registered_spk":
                    return cosyvoice.inference_zero_shot(
                        tts_payload,
                        '',
                        '',
                        zero_shot_spk_id=registered_spk_id,
                        stream=stream,
                        speed=speed,
                    )
                raise ValueError(f"Unsupported infer_mode: {infer_mode}")

            spk_prefix = ""
            if tts_text.startswith("<|spk_"):
                m = re.match(r"^(\<\|spk_[^\|]+\|\>)\s*", tts_text)
                if m is not None:
                    spk_prefix = m.group(1)
                    tts_text_core = tts_text[m.end():]
                else:
                    tts_text_core = tts_text
            else:
                tts_text_core = tts_text
            if is_use_spk_tag:
                spk_prefix = spk_tag

            segment_gens = []
            if bistream_fixed_ratio:
                text_input_mode = 'bistream_fixed_ratio'
                mix_ratio = getattr(cosyvoice.model.llm, 'mix_ratio', [5, 15])
                text_tokens_per_chunk = bistream_text_chunk_tokens if bistream_text_chunk_tokens > 0 else mix_ratio[0]
                first_block_text_tokens = (
                    bistream_first_block_text_tokens
                    if bistream_first_block_text_tokens > 0
                    else mix_ratio[0]
                )
                if bistream_first_block_text_tokens > 0:
                    first_chunk_text_tokens = bistream_first_block_text_tokens
                elif bistream_first_chunk_text_tokens > 0:
                    first_chunk_text_tokens = bistream_first_chunk_text_tokens
                else:
                    first_chunk_text_tokens = text_tokens_per_chunk
                tts_input = tts_text_core
                if spk_prefix and tts_input and not tts_input.startswith("<|spk_"):
                    tts_input = f"{spk_prefix}{tts_input}"

                def _bistream_fixed_ratio_generator():
                    nonlocal first_chunk_time, text_chunk_count
                    for chunk_tok in cosyvoice.frontend.iter_bistream_fixed_ratio_text_tokens(
                        tts_input,
                        text_tokens_per_chunk=text_tokens_per_chunk,
                        first_chunk_text_tokens=first_chunk_text_tokens,
                        text_frontend=True,
                    ):
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter()
                        text_chunk_count += 1
                        if bistream_fixed_ratio_debug:
                            debug_chunks.append(chunk_tok.detach().cpu().tolist())
                        yield chunk_tok

                if bistream_fixed_ratio_debug:
                    print(
                        f"[bistream_fixed_ratio] sample={sample['id']} "
                        f"first_block=[{first_block_text_tokens}:{mix_ratio[1]}] "
                        f"steady_block=[{text_tokens_per_chunk}:{mix_ratio[1]}] "
                        f"first_chunk_text_tokens={first_chunk_text_tokens}"
                    )
                segment_gens.append(_build_infer_gen(_bistream_fixed_ratio_generator()))
            elif stream_text_input:
                text_input_mode = 'bistream_text_stream'

                def _text_stream_generator():
                    nonlocal first_chunk_time, text_chunk_count
                    for chunk in iter_stream_text_chunks(
                        text=tts_text_core,
                        tokenize=lambda s: cosyvoice.frontend.tokenizer.encode(
                            s, allowed_special=cosyvoice.frontend.allowed_special
                        ),
                        normalize=_normalize_stream_chunk,
                        min_chunk_tokens=stream_text_min_tokens,
                        max_chunk_tokens=stream_text_max_tokens,
                        first_chunk_tokens=stream_text_first_chunk_tokens,
                        force_chunk_tokens=stream_text_force_chunk_tokens,
                        spk_tag=spk_prefix,
                    ):
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter()
                        text_chunk_count += 1
                        if stream_text_debug:
                            debug_chunks.append(chunk)
                        yield chunk

                if stream_text_debug:
                    print(f"[stream_text] sample={sample['id']} raw_text={tts_text}")
                segment_gens.append(_build_infer_gen(_text_stream_generator()))
            elif sentence_pseudo_stream:
                text_input_mode = 'sentence_pseudo_stream'
                sentence_chunks = split_text_by_punctuation(tts_text_core)
                if sentence_pseudo_stream_debug:
                    print(f"[sentence_pseudo_stream] sample={sample['id']} raw_text={tts_text}")
                    print(f"[sentence_pseudo_stream] sample={sample['id']} chunks={sentence_chunks}")
                for sentence in sentence_chunks:
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                    if spk_prefix and not sentence.startswith("<|spk_"):
                        sentence = f"{spk_prefix}{sentence}"
                    debug_chunks.append(sentence)
                    text_chunk_count += 1
                    segment_gens.append(_build_infer_gen(sentence))
            else:
                tts_input = tts_text_core
                if spk_prefix and tts_input and not tts_input.startswith("<|spk_"):
                    tts_input = f"{spk_prefix}{tts_input}"
                first_chunk_time = sample_start_time
                text_chunk_count = 1
                segment_gens.append(_build_infer_gen(tts_input))

            # 流式多包或 text_normalize 拆句时多次 yield：拼接波形；LLM 给 token2wav 的 token 按段 extend 成一条。
            wav_parts = []
            speech_tokens_flat: list = []
            sr = cosyvoice.sample_rate
            for gen in segment_gens:
                for model_output in gen:
                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()
                    audio_yield_count += 1
                    wav_parts.append(model_output['tts_speech'].detach().cpu())
                    tok = model_output.get('tts_speech_token')
                    if tok is not None:
                        if isinstance(tok, list):
                            speech_tokens_flat.extend(int(x) for x in tok)
                        elif hasattr(tok, 'tolist'):
                            speech_tokens_flat.extend(int(x) for x in tok.detach().cpu().reshape(-1).tolist())
                        else:
                            speech_tokens_flat.extend(int(x) for x in tok)
            if stream_text_input and stream_text_debug:
                print(f"[stream_text] sample={sample['id']} chunks={debug_chunks}")
            if bistream_fixed_ratio and bistream_fixed_ratio_debug:
                print(f"[bistream_fixed_ratio] sample={sample['id']} chunks={debug_chunks}")
            if not wav_parts:
                raise ValueError('empty generation')
            utterance_done_time = time.perf_counter()
            full_wav = torch.cat(wav_parts, dim=1)
            audio_duration_sec = full_wav.shape[1] / sr
            torchaudio.save(output_file, full_wav, sr)
            if speech_token_jsonl:
                append_speech_token_jsonl(
                    speech_token_jsonl,
                    sample['id'],
                    speech_tokens_flat,
                    tts_text,
                    os.path.abspath(output_file),
                )
            if latency_tsv:
                append_latency_tsv(
                    latency_tsv,
                    utt=sample['id'],
                    status='ok',
                    infer_mode=infer_mode,
                    text_input_mode=text_input_mode,
                    stream_audio=stream,
                    text_chunk_count=text_chunk_count,
                    audio_yield_count=audio_yield_count,
                    first_chunk_latency=None if first_chunk_time is None else first_chunk_time - sample_start_time,
                    first_audio_latency=None if first_audio_time is None else first_audio_time - sample_start_time,
                    utterance_done_latency=None if utterance_done_time is None else utterance_done_time - sample_start_time,
                    audio_duration_sec=audio_duration_sec,
                    wav_path=wav_path,
                )
            success_count += 1
            if shard_progress_file and shard_progress_flush_interval > 0:
                if success_count % shard_progress_flush_interval == 0:
                    _write_shard_progress(shard_progress_file, success_count)
        except Exception as e:
            print(f"Error processing sample {sample['id']}: {e}")
            if latency_tsv:
                append_latency_tsv(
                    latency_tsv,
                    utt=sample['id'],
                    status='fail',
                    infer_mode=infer_mode,
                    text_input_mode=text_input_mode,
                    stream_audio=stream,
                    text_chunk_count=text_chunk_count,
                    audio_yield_count=audio_yield_count,
                    first_chunk_latency=None if first_chunk_time is None else first_chunk_time - sample_start_time,
                    first_audio_latency=None if first_audio_time is None else first_audio_time - sample_start_time,
                    utterance_done_latency=None if utterance_done_time is None else utterance_done_time - sample_start_time,
                    audio_duration_sec=audio_duration_sec,
                    wav_path=wav_path,
                    error=str(e),
                )
            fail_count += 1

    if shard_progress_file:
        _write_shard_progress(shard_progress_file, success_count)

    print("\n" + "=" * 50)
    print("Inference completed!")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output directory: {output_dir}")
    if latency_tsv:
        print(f"Latency TSV: {latency_tsv}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='CosyVoice original-style inference on meta.lst')
    parser.add_argument('--meta_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--count_meta_only', action='store_true',
                        help='Print count of valid meta rows and exit (no model load)')
    parser.add_argument('--model_dir', type=str, default='',
                        help='Model dir that AutoModel can load (contains cosyvoice*.yaml and llm/flow/hift pt)')
    parser.add_argument('--base_model_dir', type=str, default='',
                        help='Base pretrained model dir with yaml/flow/hift assets (used with --checkpoint_pt)')
    parser.add_argument('--infer_mode', type=str, default='zero_shot',
                        choices=['zero_shot', 'sft', 'registered_spk'],
                        help='zero_shot: per-utt prompt from lst; sft: inference_sft with --sft_spk_id; '
                             'registered_spk: register speaker then inference_zero_shot by spk2info id')
    parser.add_argument('--is_sft', action='store_true',
                        help='Compatibility switch: equivalent to --infer_mode sft')
    parser.add_argument('--sft_spk_id', type=str, default='',
                        help='Required when --infer_mode sft')
    parser.add_argument('--auto_fallback_registered_spk', action='store_true',
                        help='Deprecated compatibility flag. Kept for old scripts.')
    parser.add_argument('--auto_register_sft_spk_from_meta', action='store_true',
                        help='If sft_spk_id not found, auto register it from meta first sample and keep infer_mode=sft')
    parser.add_argument('--register_spk_id', type=str, default='',
                        help='Speaker id used by add_zero_shot_spk; required for --infer_mode registered_spk')
    parser.add_argument('--register_from_meta_first', action='store_true',
                        help='Use first line of meta_file (prompt_text/prompt_wav) to register --register_spk_id')
    parser.add_argument('--checkpoint_pt', type=str, default='',
                        help='Optional llm checkpoint (.pt). If model_dir has no yaml, requires --base_model_dir.')
    parser.add_argument('--is_use_spk_tag', action='store_true',
                        help='If set, prepend spk tag to each tts_text')
    parser.add_argument('--spk_tag', type=str, default='<|spk_1|>',
                        help='Prefix tag used when --is_use_spk_tag is enabled')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--stream_text_input', action='store_true',
                        help='Chunk tts_text into a generator to trigger CosyVoice2/3 bi-stream text input path')
    parser.add_argument('--stream_text_min_tokens', type=int, default=5,
                        help='Minimum text tokens before trying to commit one text chunk')
    parser.add_argument('--stream_text_max_tokens', type=int, default=20,
                        help='Preferred text chunk size in tokenizer tokens')
    parser.add_argument('--stream_text_first_chunk_tokens', type=int, default=5,
                        help='Preferred first text chunk size in tokenizer tokens')
    parser.add_argument('--stream_text_force_chunk_tokens', type=int, default=30,
                        help='Force commit once buffered text exceeds this many tokens, unless protected suffix is active')
    parser.add_argument('--stream_text_debug', action='store_true',
                        help='Print per-utterance stream text chunks for debugging')
    parser.add_argument('--bistream_fixed_ratio', action='store_true',
                        help='Fixed text:speech bi-stream (mix_ratio[0] tokens/chunk); CosyVoice2/3, no vLLM')
    parser.add_argument('--bistream_text_chunk_tokens', type=int, default=0,
                        help='Steady-state text tokens per bi-stream block (0 => llm.mix_ratio[0], default 5)')
    parser.add_argument('--bistream_speech_chunk_tokens', type=int, default=0,
                        help='Speech tokens per bi-stream block (0 => llm.mix_ratio[1]; use 10 for 5:10 trained ckpts)')
    parser.add_argument('--bistream_first_block_text_tokens', type=int, default=0,
                        help='First TTS text block size in inference_bistream (0 => 5, try 6 or 10 for [N:15] then [5:15])')
    parser.add_argument('--bistream_first_chunk_text_tokens', type=int, default=0,
                        help='Frontend first generator chunk size (0 => follow first_block or chunk size)')
    parser.add_argument('--bistream_fixed_ratio_debug', action='store_true',
                        help='Print per-utterance fixed-ratio bi-stream token chunks')
    parser.add_argument('--sentence_pseudo_stream', action='store_true',
                        help='Baseline: split text by punctuation and run multiple independent inference_* calls')
    parser.add_argument('--sentence_pseudo_stream_debug', action='store_true',
                        help='Print per-utterance punctuation sentence chunks for debugging')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--load_jit', action='store_true')
    parser.add_argument('--load_trt', action='store_true')
    parser.add_argument('--load_vllm', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Split meta: worker keeps valid lines with idx %% num_shards == shard_index')
    parser.add_argument('--shard_index', type=int, default=0,
                        help='Which shard [0, num_shards); use with --num_shards for multi-GPU parallel infer')
    parser.add_argument('--shard_progress_file', type=str, default='',
                        help='Atomic counter file for cross-process total progress (one file per shard)')
    parser.add_argument('--shard_progress_flush_interval', type=int, default=25,
                        help='Update shard_progress_file every N successful saves')
    parser.add_argument('--speech_token_jsonl', type=str, default='',
                        help='If set, append one JSON line per utterance: utt, token (int list), text, wavpath')
    parser.add_argument('--latency_tsv', type=str, default='',
                        help='If set, append per-sample latency metrics to this TSV; default is <output_dir>/latency.tsv')
    parser.add_argument('--unfixed_seed', action='store_true',
                        help='Reseed runtime after model load to disable yaml fixed seed reproducibility')
    args = parser.parse_args()
    if args.count_meta_only:
        print(count_meta_valid_samples(args.meta_file))
        return
    if not args.model_dir:
        raise ValueError("--model_dir is required unless --count_meta_only")
    if args.is_sft:
        args.infer_mode = 'sft'
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(f"--shard_index must be in [0, {args.num_shards}), got {args.shard_index}")
    if args.stream_text_input and args.sentence_pseudo_stream:
        raise ValueError("--stream_text_input and --sentence_pseudo_stream are mutually exclusive")
    if args.bistream_fixed_ratio and args.stream_text_input:
        raise ValueError("--bistream_fixed_ratio and --stream_text_input are mutually exclusive")
    if args.bistream_fixed_ratio and args.sentence_pseudo_stream:
        raise ValueError("--bistream_fixed_ratio and --sentence_pseudo_stream are mutually exclusive")
    if args.bistream_first_chunk_text_tokens < 0:
        raise ValueError("--bistream_first_chunk_text_tokens must be >= 0")
    if args.bistream_first_block_text_tokens < 0:
        raise ValueError("--bistream_first_block_text_tokens must be >= 0")
    if args.bistream_first_block_text_tokens not in (0, 6, 7, 10):
        raise ValueError("--bistream_first_block_text_tokens must be 0, 6, or 10")
    if args.bistream_speech_chunk_tokens < 0:
        raise ValueError("--bistream_speech_chunk_tokens must be >= 0")

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.meta_file),
        f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    )
    latency_tsv = args.latency_tsv or os.path.join(output_dir, 'latency.tsv')

    runtime_model_dir = args.model_dir
    if args.checkpoint_pt:
        if not os.path.isfile(args.checkpoint_pt):
            raise ValueError(f"--checkpoint_pt not found: {args.checkpoint_pt}")
        if not args.base_model_dir:
            raise ValueError("--base_model_dir is required when --checkpoint_pt is set")
        if not _has_model_yaml(args.base_model_dir):
            raise ValueError(f"--base_model_dir has no cosyvoice*.yaml: {args.base_model_dir}")
        runtime_model_dir = _build_runtime_model_dir(args.base_model_dir, args.checkpoint_pt)
        print(f"Using runtime model dir: {runtime_model_dir}")
        print(f"Using llm checkpoint: {args.checkpoint_pt}")
    elif not _has_model_yaml(args.model_dir):
        raise ValueError(
            f"--model_dir has no cosyvoice*.yaml: {args.model_dir}. "
            "Please pass a pretrained dir, or use --base_model_dir + --checkpoint_pt"
        )

    cosyvoice = AutoModel(
        model_dir=runtime_model_dir,
        load_jit=args.load_jit,
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
        fp16=args.fp16,
    )
    if args.bistream_speech_chunk_tokens > 0:
        base_mix_ratio = getattr(cosyvoice.model.llm, 'mix_ratio', [5, 15])
        text_ratio = args.bistream_text_chunk_tokens if args.bistream_text_chunk_tokens > 0 else base_mix_ratio[0]
        cosyvoice.model.llm.mix_ratio = [text_ratio, args.bistream_speech_chunk_tokens]
        print(f"Override bistream mix_ratio to {cosyvoice.model.llm.mix_ratio} (text:speech)")
    if args.unfixed_seed:
        seed = reseed_runtime_unfixed()
        print(f"Runtime reseeded (unfixed): {seed}")
    available_spks = cosyvoice.list_available_spks()
    print(f"Available speakers ({len(available_spks)}): {available_spks[:20]}")

    if args.infer_mode == 'sft' and not args.sft_spk_id:
        raise ValueError("--sft_spk_id is required when --infer_mode sft")
    if args.infer_mode == 'sft' and args.sft_spk_id not in available_spks:
        # Backward compatible: old flag also enables this behavior.
        if args.auto_register_sft_spk_from_meta or args.auto_fallback_registered_spk:
            print(
                f"[WARN] sft_spk_id '{args.sft_spk_id}' not found. "
                f"Auto register from meta and keep infer_mode=sft."
            )
            prepare_sft_spk_from_meta(cosyvoice, args.meta_file, args.sft_spk_id)
            available_spks = cosyvoice.list_available_spks()
            print(f"Available speakers after auto register ({len(available_spks)}): {available_spks[:20]}")
            if args.sft_spk_id not in available_spks:
                raise ValueError(
                    f"auto register failed, sft_spk_id '{args.sft_spk_id}' still missing. "
                    f"Available: {available_spks[:50]}"
                )
        else:
            raise ValueError(
                f"sft_spk_id '{args.sft_spk_id}' not found in spk2info. "
                f"Available: {available_spks[:50]}"
            )
    if args.infer_mode == 'registered_spk' and not args.register_spk_id:
        raise ValueError("--register_spk_id is required when --infer_mode registered_spk")
    if args.register_from_meta_first:
        if not args.register_spk_id:
            raise ValueError("--register_spk_id is required when --register_from_meta_first is used")
        register_spk_from_meta(cosyvoice, args.meta_file, args.register_spk_id)

    inference_seed_testset(
        cosyvoice=cosyvoice,
        meta_file=args.meta_file,
        output_dir=output_dir,
        infer_mode=args.infer_mode,
        sft_spk_id=args.sft_spk_id,
        registered_spk_id=args.register_spk_id,
        is_use_spk_tag=args.is_use_spk_tag,
        spk_tag=args.spk_tag,
        stream=args.stream,
        speed=args.speed,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        shard_progress_file=args.shard_progress_file,
        shard_progress_flush_interval=args.shard_progress_flush_interval,
        speech_token_jsonl=args.speech_token_jsonl,
        stream_text_input=args.stream_text_input,
        stream_text_min_tokens=args.stream_text_min_tokens,
        stream_text_max_tokens=args.stream_text_max_tokens,
        stream_text_first_chunk_tokens=args.stream_text_first_chunk_tokens,
        stream_text_force_chunk_tokens=args.stream_text_force_chunk_tokens,
        stream_text_debug=args.stream_text_debug,
        bistream_fixed_ratio=args.bistream_fixed_ratio,
        bistream_text_chunk_tokens=args.bistream_text_chunk_tokens,
        bistream_first_block_text_tokens=args.bistream_first_block_text_tokens,
        bistream_first_chunk_text_tokens=args.bistream_first_chunk_text_tokens,
        bistream_fixed_ratio_debug=args.bistream_fixed_ratio_debug,
        sentence_pseudo_stream=args.sentence_pseudo_stream,
        sentence_pseudo_stream_debug=args.sentence_pseudo_stream_debug,
        latency_tsv=latency_tsv,
    )


if __name__ == '__main__':
    main()
