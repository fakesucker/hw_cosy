import sys
import os
import io
import torch
import librosa
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from functools import partial

# 添加自定义路径（根据你的实际环境调整）
sys.path.append("/mnt/xkx_data1/work/acc_llasa")
# sys.path.append("/home/A02_tmpdata1/wenhaoli/code/campplus")
sys.path.append("/home/work_nfs23/hkxie/data/lance_test")
from tools import SpkEmbedding
from aslp.data import FloatNPYData
from aslp.data.mp3data import Mp3Data
from aslp.tools import LanceReader

# -------------------------- 配置参数 --------------------------
input_path = '/mnt/xkx_data1/emilia_data_1_mp3data'  # 输入数据根目录（含子文件夹）
base_output_dir = '/home/A02_tmpdata1/wenhaoli/data/spk_emb/cam/10wh_emilia_spkemb_all'  # 基础输出目录
batch_size = 96  # 每个进程的批次大小
target_sr = 24000  # 音频采样率（与模型匹配）
used_gpus = [0,1,2,3]  # 要使用的GPU列表
processes_per_gpu = 8  # 每个GPU分配的进程数
half_precision = True  # 是否使用半精度加速
max_audio_len = None  # 可选：限制最大音频长度
items_per_subdir = 200000  # 每个子文件夹存储的条目数（20万条）
# --------------------------------------------------------------

def get_lance_filelist(input_root):
    """递归遍历输入根目录，获取所有有效 Lance 数据条目（reader, row_idx）"""
    training_files = []
    datatype2class = {"Mp3Data": Mp3Data, "FloatNPYData": FloatNPYData}

    for root, dirs, files in os.walk(input_root):
        dir_name = os.path.basename(root)
        datatype = dir_name.split('_')[-1] if '_' in dir_name else ''
        if datatype not in datatype2class:
            continue

        try:
            reader = LanceReader(root, target_cls=datatype2class[datatype])
            cur_rows = reader.ds.count_rows()
            print(f"加载Lance文件: {root}，共 {cur_rows} 条数据")
            training_files.extend([(reader, i) for i in range(cur_rows)])
        except Exception as e:
            print(f"警告：加载目录 {root} 失败，跳过。错误：{str(e)}")
            continue

    print(f"\n总数据量: {len(training_files)} 条")
    return training_files

def collate_batch(batch_items, target_sr, max_len=None):
    """整理批次数据：加载音频、统一采样率、padding到同长度"""
    wavs = []
    seq_lens = []
    utts = []
    global_indices = []  # 记录每条数据的全局索引（用于划分文件夹）

    # 关键修复：batch_items 每个元素是 (reader, row_idx, global_idx)，需解包3个值
    for item in batch_items:
        reader, row_idx, global_idx = item  # 正确解包3个值
        datas = reader.get_datas_by_rows([row_idx])
        data = datas[0]
        
        if hasattr(data, 'mp3_binary'):
            utt = data.data_id
            wav, sr = librosa.load(io.BytesIO(data.mp3_binary), sr=target_sr, mono=True)
        elif hasattr(data, 'data'):
            utt = data.data_id
            wav = data.data
            sr = target_sr
        else:
            raise ValueError(f"未知数据格式: {data.__dict__.keys()}")

        if max_len is not None and len(wav) > max_len:
            wav = wav[:max_len]
        seq_lens.append(len(wav))
        wavs.append(wav)
        utts.append(utt)
        global_indices.append(global_idx)  # 直接添加全局索引

    # Padding到批次内最大长度
    max_len_in_batch = max(seq_lens) if seq_lens else 0
    padded_wavs = [np.pad(wav, (0, max_len_in_batch - len(wav)), mode='constant') for wav in wavs]
    wav_batch = torch.tensor(np.stack(padded_wavs), dtype=torch.float32)
    
    return wav_batch, seq_lens, utts, global_indices

def process_subset(subset_items_with_global_idx, gpu_id, process_id, base_output_dir, batch_size, target_sr, half_precision, max_audio_len, items_per_subdir):
    """单个进程的处理逻辑：绑定GPU，按20万条划分到子文件夹"""
    # 绑定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    print(f"进程 {process_id}（GPU:{gpu_id}）启动，处理 {len(subset_items_with_global_idx)} 条数据")

    # 初始化模型
    spk_model = SpkEmbedding(device, half=half_precision)
    
    # 按批次处理
    total_batches = (len(subset_items_with_global_idx) + batch_size - 1) // batch_size
    pbar = tqdm(range(total_batches), desc=f"GPU:{gpu_id}-进程{process_id}")
    for batch_idx in pbar:
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(subset_items_with_global_idx))
        batch_items = subset_items_with_global_idx[start:end]

        # 整理批次数据（包含全局索引）
        wav_batch, seq_lens, utts, global_indices = collate_batch(
            batch_items, target_sr, max_audio_len
        )
        wav_batch = wav_batch.to(device)
        if half_precision:
            wav_batch = wav_batch.half()

        # 提取embedding
        with torch.no_grad():
            embs = spk_model.extract_features(wav_batch, seq_lens)

        # 保存到对应子文件夹
        for emb, utt, global_idx in zip(embs, utts, global_indices):
            # 计算当前数据所属的子文件夹编号（0,1,2...）
            subdir_idx = global_idx // items_per_subdir
            subdir_path = os.path.join(base_output_dir, f"{base_output_dir.split('/')[-1]}_{subdir_idx}")
            # 创建子文件夹（不存在则创建）
            Path(subdir_path).mkdir(parents=True, exist_ok=True)
            
            # 保存embedding
            emb_np = emb.cpu().numpy()
            save_path = os.path.join(subdir_path, f"{utt}.npy")
            np.save(save_path, emb_np)
        
        pbar.set_postfix({"已处理": end, "总": len(subset_items_with_global_idx)})
    
    print(f"进程 {process_id}（GPU:{gpu_id}）处理完成！")

def batch_extract_embeddings():
    # 初始化基础输出目录
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    print(f"基础输出目录: {base_output_dir}")
    print(f"配置：使用GPU {used_gpus}，每个GPU {processes_per_gpu} 个进程，总进程数 {len(used_gpus)*processes_per_gpu}")
    print(f"每个子文件夹存储 {items_per_subdir} 条数据")

    # 加载所有数据条目，并添加全局索引（用于划分文件夹）
    all_items = get_lance_filelist(input_path)
    if not all_items:
        print("没有找到有效Lance数据，程序退出")
        return
    # 为每条数据添加全局索引（0,1,2,...），用于判断所属子文件夹
    all_items_with_global_idx = [(item[0], item[1], idx) for idx, item in enumerate(all_items)]

    # 1. 按总进程数均匀分配数据（包含全局索引）
    total_processes = len(used_gpus) * processes_per_gpu
    subset_size = len(all_items_with_global_idx) // total_processes
    remainders = len(all_items_with_global_idx) % total_processes
    subsets = []
    start = 0

    for i in range(total_processes):
        end = start + subset_size + (1 if i < remainders else 0)
        subsets.append(all_items_with_global_idx[start:end])
        start = end

    # 2. 准备进程参数
    process_args = []
    process_id = 0
    for gpu_id in used_gpus:
        for _ in range(processes_per_gpu):
            if process_id < len(subsets):
                process_args.append((
                    subsets[process_id],
                    gpu_id,
                    process_id,
                    base_output_dir,
                    batch_size,
                    target_sr,
                    half_precision,
                    max_audio_len,
                    items_per_subdir
                ))
                process_id += 1

    # 3. 启动多进程并行处理
    print(f"\n启动 {len(process_args)} 个进程并行处理...")
    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=len(process_args)) as pool:
        process_func = partial(process_subset)
        pool.starmap(process_func, process_args)

    # 统计结果
    total_subdirs = (len(all_items) + items_per_subdir - 1) // items_per_subdir
    print(f"\n所有进程处理完成！")
    print(f"总数据量: {len(all_items)} 条")
    print(f"生成子文件夹数量: {total_subdirs} 个（命名格式：{base_output_dir.split('/')[-1]}_0, {base_output_dir.split('/')[-1]}_1...）")
    print(f"基础输出目录: {base_output_dir}")

if __name__ == "__main__":
    # 验证配置合法性
    assert len(used_gpus) > 0, "必须指定至少一个GPU"
    assert processes_per_gpu > 0, "每个GPU的进程数必须大于0"
    assert items_per_subdir > 0, "每个子文件夹的条目数必须大于0"
    # 检查GPU是否可用
    for gpu_id in used_gpus[:]:  # 遍历副本，避免修改迭代中的列表
        if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
            print(f"警告：GPU {gpu_id} 不可用，将跳过")
            used_gpus.remove(gpu_id)
    assert len(used_gpus) > 0, "没有可用的GPU"
    
    batch_extract_embeddings()