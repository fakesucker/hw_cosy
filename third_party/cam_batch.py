import sys
import os
import io
import torch
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path

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
output_dir = '/home/A02_tmpdata1/wenhaoli/data/spk_emb/cam/10wh_emilia_spkemb'  # emb保存目录
batch_size = 128  # 固定批次大小
target_sr = 24000  # 音频采样率（与模型匹配）
device = 'cuda:7'  # 运行设备
half_precision = True  # 是否使用半精度加速
max_audio_len = None  # 可选：限制最大音频长度（None表示不限制）
# --------------------------------------------------------------

def get_lance_filelist(input_root):
    """
    递归遍历输入根目录下所有子文件夹，获取所有有效 Lance 数据条目（reader, row_idx）
    Args:
        input_root: 输入数据根目录
    Returns:
        training_files: 列表，每个元素为 (reader, row_idx)
    """
    training_files = []
    datatype2class = {
        "Mp3Data": Mp3Data,
        "FloatNPYData": FloatNPYData,
    }

    # 递归遍历所有子目录，查找有效 Lance 文件夹
    for root, dirs, files in os.walk(input_root):
        # 判断当前目录是否为 Lance 数据目录（根据后缀判断，如 "_Mp3Data" 结尾）
        dir_name = os.path.basename(root)
        datatype = dir_name.split('_')[-1] if '_' in dir_name else ''
        if datatype not in datatype2class:
            continue  # 跳过非目标数据目录

        # 加载当前 Lance 目录
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
    """
    整理批次数据：加载音频、统一采样率、padding到同长度
    Args:
        batch_items: 列表，每个元素为 (reader, row_idx)
        target_sr: 目标采样率
        max_len: 最大音频长度（超过则截断）
    Returns:
        wav_batch: 形状为 (B, T) 的张量（已padding）
        seq_lens: 列表，每个元素为原始音频长度（未padding）
        utts: 列表，utt标识符
    """
    wavs = []
    seq_lens = []
    utts = []

    for reader, row_idx in batch_items:
        # 读取数据
        datas = reader.get_datas_by_rows([row_idx])
        data = datas[0]
        
        # 解析音频和utt
        if hasattr(data, 'mp3_binary'):
            utt = data.data_id
            # 从mp3二进制数据加载音频
            wav, sr = librosa.load(io.BytesIO(data.mp3_binary), sr=target_sr, mono=True)
        elif hasattr(data, 'data'):
            utt = data.data_id
            # 从npy数据加载（假设是已经处理好的音频）
            wav = data.data
            sr = target_sr  # 假设npy数据已匹配采样率
        else:
            raise ValueError(f"未知数据格式: {data.__dict__.keys()}")

        # 处理音频长度
        if max_len is not None and len(wav) > max_len:
            wav = wav[:max_len]
        seq_lens.append(len(wav))
        wavs.append(wav)
        utts.append(utt)

    # 计算批次内最大长度，进行padding
    max_len_in_batch = max(seq_lens)
    padded_wavs = []
    for wav in wavs:
        pad_len = max_len_in_batch - len(wav)
        padded_wav = np.pad(wav, (0, pad_len), mode='constant')  # 末尾补0
        padded_wavs.append(padded_wav)

    # 转为张量并移动到设备
    wav_batch = torch.tensor(np.stack(padded_wavs), dtype=torch.float32).to(device)
    # if half_precision:
    #     wav_batch = wav_batch.half()  # 半精度加速

    return wav_batch, seq_lens, utts

def batch_extract_embeddings():
    # 初始化输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"embedding将保存到: {output_dir}")

    # 加载数据列表（递归遍历所有子文件夹）
    all_items = get_lance_filelist(input_path)
    if not all_items:
        print("没有找到有效Lance数据，程序退出")
        return

    # 初始化说话人模型
    spk_model = SpkEmbedding(device, half=half_precision)
    print(f"使用设备: {device}，半精度模式: {half_precision}")

    # 按批次处理
    total_batches = (len(all_items) + batch_size - 1) // batch_size  # 向上取整
    for batch_idx in tqdm(range(total_batches), desc="批量提取embedding"):
        # 截取当前批次的条目
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(all_items))
        batch_items = all_items[start:end]
        # import pdb;pdb.set_trace()
        # 整理批次数据（加载+padding）
        wav_batch, seq_lens, utts = collate_batch(
            batch_items,
            target_sr=target_sr,
            max_len=max_audio_len
        )

        # 提取embedding
        with torch.no_grad():  # 关闭梯度计算，节省内存
            embs = spk_model.extract_features(wav_batch, seq_lens)  # 形状: (B, emb_dim)

        # 保存每个utt的embedding（格式：npy）
        for emb, utt in zip(embs, utts):
            emb_np = emb.cpu().numpy()  # 转移到CPU并转为numpy
            save_path = os.path.join(output_dir, f"{utt}.npy")
            np.save(save_path, emb_np)

    print(f"\n所有embedding提取完成，共 {len(all_items)} 条，保存至 {output_dir}")

if __name__ == "__main__":
    batch_extract_embeddings()