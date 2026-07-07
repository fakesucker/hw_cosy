#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpkEmbedding.extract_features 使用说明和示例

参数分析：
1. wavs: List[torch.Tensor] - 波形列表
   - 每个元素是一个 torch.Tensor，形状为 (length,) 或 (channels, length)
   - 值范围通常在 [-1.0, 1.0] 之间（归一化的音频）
   - 采样率应该是 16000 Hz

2. wav_lengths: List[int] | torch.Tensor - 每个波形的实际长度
   - 如果波形被padding到相同长度，wav_lengths 指定实际有效长度
   - 如果波形没有被padding，wav_lengths 应该等于每个波形的 shape[0] 或 shape[1]

返回值：
- spkembeds: torch.Tensor，形状为 (batch_size, embedding_size=192)
  - 每个样本的说话人嵌入向量
"""

import torch
import torchaudio
from cosyvoice.third_party.campplus.tools import SpkEmbedding

# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
spk_embedding = SpkEmbedding(device=device, half=False)


# ========== 示例1: 从音频文件批量加载 ==========
def example1_load_from_files(wav_paths):
    """
    从多个音频文件加载并提取说话人嵌入
    
    Args:
        wav_paths: List[str] - 音频文件路径列表
    """
    wavs = []
    wav_lengths = []
    
    for wav_path in wav_paths:
        # 加载音频文件，返回 (waveform, sample_rate)
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # 转换为单声道（如果是多声道）
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样到 16000 Hz（如果需要）
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # 转换为1D tensor（去除channel维度）
        waveform = waveform.squeeze(0)  # shape: (length,)
        
        # 归一化到 [-1, 1]（如果还没有）
        max_val = waveform.abs().max()
        if max_val > 1.0:
            waveform = waveform / max_val
        
        wavs.append(waveform)
        wav_lengths.append(waveform.shape[0])
    
    # 批量提取特征
    embeddings = spk_embedding.extract_features(wavs, wav_lengths)
    # embeddings shape: (batch_size, 192)
    
    return embeddings


# ========== 示例2: 从已加载的tensor批量处理 ==========
def example2_batch_from_tensors(waveforms, actual_lengths=None):
    """
    从已加载的波形tensor批量提取特征
    
    Args:
        waveforms: List[torch.Tensor] - 波形列表，每个tensor shape为 (length,)
        actual_lengths: List[int] | None - 实际长度列表
                       如果为None，则使用每个waveform的长度
    """
    if actual_lengths is None:
        actual_lengths = [w.shape[0] for w in waveforms]
    
    # 确保所有波形在同一设备上
    wavs = [w.to(device) for w in waveforms]
    wav_lengths = [int(l) for l in actual_lengths]
    
    # 批量提取特征
    embeddings = spk_embedding.extract_features(wavs, wav_lengths)
    
    return embeddings


# ========== 示例3: 处理已padding的批次 ==========
def example3_padded_batch(waveforms_padded, wav_lengths):
    """
    处理已经被padding到相同长度的批次
    
    Args:
        waveforms_padded: torch.Tensor - shape (batch_size, max_length)
        wav_lengths: List[int] | torch.Tensor - 每个波形的实际长度
    """
    # 将padded tensor转换为列表
    batch_size = waveforms_padded.shape[0]
    wavs = [waveforms_padded[i] for i in range(batch_size)]
    
    # 确保wav_lengths是列表
    if isinstance(wav_lengths, torch.Tensor):
        wav_lengths = wav_lengths.tolist()
    
    # 批量提取特征
    embeddings = spk_embedding.extract_features(wavs, wav_lengths)
    
    return embeddings


# ========== 示例4: 单个样本处理 ==========
def example4_single_sample(waveform):
    """
    处理单个样本（内部会转换为batch）
    
    Args:
        waveform: torch.Tensor - shape (length,)
    """
    # 单个样本也需要包装成列表
    wavs = [waveform]
    wav_lengths = [waveform.shape[0]]
    
    embeddings = spk_embedding.extract_features(wavs, wav_lengths)
    # 返回单个嵌入向量
    return embeddings[0]  # shape: (192,)


# ========== 完整使用示例 ==========
if __name__ == "__main__":
    # 示例：创建一些模拟的音频数据
    batch_size = 4
    sample_rate = 16000
    duration = 2.0  # 秒
    
    # 创建随机波形（模拟音频）
    wavs = []
    wav_lengths = []
    
    for i in range(batch_size):
        # 每个样本长度略有不同
        length = int(sample_rate * duration) + torch.randint(-1000, 1000, (1,)).item()
        waveform = torch.randn(length) * 0.3  # 随机噪声，模拟音频
        waveform = torch.clamp(waveform, -1.0, 1.0)  # 限制在 [-1, 1]
        
        wavs.append(waveform)
        wav_lengths.append(length)
    
    print(f"批次大小: {batch_size}")
    print(f"波形长度: {wav_lengths}")
    
    # 提取说话人嵌入
    embeddings = spk_embedding.extract_features(wavs, wav_lengths)
    
    print(f"\n提取的嵌入形状: {embeddings.shape}")
    print(f"嵌入向量示例（前5维）: {embeddings[0, :5]}")
    
    # 如果需要计算相似度
    # 可以使用余弦相似度
    import torch.nn.functional as F
    normalized_embeds = F.normalize(embeddings, dim=1)
    similarity_matrix = torch.mm(normalized_embeds, normalized_embeds.t())
    print(f"\n相似度矩阵形状: {similarity_matrix.shape}")

