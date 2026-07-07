# AB-test 盲测打包说明

## 现有 `mix_samples_mapping.csv` 在做什么

| 列 | 含义 |
|----|------|
| `mixed_filename` | 盲测用的最终文件名：`{原utt_id}_{随机6位}.wav`，例如 `001_000000_9ita3C.wav` |
| `suffix` | 随机后缀（与 `mixed_filename` 中一致），用于区分 A/B，听众无法从后缀看出模型 |
| `source_model_folder` | 模型侧短目录名（如 `shenhuonly`、`bigbatch`），仅作记录 |
| `source_epoch_folder` | 如 `epoch_2_whole` / `epoch_4_whole` |
| `source_filename` | 原始 wav 名，如 `001_000000.wav` |
| `source_full_path` | 拷贝来源的绝对路径 |

**规则小结**：

1. 两个模型各自有一条与 **同一 utterance id** 对应的 wav（如都来自 `001_000000.wav`）。
2. 为 **每一条、每一侧** 独立采样一个 **6 位** `[A-Za-z0-9]` 后缀（同一次 batch 内保证不重复）。
3. 重命名：`{stem}_{suffix}.wav`，其中 `stem` 为去掉 `.wav` 的文件名。
4. CSV 记录「混音名 → 真实路径」，便于评测后反查 A/B。

## 如何为新的一对推理目录再生成一版

在仓库内使用脚本（需本机能访问两个 epoch 目录下的 wav）：

```bash
cd /path/to/CosyVoice
python3 data_list/AB-test/build_ab_mix.py \
  --label-a sft_shenhu_only \
  --wav-dir-a /绝对路径/.../sft_shenhu_only_1e-5_from_llm/epoch_4_whole \
  --label-b dpo_bigbatch \
  --wav-dir-b /绝对路径/.../dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch/epoch_4_whole \
  --csv-out data_list/AB-test/mix_samples_mapping_kefu0506.csv \
  --copy-dest data_list/AB-test/mixed_kefu0506
```

- 默认只取 **两边都存在的** `*.wav` stem 做交集。
- 不加 `--copy-dest` 则只写 CSV，不拷贝文件。

将上述路径换成你的「第二套数据」对应目录即可。
