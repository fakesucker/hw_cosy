# 客服语音场景对比打分网页

这是一个基于 Streamlit 的语音主观评测工具，用于对**同一场景**下不同模型生成的客服语音进行对比打分。

## 当前功能

- 按场景加载语音（场景 ID 如 `001_000000`）
- 同屏对比两个模型音频
- 展示对应场景文本（来自 `kefu_scene_dialogue.jsonl`）
- 三维 0-5 分打分：
  - 自然度
  - 客服度
  - 可懂度
- 自动计算总分（均值）
- 评测结果保存到 `backend/data/results.json`
- 支持结果汇总与 JSON/CSV 导出

## 项目结构

```text
web/
├── backend/
│   ├── app.py                  # Streamlit 主程序
│   ├── data/
│   │   └── results.json        # 评测结果
│   └── utils/
├── run.sh                      # 启动脚本
├── assets/                     # 预留目录（当前流程不依赖）
└── yuzhou_data/
    ├── text/
    │   ├── kefu.lst
    │   └── kefu_scene_dialogue.jsonl
    └── test_wav/               # 当前音频目录（也兼容 text_wav）
        ├── kefu_0421_onlyhw_niren/
        └── kefu_0421_onlymale/
```

## 数据约定

### 1) 场景文本

- 文件：`yuzhou_data/text/kefu_scene_dialogue.jsonl`
- 每行一个 JSON，至少包含：
  - `scene`: 场景 ID（例如 `001_000000`）
  - `dialogue`: 文本内容

### 2) 音频目录

- 程序会优先读取：`yuzhou_data/text_wav`
- 若不存在，会回退读取：`yuzhou_data/test_wav`
- 会自动扫描“最底层且包含 `.wav` 文件”的目录作为可选模型
- 音频文件名（不含后缀）应与 `scene` 一致，例如：`001_000000.wav`

## 运行方式

### 方式一：脚本启动（推荐）

```bash
cd /home/work_nfs22/xmren/code/web
bash run.sh
```

### 方式二：手动启动

```bash
cd /home/work_nfs22/xmren/code/web
python -m streamlit run backend/app.py
```

启动后在浏览器打开终端输出中的本地地址（一般是 `http://localhost:8501`）。

## 网页使用说明

1. 左侧填写“评测人”。
2. 在左侧选择两个不同模型（两个最底层音频目录）。
3. 在主页面切换场景，试听两路音频。
4. 结合场景文本，对每个模型分别打三维分数（0-5）。
5. 点击“提交本场景评分”保存。
6. 在 `Results Summary` 查看统计，在 `Data Export` 导出结果。

## 常见问题

### 1) 页面提示找不到音频目录

请检查以下目录至少存在一个：

- `yuzhou_data/text_wav`
- `yuzhou_data/test_wav`

### 2) 两个模型没有可对齐场景

说明两目录下的 `.wav` 文件名（场景 ID）交集为空，请检查文件命名是否一致。

### 3) 页面提示找不到场景文本

请确认 `yuzhou_data/text/kefu_scene_dialogue.jsonl` 存在且为合法 JSONL。

## 结果文件说明

- 结果文件：`backend/data/results.json`
- 每次提交会追加一条记录，包含：
  - 评测人
  - 场景 ID
  - 两个模型的三维分与总分
  - 时间戳