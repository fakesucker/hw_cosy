#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 可选：如果需要 conda 环境，可在外部先激活后再执行本脚本
# source /path/to/miniconda3/bin/activate your_env
source /home/environment2/jbhu/miniconda3/bin/activate /home/environment2/jbhu/miniconda3/envs/mmaudio
if ! command -v streamlit >/dev/null 2>&1; then
  echo "未找到 streamlit，请先安装：pip install streamlit"
  exit 1
fi

streamlit run backend/app.py --server.address=0.0.0.0 --server.port=8505