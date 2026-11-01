#!/usr/bin/env bash
# 服务器一键环境：venv + torch + 训练依赖 + 基座权重
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== 0. Python (需 >=3.10) =="
python3 -c "import sys; assert sys.version_info >= (3,10), sys.version; print('python', sys.version.split()[0])"

echo "== 1. 检测系统自带 CUDA torch（AutoDL 等镜像通常已装） =="
SYS_TORCH=0
if python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then SYS_TORCH=1; fi
echo "system torch with cuda: $SYS_TORCH"

echo "== 2. 建 venv =="
rm -rf "$ROOT/.venv"
if [ "$SYS_TORCH" = "1" ]; then
  python3 -m venv --system-site-packages "$ROOT/.venv"
else
  python3 -m venv "$ROOT/.venv"
fi
VP="$ROOT/.venv/bin/python"
$VP -m pip install -U pip -q -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "== 3. torch =="
if [ "$SYS_TORCH" = "1" ]; then
  echo "复用系统 torch，跳过下载"
else
  $VP -m pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

echo "== 4. 训练依赖 =="
$VP -m pip install "numpy==1.26.4" "transformers==4.57.6" "datasets==3.6.0" -q \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "== 5. 基座权重 full_sft_768.pth =="
mkdir -p "$ROOT/minimind-code/out"
if [ ! -s "$ROOT/minimind-code/out/full_sft_768.pth" ]; then
  curl -L --fail --retry 3 -o "$ROOT/minimind-code/out/full_sft_768.pth" \
    "https://modelscope.cn/api/v1/models/gongjy/minimind-3-pytorch/repo?Revision=master&FilePath=full_sft_768.pth"
fi
SIZE=$(stat -c%s "$ROOT/minimind-code/out/full_sft_768.pth")
echo "full_sft_768.pth = $SIZE bytes (期望 137683346)"
[ "$SIZE" = "137683346" ] || { echo "!! 权重不完整，删掉重跑"; exit 1; }

echo "== 6. 自检 =="
$VP -c "import datasets, transformers, torch; print('datasets', datasets.__version__); print('transformers', transformers.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "== setup done =="