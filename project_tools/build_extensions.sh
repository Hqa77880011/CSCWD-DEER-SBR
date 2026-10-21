#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CUDA_HOME:=/usr/local/cuda}"
: "${MAX_JOBS:=8}"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print(";".join(sorted({"%d.%d" % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})))')"
  if [[ -z "$TORCH_CUDA_ARCH_LIST" ]]; then
    echo 'No CUDA GPU detected; set TORCH_CUDA_ARCH_LIST explicitly for a cross-build.' >&2
    exit 1
  fi
fi
export CUDA_HOME TORCH_CUDA_ARCH_LIST MAX_JOBS
python opencood/utils/setup.py build_ext --inplace
python opencood/pcdet_utils/setup.py build_ext --inplace
