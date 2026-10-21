#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10"'
python -m pip install 'pip==24.0' 'setuptools==65.5.0' 'wheel==0.42.0'
python -m pip install 'torch==2.1.2+cu121' 'torchvision==0.16.2+cu121' --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-repro.txt -c requirements-repro-lock.txt
CSCWD_REQUIREMENTS=requirements-repro.txt python -m pip install --no-deps --no-build-isolation -e .
bash project_tools/build_extensions.sh
python -m pip check
python project_tools/check_environment.py --output outputs/environment_check.json
