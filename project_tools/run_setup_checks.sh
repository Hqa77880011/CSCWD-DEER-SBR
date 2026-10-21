#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m pip check
python project_tools/check_environment.py --output outputs/environment_check.json
bash scripts/test.sh
