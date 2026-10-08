#!/usr/bin/env bash
set -euo pipefail
task=/inspire/ssd/project/video-generation/public/huangyuwei/experiments/omni-bitwise-20261008
cd "$task"
exec bash "$task/run_production_models.sh"
