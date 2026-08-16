#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir -p results checkpoints
nohup python long_search.py --mode extended --max-hours 10 > results/live.log 2>&1 &
echo "Long search started with PID $!"
echo "Monitor it with: tail -f results/live.log"
