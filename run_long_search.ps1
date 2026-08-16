$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path '.venv')) {
    py -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
New-Item -ItemType Directory -Force -Path results, checkpoints | Out-Null
Start-Process python -ArgumentList 'long_search.py --mode extended --max-hours 10' -RedirectStandardOutput 'results\live.log' -RedirectStandardError 'results\live_error.log' -NoNewWindow
Write-Output 'Long search started. Monitor results\live.log and results\leaderboard.csv.'
