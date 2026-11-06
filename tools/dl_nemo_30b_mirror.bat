@echo off
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HUB_DISABLE_XET=1"
python D:\AI\tools\dl_nemo_30b.py > D:\AI\tools\dl_nemo_30b_mirror.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\AI\tools\dl_nemo_30b_mirror.log
