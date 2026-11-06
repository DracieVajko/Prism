@echo off
cd /d D:\AI\DiffusionServer\pc
python -m uvicorn unified_server:app --host 0.0.0.0 --port 8102 > D:\AI\tools\pc_server.log 2>&1
