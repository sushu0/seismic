@echo off
title ThinLayerNet V2 Training
cd /d D:\SEISMIC_CODING\new

echo ============================================================
echo   ThinLayerNet V2 Training - Direct Console Output
echo ============================================================
echo.

set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

.venv\Scripts\python.exe -u train_thinlayer_v2.py

echo.
echo ============================================================
echo   Training finished. Press any key to exit.
echo ============================================================
pause >nul
