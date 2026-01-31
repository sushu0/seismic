@echo off
REM 批量训练所有频率的V6模型
echo ==========================================
echo Training all frequency models
echo ==========================================

set PYTHON=D:\SEISMIC_CODING\new\.venv\Scripts\python.exe
set SCRIPT=D:\SEISMIC_CODING\new\train_v6.py

echo.
echo Training 20Hz...
%PYTHON% %SCRIPT% --freq 20Hz --epochs 800

echo.
echo Training 40Hz...
%PYTHON% %SCRIPT% --freq 40Hz --epochs 800

echo.
echo Training 50Hz...
%PYTHON% %SCRIPT% --freq 50Hz --epochs 800

echo.
echo ==========================================
echo All training completed!
echo ==========================================
pause
