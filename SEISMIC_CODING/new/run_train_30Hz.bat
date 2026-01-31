@echo off
cd /d D:\SEISMIC_CODING\new
call .venv\Scripts\activate.bat
python train_30Hz_thinlayer_v2.py
pause
