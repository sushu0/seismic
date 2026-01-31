@echo off
REM ===================================================
REM  30Hz ThinLayerNet V2 断点续训脚本
REM  自动从 last.pt 恢复训练
REM ===================================================
chcp 65001 >nul
cd /d %~dp0

echo ========================================
echo   30Hz 薄层模型断点续训
echo ========================================
echo.

REM 检测虚拟环境
if exist ".venv\Scripts\python.exe" (
    echo [OK] 使用虚拟环境: .venv
    set PYTHON_EXE=.venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    echo [OK] 使用虚拟环境: venv
    set PYTHON_EXE=venv\Scripts\python.exe
) else (
    echo [警告] 未找到虚拟环境，使用系统 Python
    set PYTHON_EXE=python
)

REM 检测断点文件
if exist "results\01_30Hz_thinlayer_v2\checkpoints\last.pt" (
    echo [OK] 检测到断点: last.pt
    echo.
) else (
    echo [提示] 未找到断点文件，将从头开始训练
    echo.
)

echo 启动训练...
echo ----------------------------------------
%PYTHON_EXE% train_30Hz_thinlayer_v2.py
echo ----------------------------------------
echo.

if %ERRORLEVEL% EQU 0 (
    echo [完成] 训练正常结束
) else (
    echo [错误] 训练异常退出，错误码: %ERRORLEVEL%
)

pause
