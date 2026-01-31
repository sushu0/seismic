@echo off
REM ============================================================
REM  30Hz 训练一键启动 - 自动检查环境和断点
REM ============================================================
chcp 65001 >nul
cd /d %~dp0

echo.
echo ============================================================
echo   30Hz 薄层模型训练 - 一键启动
echo ============================================================
echo.

REM 步骤 1: 检测虚拟环境
echo [1/4] 检测虚拟环境...
if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
    echo       ✓ 使用虚拟环境: .venv
) else if exist "venv\Scripts\python.exe" (
    set PYTHON_EXE=venv\Scripts\python.exe
    echo       ✓ 使用虚拟环境: venv
) else (
    echo       × 未找到虚拟环境 (.venv 或 venv^)
    echo.
    echo [建议] 创建虚拟环境:
    echo        python -m venv .venv
    echo        .venv\Scripts\activate
    echo        pip install -r requirements.txt
    pause
    exit /b 1
)
echo.

REM 步骤 2: 验证 PyTorch
echo [2/4] 验证 PyTorch 安装...
%PYTHON_EXE% -c "import torch; print(f'       ✓ PyTorch {torch.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo       × PyTorch 未安装
    echo.
    echo [建议] 安装依赖:
    echo        %PYTHON_EXE% -m pip install torch numpy scipy segyio matplotlib
    pause
    exit /b 1
)
echo.

REM 步骤 3: 检查断点
echo [3/4] 检查训练断点...
if exist "results\01_30Hz_thinlayer_v2\checkpoints\last.pt" (
    echo       ✓ 检测到断点: last.pt
    %PYTHON_EXE% test_resume.py >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo       ✓ 断点验证通过，将从断点继续训练
    ) else (
        echo       ! 断点验证失败，将尝试从头开始
    )
) else (
    echo       - 未找到断点，将从头开始训练
)
echo.

REM 步骤 4: 启动训练
echo [4/4] 启动训练...
echo ------------------------------------------------------------
echo   训练日志: results\01_30Hz_thinlayer_v2\train_log.txt
echo   Checkpoint: results\01_30Hz_thinlayer_v2\checkpoints\
echo   目标: 500 epochs (val_pcc≥0.93, val_r2≥0.86)
echo ------------------------------------------------------------
echo.
echo [提示] 可在另一终端运行以下命令监控训练:
echo        .\monitor_train_30Hz.ps1
echo.

timeout /t 3 /nobreak >nul
echo 训练开始...
echo ============================================================
echo.

%PYTHON_EXE% train_30Hz_thinlayer_v2.py

echo.
echo ============================================================
if %ERRORLEVEL% EQU 0 (
    echo [完成] 训练正常结束
    echo.
    echo [下一步] 运行可视化:
    echo          %PYTHON_EXE% visualize_complete.py
) else (
    echo [错误] 训练异常退出 (错误码: %ERRORLEVEL%^)
    echo.
    echo [建议]
    echo   1. 查看日志: type results\01_30Hz_thinlayer_v2\train_log.txt
    echo   2. 检查显存/内存是否充足
    echo   3. 尝试减小 batch_size (修改 train_30Hz_thinlayer_v2.py)
)
echo ============================================================
echo.

pause
