# 30Hz 模型训练脚本
# 使用方法: 在PowerShell中运行 .\train_30Hz_background.ps1

$scriptPath = "D:\SEISMIC_CODING\new\train_30Hz_thinlayer_v2.py"
$pythonPath = "D:\SEISMIC_CODING\new\.venv\Scripts\python.exe"
$logPath = "D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_v2\train_log.txt"
$workDir = "D:\SEISMIC_CODING\new"

Write-Host "Starting 30Hz training in background..."
Write-Host "Log file: $logPath"

# Remove old checkpoints to start fresh
Remove-Item "D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_v2\checkpoints\*" -Force -ErrorAction SilentlyContinue

# Start the training process
$proc = Start-Process -FilePath $pythonPath -ArgumentList "-u $scriptPath" -WorkingDirectory $workDir -RedirectStandardOutput $logPath -RedirectStandardError "$logPath.err" -PassThru -WindowStyle Hidden

Write-Host "Process started with PID: $($proc.Id)"
Write-Host "Monitor with: Get-Content $logPath -Wait"
