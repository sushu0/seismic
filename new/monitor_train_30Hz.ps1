# ===================================================
#  30Hz 训练监控脚本 - 实时显示训练日志和最佳指标
# ===================================================
param(
    [int]$RefreshSec = 5,  # 刷新间隔(秒)
    [int]$TailLines = 20   # 显示最后N行日志
)

$LogPath = "results\01_30Hz_thinlayer_v2\train_log.txt"
$BestCkpt = "results\01_30Hz_thinlayer_v2\checkpoints\best.pt"

if (!(Test-Path $LogPath)) {
    Write-Host "[错误] 训练日志不存在: $LogPath" -ForegroundColor Red
    exit 1
}

Clear-Host
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  30Hz 薄层模型训练监控" -ForegroundColor Cyan
Write-Host "  日志: $LogPath" -ForegroundColor Cyan
Write-Host "  刷新间隔: $RefreshSec 秒 (Ctrl+C 停止)" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

while ($true) {
    Clear-Host
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host "  训练监控 - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host ""
    
    # 检测最佳 checkpoint 信息
    if (Test-Path $BestCkpt) {
        $BestInfo = python -c @"
import torch, json
ckpt = torch.load('$BestCkpt', map_location='cpu', weights_only=False)
print(f\"Epoch {ckpt.get('epoch', '?')}: PCC={ckpt.get('val_metrics', {}).get('pcc', -1):.4f}, R2={ckpt.get('val_metrics', {}).get('r2', -1):.4f}\")
"@ 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[最佳模型] $BestInfo" -ForegroundColor Green
        }
    }
    
    # 显示最后N行日志
    Write-Host ""
    Write-Host "[最近 $TailLines 行日志]" -ForegroundColor Yellow
    Write-Host "-" * 70 -ForegroundColor DarkGray
    Get-Content $LogPath -Tail $TailLines -Encoding UTF8 | ForEach-Object {
        if ($_ -match "Epoch\s+(\d+)") {
            Write-Host $_ -ForegroundColor White
        } elseif ($_ -match "val_pcc=") {
            Write-Host $_ -ForegroundColor Cyan
        } else {
            Write-Host $_ -ForegroundColor Gray
        }
    }
    Write-Host "-" * 70 -ForegroundColor DarkGray
    
    # 文件大小和最后修改时间
    $LogFile = Get-Item $LogPath
    Write-Host ""
    Write-Host "[日志状态] 大小: $([math]::Round($LogFile.Length/1KB, 2)) KB | 最后更新: $($LogFile.LastWriteTime.ToString('HH:mm:ss'))" -ForegroundColor DarkCyan
    
    Start-Sleep -Seconds $RefreshSec
}
