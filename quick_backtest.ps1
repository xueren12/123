# 一键回测脚本 - 支持多种数据源
# 使用方法：
#   .\quick_backtest.ps1 -Mode offline                    # 使用现有CSV文件
#   .\quick_backtest.ps1 -Mode download -Exchange okx     # 下载OKX数据并回测
#   .\quick_backtest.ps1 -Mode download -Exchange binanceusdm -Proxy "http://127.0.0.1:7890"  # 使用代理下载Binance数据

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("offline", "download")]
    [string]$Mode,
    
    [string]$Exchange = "okx",
    [string]$Inst = "ETH-USDT-SWAP",
    [string]$Timeframe = "5m",
    [string]$Start = "2025-01-01",
    [string]$End = "2025-01-02",
    [string]$Proxy = "",
    [string]$DataDir = "data"
)

# 设置错误处理
$ErrorActionPreference = "Stop"

Write-Host "===== 一键回测脚本 =====" -ForegroundColor Green
Write-Host "模式: $Mode" -ForegroundColor Yellow
Write-Host "交易所: $Exchange" -ForegroundColor Yellow
Write-Host "合约: $Inst" -ForegroundColor Yellow
Write-Host "时间框架: $Timeframe" -ForegroundColor Yellow
Write-Host "时间范围: $Start 到 $End" -ForegroundColor Yellow

if ($Mode -eq "offline") {
    # 离线模式：使用现有CSV文件
    Write-Host "`n===== 离线模式：查找现有CSV文件 =====" -ForegroundColor Cyan
    
    $csvFiles = Get-ChildItem -Path $DataDir -Filter "*ohlcv*.csv" | Sort-Object LastWriteTime -Descending
    
    if ($csvFiles.Count -eq 0) {
        Write-Host "错误：未找到任何OHLCV CSV文件！" -ForegroundColor Red
        Write-Host "请先下载数据或使用 -Mode download" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "找到 $($csvFiles.Count) 个CSV文件：" -ForegroundColor Green
    for ($i = 0; $i -lt $csvFiles.Count; $i++) {
        $size = [math]::Round($csvFiles[$i].Length / 1KB, 2)
        Write-Host "  [$i] $($csvFiles[$i].Name) ($size KB)" -ForegroundColor White
    }
    
    # 使用最新的CSV文件
    $csvFile = $csvFiles[0].FullName
    Write-Host "`n使用最新文件: $($csvFiles[0].Name)" -ForegroundColor Green
    
} elseif ($Mode -eq "download") {
    # 下载模式：先下载数据再回测
    Write-Host "`n===== 下载模式：获取最新数据 =====" -ForegroundColor Cyan
    
    # 设置代理环境变量
    if ($Proxy -ne "") {
        Write-Host "设置代理: $Proxy" -ForegroundColor Yellow
        $env:HTTP_PROXY = $Proxy
        $env:HTTPS_PROXY = $Proxy
    }
    
    # 构建CSV文件名
    $csvFileName = "ohlcv_${Inst}_${Timeframe}_${Start}_${End}.csv"
    $csvFile = Join-Path $DataDir $csvFileName
    
    # 构建下载命令
    $downloadCmd = @(
        "python", "-u", "mvp/scripts/download_ohlcv_okx.py",
        "--engine", "ccxt",
        "--ccxt-exchange", $Exchange,
        "--inst", $Inst,
        "--timeframe", $Timeframe,
        "--start", $Start,
        "--end", $End,
        "--out", $csvFile,
        "--timeout", "30000",
        "--max-retries", "5"
    )
    
    if ($Proxy -ne "") {
        $downloadCmd += @("--proxy", $Proxy)
    }
    
    Write-Host "执行下载命令..." -ForegroundColor Yellow
    Write-Host ($downloadCmd -join " ") -ForegroundColor Gray
    
    & $downloadCmd[0] $downloadCmd[1..($downloadCmd.Length-1)]
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误：数据下载失败！" -ForegroundColor Red
        exit 1
    }
    
    if (!(Test-Path $csvFile)) {
        Write-Host "错误：CSV文件未生成！" -ForegroundColor Red
        exit 1
    }
    
    $size = [math]::Round((Get-Item $csvFile).Length / 1KB, 2)
    Write-Host "下载完成: $csvFileName ($size KB)" -ForegroundColor Green
}

# 执行回测
Write-Host "`n===== 执行回测 =====" -ForegroundColor Cyan

$backtest_inst = $Inst -replace "-SWAP", ""  # 移除-SWAP后缀用于回测
$backtest_timeframe = $Timeframe -replace "m", "min"  # 转换时间框架格式

$backtestCmd = @(
    "python", "-u", "mvp/backtest/ma_backtest.py",
    "--source", "csv",
    "--csv", $csvFile,
    "--inst", $backtest_inst,
    "--start", $Start,
    "--end", $End,
    "--timeframe", $backtest_timeframe,
    "--plot", "1"
)

Write-Host "执行回测命令..." -ForegroundColor Yellow
Write-Host ($backtestCmd -join " ") -ForegroundColor Gray

& $backtestCmd[0] $backtestCmd[1..($backtestCmd.Length-1)]

if ($LASTEXITCODE -ne 0) {
    Write-Host "警告：回测执行可能有问题，但继续检查产物..." -ForegroundColor Yellow
}

# 检查回测产物
Write-Host "`n===== 回测产物 =====" -ForegroundColor Cyan

$products = @(
    "data/backtest_ma_breakout.csv",
    "data/backtest_ma_breakout.svg", 
    "data/backtest_summary.json"
)

$foundProducts = @()
foreach ($product in $products) {
    if (Test-Path $product) {
        $foundProducts += $product
        Write-Host "[✓] $product" -ForegroundColor Green
    } else {
        Write-Host "[✗] $product" -ForegroundColor Red
    }
}

if ($foundProducts.Count -gt 0) {
    Write-Host "`n===== 回测结果摘要 =====" -ForegroundColor Cyan
    
    if (Test-Path "data/backtest_summary.json") {
        $summary = Get-Content "data/backtest_summary.json" | ConvertFrom-Json
        Write-Host "夏普比率: $($summary.metrics.sharpe)" -ForegroundColor White
        Write-Host "胜率: $($summary.metrics.winrate * 100)%" -ForegroundColor White
        Write-Host "最大回撤: $($summary.metrics.max_drawdown * 100)%" -ForegroundColor White
        Write-Host "数据条数: $($summary.context.bars)" -ForegroundColor White
    }
    
    Write-Host "`n回测完成！产物已保存到 data/ 目录" -ForegroundColor Green
} else {
    Write-Host "`n回测失败：未生成任何产物" -ForegroundColor Red
    exit 1
}