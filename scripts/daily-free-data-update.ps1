param(
    [int]$BundleId = 3674,
    [int]$LookbackDays = 10,
    [int]$CorporateActionLookbackDays = 180,
    [int]$EventRiskLookbackDays = 30,
    [int]$EventRiskForwardDays = 120,
    [double]$ThrottleSeconds = 0.05,
    [switch]$RunQuality,
    [switch]$SkipEventRisk,
    [switch]$RunOperate,
    [string]$OperateTimeframe = "1d",
    [string]$OperateRegime = "TREND_UP",
    [int]$OperateMaxRuntimeSeconds = 10800
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogsRoot = Join-Path $RepoRoot "data\logs"
New-Item -ItemType Directory -Force -Path $LogsRoot | Out-Null

$RunDate = Get-Date
$StartDate = $RunDate.Date.AddDays(-1 * [Math]::Max(1, $LookbackDays)).ToString("yyyy-MM-dd")
$CorporateActionStartDate = $RunDate.Date.AddDays(-1 * [Math]::Max(1, $CorporateActionLookbackDays)).ToString("yyyy-MM-dd")
$EventRiskStartDate = $RunDate.Date.AddDays(-1 * [Math]::Max(1, $EventRiskLookbackDays)).ToString("yyyy-MM-dd")
$EventRiskEndDate = $RunDate.Date.AddDays([Math]::Max(1, $EventRiskForwardDays)).ToString("yyyy-MM-dd")
$EndDate = $RunDate.Date.ToString("yyyy-MM-dd")
$LogPath = Join-Path $LogsRoot ("daily-free-data-update-{0}.log" -f $RunDate.ToString("yyyyMMdd-HHmmss"))

Set-Location $RepoRoot
$env:PYTHONPATH = Join-Path $RepoRoot "apps\api"

$ImportArgs = @(
    "scripts/free_nse_bhavcopy_backfill.py",
    "--bundle-id", "$BundleId",
    "--start-date", $StartDate,
    "--end-date", $EndDate,
    "--throttle-seconds", "$ThrottleSeconds"
)

if (-not $RunQuality) {
    $ImportArgs += "--skip-quality"
}

Start-Transcript -Path $LogPath -Append | Out-Null
try {
    Write-Host "Atlas free NSE daily update"
    Write-Host "Repo: $RepoRoot"
    Write-Host "Bundle: $BundleId"
    Write-Host "Window: $StartDate to $EndDate"
    Write-Host "Corporate action window: $CorporateActionStartDate to $EndDate"
    if (-not $SkipEventRisk) {
        Write-Host "Event risk window: $EventRiskStartDate to $EventRiskEndDate"
    }
    if ($RunOperate) {
        Write-Host "Operate run: enabled ($OperateTimeframe, $OperateRegime)"
    }
    Write-Host "Log: $LogPath"
    & python @ImportArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Importer exited with code $LASTEXITCODE"
    }
    & python @(
        "scripts/free_nse_corporate_actions_import.py",
        "--bundle-id", "$BundleId",
        "--start-date", $CorporateActionStartDate,
        "--end-date", $EndDate,
        "--mode", "UPSERT"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Corporate action importer exited with code $LASTEXITCODE"
    }
    if (-not $SkipEventRisk) {
        & python @(
            "scripts/free_event_risk_sync.py",
            "--bundle-id", "$BundleId",
            "--start-date", $EventRiskStartDate,
            "--end-date", $EventRiskEndDate
        )
        if ($LASTEXITCODE -ne 0) {
            throw "Event-risk sync exited with code $LASTEXITCODE"
        }
    }
    if ($RunOperate) {
        & python @(
            "scripts/run_operate_inline.py",
            "--bundle-id", "$BundleId",
            "--timeframe", "$OperateTimeframe",
            "--regime", "$OperateRegime",
            "--date", $EndDate,
            "--source", "windows_daily_free_data_task",
            "--max-runtime-seconds", "$OperateMaxRuntimeSeconds",
            "--mark-auto-run-date"
        )
        if ($LASTEXITCODE -ne 0) {
            throw "Operate inline run exited with code $LASTEXITCODE"
        }
    }
    Write-Host "Atlas free NSE daily update finished"
}
finally {
    Stop-Transcript | Out-Null
}
