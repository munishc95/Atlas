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
    [int]$OperateShadowOnly = 1,
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

function Invoke-AtlasPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$FailureLabel
    )

    Write-Host ("python " + ($Arguments -join " "))
    $Output = & python @Arguments 2>&1
    $ExitCode = $LASTEXITCODE
    foreach ($Line in $Output) {
        Write-Host $Line
    }
    if ($ExitCode -ne 0) {
        throw "$FailureLabel exited with code $ExitCode"
    }
}

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
        Write-Host "Operate shadow-only: $OperateShadowOnly"
    }
    Write-Host "Log: $LogPath"
    Invoke-AtlasPython -Arguments $ImportArgs -FailureLabel "Importer"
    Invoke-AtlasPython -FailureLabel "Corporate action importer" -Arguments @(
            "scripts/free_nse_corporate_actions_import.py",
            "--bundle-id", "$BundleId",
            "--start-date", $CorporateActionStartDate,
            "--end-date", $EndDate,
            "--mode", "UPSERT"
        )
    if (-not $SkipEventRisk) {
        Invoke-AtlasPython -FailureLabel "Event-risk sync" -Arguments @(
                "scripts/free_event_risk_sync.py",
                "--bundle-id", "$BundleId",
                "--start-date", $EventRiskStartDate,
                "--end-date", $EventRiskEndDate
            )
    }
    if ($RunOperate) {
        $OperateArgs = @(
                "scripts/run_operate_inline.py",
                "--bundle-id", "$BundleId",
                "--timeframe", "$OperateTimeframe",
                "--regime", "$OperateRegime",
                "--date", $EndDate,
                "--source", "windows_daily_free_data_task",
                "--max-runtime-seconds", "$OperateMaxRuntimeSeconds",
                "--skip-if-auto-run-date-marked",
                "--mark-auto-run-date"
            )
        if ($OperateShadowOnly) {
            $OperateArgs += "--shadow-only"
        }
        Invoke-AtlasPython -FailureLabel "Operate inline run" -Arguments $OperateArgs
    }
    Write-Host "Atlas free NSE daily update finished"
}
finally {
    Stop-Transcript | Out-Null
}
