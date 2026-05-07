param(
    [string]$TaskName = "Atlas Free NSE Daily Update",
    [string]$StartTime = "18:45",
    [int]$BundleId = 3674,
    [int]$LookbackDays = 10,
    [int]$CorporateActionLookbackDays = 180,
    [int]$EventRiskLookbackDays = 30,
    [int]$EventRiskForwardDays = 120,
    [switch]$SkipEventRisk,
    [switch]$RunOperate,
    [string]$OperateTimeframe = "1d",
    [string]$OperateRegime = "TREND_UP",
    [int]$OperateMaxRuntimeSeconds = 10800,
    [int]$ExecutionHours = 4
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$UpdateScript = Join-Path $RepoRoot "scripts\daily-free-data-update.ps1"
if (-not (Test-Path $UpdateScript)) {
    throw "Missing update script: $UpdateScript"
}

$At = [datetime]::Today.Add([TimeSpan]::Parse($StartTime))
$UserId = if ($env:USERDOMAIN) { "$env:USERDOMAIN\$env:USERNAME" } else { $env:USERNAME }

$ActionArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$UpdateScript`"",
    "-BundleId", "$BundleId",
    "-LookbackDays", "$LookbackDays",
    "-CorporateActionLookbackDays", "$CorporateActionLookbackDays",
    "-EventRiskLookbackDays", "$EventRiskLookbackDays",
    "-EventRiskForwardDays", "$EventRiskForwardDays"
)
if ($SkipEventRisk) {
    $ActionArgs += "-SkipEventRisk"
}
if ($RunOperate) {
    $ActionArgs += @(
        "-RunOperate",
        "-OperateTimeframe", "$OperateTimeframe",
        "-OperateRegime", "$OperateRegime",
        "-OperateMaxRuntimeSeconds", "$OperateMaxRuntimeSeconds"
    )
}
$ActionArgs = $ActionArgs -join " "

$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $ActionArgs -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At $At
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -WakeToRun `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours $ExecutionHours)
$Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType Interactive -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Pulls free NSE bhavcopy, corporate actions, event-risk data, and optionally runs Atlas operate." `
    -Force | Out-Null

$Task = Get-ScheduledTask -TaskName $TaskName
$Info = Get-ScheduledTaskInfo -TaskName $TaskName

[pscustomobject]@{
    TaskName = $Task.TaskName
    State = $Task.State
    StartTime = $At.ToString("HH:mm")
    RunOperate = [bool]$RunOperate
    User = $UserId
    Script = $UpdateScript
    NextRunTime = $Info.NextRunTime
    LastRunTime = $Info.LastRunTime
    LastTaskResult = $Info.LastTaskResult
}
