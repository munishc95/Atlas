param(
  [switch]$SkipDocker,
  [switch]$InstallE2E
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

function Start-AtlasWindow {
  param(
    [string]$Title,
    [string]$Command
  )

  Start-Process powershell -WorkingDirectory $root -ArgumentList @(
    "-NoExit",
    "-Command",
    "`$Host.UI.RawUI.WindowTitle = '$Title'; $Command"
  ) | Out-Null
}

function Get-PnpmCommand {
  $shimPath = Join-Path $env:APPDATA "npm\pnpm.cmd"
  if (Test-Path $shimPath) {
    return $shimPath
  }
  if (Get-Command pnpm -ErrorAction SilentlyContinue) {
    return "pnpm"
  }
  return $null
}

function Test-PlaywrightChromiumInstalled {
  $browserPath = if ($env:PLAYWRIGHT_BROWSERS_PATH) {
    $env:PLAYWRIGHT_BROWSERS_PATH
  } else {
    Join-Path $env:LOCALAPPDATA "ms-playwright"
  }

  if (-not (Test-Path $browserPath)) {
    return $false
  }

  $chromiumDirs = Get-ChildItem -Path $browserPath -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "chromium-*" }
  return ($chromiumDirs.Count -gt 0)
}

if (-not $SkipDocker) {
  Write-Host "Starting Docker services (Postgres + Redis)..." -ForegroundColor Cyan
  docker compose -f infra/docker-compose.yml up -d
}

$pnpmCmd = Get-PnpmCommand
if (-not $pnpmCmd) {
  Write-Warning "pnpm not found in PATH or APPDATA shim. Web server and E2E setup may fail."
}

if (-not (Test-PlaywrightChromiumInstalled)) {
  Write-Host "Playwright Chromium browser not found." -ForegroundColor Yellow
  if ($InstallE2E -and $pnpmCmd) {
    Write-Host "Installing Playwright Chromium (apps/web)..." -ForegroundColor Cyan
    & $pnpmCmd -C apps/web test:e2e:setup
  } else {
    Write-Host "Run: pnpm -C apps/web test:e2e:setup" -ForegroundColor Yellow
    Write-Host "Or launch stack with: pnpm dev:stack -- -InstallE2E" -ForegroundColor Yellow
  }
}

$apiCommand = '$env:PYTHONPATH="apps/api"; python -m uvicorn app.main:app --reload --app-dir apps/api --host 127.0.0.1 --port 8000'
$workerCommand = '$env:PYTHONPATH="apps/api"; python -m app.worker'
$webCommand = '$pnpmPath = Join-Path $env:APPDATA "npm\pnpm.cmd"; if (Test-Path $pnpmPath) { & $pnpmPath -C apps/web dev } elseif (Get-Command pnpm -ErrorAction SilentlyContinue) { pnpm -C apps/web dev } else { Write-Error "pnpm not found in PATH or APPDATA shim" }'

Write-Host "Opening API, Worker, and Web terminals..." -ForegroundColor Cyan
Start-AtlasWindow -Title "Atlas API" -Command $apiCommand
Start-AtlasWindow -Title "Atlas Worker" -Command $workerCommand
Start-AtlasWindow -Title "Atlas Web" -Command $webCommand

Write-Host "Atlas stack launch initiated." -ForegroundColor Green
Write-Host "Web: http://localhost:3000"
Write-Host "API: http://127.0.0.1:8000/api/health"
