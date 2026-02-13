param(
  [switch]$Install,
  [switch]$WithDeps
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

function Get-PnpmCommand {
  $shimPath = Join-Path $env:APPDATA "npm\pnpm.cmd"
  if (Test-Path $shimPath) {
    return $shimPath
  }
  if (Get-Command pnpm -ErrorAction SilentlyContinue) {
    return "pnpm"
  }
  throw "pnpm not found. Install Node.js + pnpm first."
}

function Get-PlaywrightBrowserPath {
  if ($env:PLAYWRIGHT_BROWSERS_PATH) {
    return $env:PLAYWRIGHT_BROWSERS_PATH
  }
  return Join-Path $env:LOCALAPPDATA "ms-playwright"
}

function Test-ChromiumInstalled {
  $browserPath = Get-PlaywrightBrowserPath
  if (-not (Test-Path $browserPath)) {
    return $false
  }
  $dirs = Get-ChildItem -Path $browserPath -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "chromium-*" }
  return ($dirs.Count -gt 0)
}

$pnpmCmd = Get-PnpmCommand
$installed = Test-ChromiumInstalled

if ($installed) {
  Write-Host "Playwright Chromium already installed." -ForegroundColor Green
  exit 0
}

Write-Host "Playwright Chromium is missing." -ForegroundColor Yellow
Write-Host "Command: pnpm -C apps/web test:e2e:setup" -ForegroundColor Yellow
Write-Host "CI/Linux alternative: pnpm -C apps/web exec playwright install --with-deps chromium" -ForegroundColor Yellow

if (-not $Install) {
  Write-Host "Pass -Install to run setup now." -ForegroundColor Yellow
  exit 0
}

if ($WithDeps) {
  & $pnpmCmd -C apps/web exec playwright install --with-deps chromium
} else {
  & $pnpmCmd -C apps/web test:e2e:setup
}

if (Test-ChromiumInstalled) {
  Write-Host "Playwright Chromium setup complete." -ForegroundColor Green
} else {
  throw "Playwright Chromium setup did not complete successfully."
}
