param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($Clean) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, dist
}

python -m pip install -r requirements.txt
python -m pip install pyinstaller
python -m PyInstaller --noconfirm NINA.spec

Write-Host ""
Write-Host "Build complete:"
Write-Host "  $Root\dist\NINA\NINA.exe"
