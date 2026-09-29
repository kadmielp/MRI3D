param([switch]$SkipTests)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$mriPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $mriPython)) {
    throw 'Create a Python 3.11 virtual environment and install requirements-dev.txt first. See README.md.'
}
if (-not $SkipTests) {
    & $mriPython -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
}
# PyInstaller logs to stderr; Windows PowerShell 5.1 would treat that as a terminating error.
$ErrorActionPreference = 'Continue'
& $mriPython -m PyInstaller --noconfirm MRI3D.spec
if ($LASTEXITCODE -ne 0) { throw 'Packaging failed.' }
Write-Host "Ready: $PSScriptRoot\dist\MRI3D\MRI3D.exe"
