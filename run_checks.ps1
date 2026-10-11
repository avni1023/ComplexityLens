# PowerShell test and lint runner for ComplexityLens on Windows

$ErrorActionPreference = "Stop"

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$VenvPytest = Join-Path $PSScriptRoot ".venv\Scripts\pytest.exe"
$VenvRuff = Join-Path $PSScriptRoot ".venv\Scripts\ruff.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found at .venv. Run setup first."
    exit 1
}

Write-Host "==> Running ruff lint check..." -ForegroundColor Cyan
& $VenvRuff check .
if ($LASTEXITCODE -ne 0) {
    Write-Error "Ruff check failed"
    exit $LASTEXITCODE
}

Write-Host "==> Running pytest suite..." -ForegroundColor Cyan
& $VenvPytest
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pytest suite failed"
    exit $LASTEXITCODE
}

Write-Host "==> All checks passed successfully!" -ForegroundColor Green
