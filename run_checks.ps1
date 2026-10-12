# PowerShell test and lint runner for ComplexityLens on Windows

$ErrorActionPreference = "Stop"

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$VenvPytest = Join-Path $PSScriptRoot ".venv\Scripts\pytest.exe"
$VenvRuff = Join-Path $PSScriptRoot ".venv\Scripts\ruff.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found at .venv. Run setup first."
    exit 1
}

if (-not $env:CXX_INCLUDE_DIRS) {
    $userVal = [System.Environment]::GetEnvironmentVariable('CXX_INCLUDE_DIRS', 'User')
    if ($userVal) {
        $env:CXX_INCLUDE_DIRS = $userVal
    } elseif (Test-Path 'C:\msys64\ucrt64\include\c++\16.2.0') {
        $env:CXX_INCLUDE_DIRS = @(
            'C:\msys64\ucrt64\include\c++\16.2.0',
            'C:\msys64\ucrt64\include\c++\16.2.0\x86_64-w64-mingw32',
            'C:\msys64\ucrt64\include',
            'C:\msys64\ucrt64\lib\gcc\x86_64-w64-mingw32\16.2.0\include',
            'C:\msys64\ucrt64\lib\gcc\x86_64-w64-mingw32\16.2.0\include-fixed'
        ) -join ';'
    }
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
