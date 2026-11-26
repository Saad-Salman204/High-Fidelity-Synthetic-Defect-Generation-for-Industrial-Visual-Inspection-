$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python environment not found: $Python"
}
$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
Set-Location -LiteralPath $RepoRoot
& $Python -m uvicorn presentation.backend.main:app --host 127.0.0.1 --port 8000
