$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Vite = Join-Path $RepoRoot "presentation\web\node_modules\vite\bin\vite.js"
$Node = (Get-Command node.exe -ErrorAction Stop).Source
$StatePath = Join-Path $RepoRoot "artifacts\presentation\services.json"
$LogRoot = Join-Path $RepoRoot "artifacts\presentation\logs"
if (-not (Test-Path -LiteralPath $Python)) { throw "Project .venv is missing." }
if (-not (Test-Path -LiteralPath $Vite)) { throw "Frontend is not installed. Run npm ci in presentation\web." }
if (Test-Path -LiteralPath $StatePath) {
    $Old = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    foreach ($Entry in @($Old.backend, $Old.frontend)) {
        if ($Entry -and (Get-Process -Id $Entry -ErrorAction SilentlyContinue)) {
            throw "Presentation service PID $Entry is already running. Stop it first with scripts\stop_presentation.ps1."
        }
    }
}
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$BackendOut = Join-Path $LogRoot "backend.stdout.log"
$BackendErr = Join-Path $LogRoot "backend.stderr.log"
$FrontendOut = Join-Path $LogRoot "frontend.stdout.log"
$FrontendErr = Join-Path $LogRoot "frontend.stderr.log"
$PythonPath = "$RepoRoot\src;$RepoRoot"
$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $PythonPath
$Backend = Start-Process -FilePath $Python -ArgumentList @("-m", "uvicorn", "presentation.backend.main:app", "--app-dir", $RepoRoot, "--host", "127.0.0.1", "--port", "8000") -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $BackendOut -RedirectStandardError $BackendErr
$env:PYTHONPATH = $PreviousPythonPath
$Frontend = Start-Process -FilePath $Node -ArgumentList @($Vite, "--host", "127.0.0.1", "--port", "5173") -WorkingDirectory (Join-Path $RepoRoot "presentation\web") -WindowStyle Hidden -PassThru -RedirectStandardOutput $FrontendOut -RedirectStandardError $FrontendErr
$State = @{ backend = $Backend.Id; frontend = $Frontend.Id; project_root = $RepoRoot }
$State | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding utf8
Start-Sleep -Seconds 2
Write-Host "Backend:  http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:5173"
Write-Host "Logs:     $LogRoot"
