$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StatePath = Join-Path $RepoRoot "artifacts\presentation\services.json"
if (-not (Test-Path -LiteralPath $StatePath)) {
    Write-Host "No presentation service PID record is present."
    exit 0
}
$State = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
try {
    $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 2
    if ($Health.busy) {
        throw "A GPU task is active ($($Health.active_task)). Wait for it to finish, then stop the app."
    }
} catch {
    if ($_.Exception.Message -like "A GPU task is active*") { throw }
}
foreach ($Entry in @(@{ Id = $State.backend; Match = "uvicorn" }, @{ Id = $State.frontend; Match = "vite" })) {
    if (-not $Entry.Id) { continue }
    $ProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($Entry.Id)" -ErrorAction SilentlyContinue
    $IsProjectBackend = $Entry.Match -eq "uvicorn" -and $ProcessInfo -and
        $ProcessInfo.ExecutablePath -ieq (Join-Path $RepoRoot ".venv\Scripts\python.exe") -and
        $ProcessInfo.CommandLine -like "*presentation.backend.main:app*"
    $IsProjectFrontend = $Entry.Match -eq "vite" -and $ProcessInfo -and
        $ProcessInfo.CommandLine -like "*$RepoRoot*" -and $ProcessInfo.CommandLine -like "*vite*"
    if ($IsProjectBackend -or $IsProjectFrontend) {
        Stop-Process -Id $Entry.Id -Force
        Write-Host "Stopped $($Entry.Match) process $($Entry.Id)."
    }
}
Remove-Item -LiteralPath $StatePath -Force
