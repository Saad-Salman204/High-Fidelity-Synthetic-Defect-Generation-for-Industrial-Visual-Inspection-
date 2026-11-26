$ErrorActionPreference = "Stop"
$WebRoot = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")).Path "presentation\web"
if (-not (Test-Path -LiteralPath (Join-Path $WebRoot "node_modules\vite\bin\vite.js"))) {
    throw "Frontend dependencies are missing. Run: cd presentation/web; npm ci"
}
Set-Location -LiteralPath $WebRoot
& npm.cmd run dev
