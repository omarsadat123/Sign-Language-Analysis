$ErrorActionPreference = "Stop"

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$CommandArguments
    )

    & $Executable @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Executable $($CommandArguments -join ' ')"
    }
}

$projectRoot = $PSScriptRoot
$venvRoot = Join-Path $env:LOCALAPPDATA "SignLearn\venv-py311"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating a short-path Python 3.11 environment at $venvRoot ..."
    New-Item -ItemType Directory -Force -Path (Split-Path $venvRoot -Parent) | Out-Null
    Invoke-CheckedCommand "py" @("-3.11", "-m", "venv", $venvRoot)
}

Invoke-CheckedCommand $venvPython @("-m", "pip", "install", "--upgrade", "pip")
Invoke-CheckedCommand $venvPython @("-m", "pip", "install", "-r", (Join-Path $projectRoot "requirements.txt"))
Invoke-CheckedCommand $venvPython @((Join-Path $projectRoot "app.py"))
