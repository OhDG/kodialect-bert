[CmdletBinding()]
param(
    [switch]$Setup,
    [switch]$SetupOnly,
    [string]$TokenizersArchive,
    [string]$Corpus,
    [string]$OutputDir,
    [int]$MaxLines = 0,
    [int]$BatchSize = 1024
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$environmentPython = Join-Path $projectRoot '.env\Scripts\python.exe'
$savedEnvironment = @{}
$localEnvironment = @{
    PIP_CACHE_DIR = (Join-Path $projectRoot '.cache\pip')
    PIP_REQUIRE_VIRTUALENV = 'true'
    PYTHONNOUSERSITE = '1'
    PYTHONUTF8 = '1'
    HF_HOME = (Join-Path $projectRoot '.cache\huggingface')
    HF_MODULES_CACHE = (Join-Path $projectRoot '.cache\huggingface\modules')
    HF_HUB_OFFLINE = '1'
    TRANSFORMERS_OFFLINE = '1'
    CUDA_VISIBLE_DEVICES = ''
}

function Invoke-EnvironmentPython {
    param([string[]]$Arguments)
    & $environmentPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Project Python exited with code $LASTEXITCODE"
    }
}

Push-Location -LiteralPath $projectRoot
try {
    foreach ($entry in $localEnvironment.GetEnumerator()) {
        $savedEnvironment[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, 'Process')
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
    if ($Setup -or $SetupOnly) {
        if (-not (Test-Path -LiteralPath $environmentPython)) {
            & py -3.10 -m venv (Join-Path $projectRoot '.env')
            if ($LASTEXITCODE -ne 0) { throw 'Could not create the project Python 3.10 virtual environment.' }
        }
        Invoke-EnvironmentPython -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $projectRoot 'requirements.txt'))
    }
    if (-not (Test-Path -LiteralPath $environmentPython)) {
        throw 'Project environment is missing. Run .\run_local.ps1 -SetupOnly first.'
    }
    Invoke-EnvironmentPython -Arguments @('-c', 'import sys; assert sys.version_info[:2] == (3, 10), sys.version; print(sys.executable)')
    if ($SetupOnly) { return }
    if ($TokenizersArchive) {
        Invoke-EnvironmentPython -Arguments @((Join-Path $projectRoot 'tools\pack_tokenizers.py'), '--import', $TokenizersArchive)
    }
    if ($MaxLines -lt 0 -or $BatchSize -le 0) { throw 'MaxLines cannot be negative; BatchSize must be positive.' }
    $evaluationArguments = @((Join-Path $projectRoot 'evaluate.py'), '--batch-size', "$BatchSize")
    if ($Corpus) { $evaluationArguments += @('--corpus', $Corpus) }
    if ($OutputDir) { $evaluationArguments += @('--output-dir', $OutputDir) }
    if ($MaxLines -gt 0) { $evaluationArguments += @('--max-lines', "$MaxLines") }
    Invoke-EnvironmentPython -Arguments $evaluationArguments
}
finally {
    foreach ($entry in $savedEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
    Pop-Location
}

