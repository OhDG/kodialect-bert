param(
    [string]$OldRoot = 'D:\documents\project_1\git\project1_dataset',
    [string]$NewRoot = 'D:\방언 데이터 총정리',
    [int]$Workers = 8,
    [string]$AuditCounts = '',
    [switch]$IncludeNewJeollaValidation
)

$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw 'Local venv is missing. Run python -m venv .venv, then .venv\Scripts\python.exe -m pip install -r requirements_local.txt here.'
}
$taskArgs = @(
    (Join-Path $PSScriptRoot 'prepare_data.py'),
    '--old-root', $OldRoot, '--new-root', $NewRoot,
    '--workers', [string]$Workers
)
if ($AuditCounts) { $taskArgs += @('--audit-counts', $AuditCounts) }
if ($IncludeNewJeollaValidation) { $taskArgs += '--include-new-jeolla-validation' }
& $taskPython @taskArgs
if ($LASTEXITCODE -ne 0) { throw "Local data preparation failed with exit code $LASTEXITCODE." }
