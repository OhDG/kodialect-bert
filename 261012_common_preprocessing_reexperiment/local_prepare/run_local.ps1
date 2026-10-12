param(
    [string]$SplitManifest = '',
    [int]$Workers = 8
)

$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    $taskPython = Join-Path $PSScriptRoot '..\..\261012_base_only_reexperiment\local_prepare\.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw 'Local Python environment is missing. Create .venv here and install requirements_local.txt.'
}
$taskArgs = @('-X', 'utf8', (Join-Path $PSScriptRoot 'prepare_data.py'), '--workers', [string]$Workers)
if ($SplitManifest) { $taskArgs += @('--split-manifest', $SplitManifest) }
& $taskPython @taskArgs
if ($LASTEXITCODE -ne 0) { throw "Common data preparation failed with exit code $LASTEXITCODE." }
