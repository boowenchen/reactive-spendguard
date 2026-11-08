param(
    [Parameter(Mandatory=$true)][string]$Step,
    [Parameter(Mandatory=$true)][ValidateSet('base','lasna')][string]$Network,
    [switch]$Broadcast
)
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    $cfg = Get-Content -Raw config/networks.json | ConvertFrom-Json
    $net = $cfg.$Network
    $env:STEP = $Step
    $env:EXPECTED_CHAIN_ID = [string]$net.chainId
    $env:SVM_HOME = Join-Path $PWD '.toolchain/svm'
    $command = Get-Command forge -ErrorAction SilentlyContinue
    $forge = if ($command) { $command.Source } else { Join-Path $env:USERPROFILE '.foundry/bin/forge.exe' }
    $rpc = [Environment]::GetEnvironmentVariable(($Network.ToUpper() + '_RPC'))
    if (-not $rpc) { $rpc = $net.rpc }
    $arguments = @('script','script/Deploy.s.sol:Deploy','--rpc-url',$rpc)
    if ($Broadcast) { $arguments += '--broadcast' }
    & $forge @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Deployment stage failed; reconcile broadcast receipts before rerunning' }
} finally { Pop-Location }
