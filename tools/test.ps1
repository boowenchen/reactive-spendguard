$ErrorActionPreference = 'Stop'
$env:SVM_HOME = Join-Path $PSScriptRoot '../.toolchain/svm'
$forge = Get-Command forge -ErrorAction SilentlyContinue
if ($forge) { $exe = $forge.Source } else { $exe = Join-Path $env:USERPROFILE '.foundry/bin/forge.exe' }
Push-Location (Join-Path $PSScriptRoot '..')
try { & $exe test -vv; if ($LASTEXITCODE -ne 0) { throw 'forge test failed' } } finally { Pop-Location }
