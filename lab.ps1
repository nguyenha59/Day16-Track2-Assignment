# Lab 16 helper (Windows PowerShell). Run from the repo root after `terraform apply`.
#
#   .\lab.ps1 config   # generate terraform\ssh_config from terraform outputs
#   .\lab.ps1 upload   # copy benchmark.py (+ ~/.kaggle/kaggle.json if present) to the compute node
#   .\lab.ps1 run      # wait for user_data, download dataset, run benchmark (output -> results\)
#   .\lab.ps1 stats    # print CPU / RAM / network usage of the compute node
#   .\lab.ps1 fetch    # download benchmark_result.json + log into results\
#   .\lab.ps1 ssh      # interactive SSH into the compute node (via bastion)
#   .\lab.ps1 bastion  # interactive SSH into the bastion host
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("config", "upload", "run", "stats", "fetch", "ssh", "bastion")]
    [string]$Action
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$TfDir = Join-Path $Root "terraform"
$SshConfig = Join-Path $TfDir "ssh_config"
$Results = Join-Path $Root "results"
$Ssh = "C:\Windows\System32\OpenSSH\ssh.exe"
$Scp = "C:\Windows\System32\OpenSSH\scp.exe"

function Invoke-Node([string]$Cmd) {
    & $Ssh -F $SshConfig lab-node $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed (exit $LASTEXITCODE)" }
}

function Assert-Config {
    if (-not (Test-Path $SshConfig)) { throw "Missing $SshConfig - run '.\lab.ps1 config' first" }
}

switch ($Action) {
    "config" {
        $out = terraform "-chdir=$TfDir" output -json | ConvertFrom-Json
        $key = (Join-Path $TfDir "lab-key") -replace "\\", "/"
        $known = (Join-Path $TfDir "known_hosts") -replace "\\", "/"
        Remove-Item (Join-Path $TfDir "known_hosts") -ErrorAction SilentlyContinue
        @"
Host lab-bastion
    HostName $($out.bastion_public_ip.value)
    User ubuntu
    IdentityFile $key
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    UserKnownHostsFile $known

Host lab-node
    HostName $($out.gpu_private_ip.value)
    User ubuntu
    IdentityFile $key
    IdentitiesOnly yes
    ProxyJump lab-bastion
    StrictHostKeyChecking accept-new
    UserKnownHostsFile $known
"@ | Set-Content -Encoding ascii $SshConfig
        Write-Host "Wrote $SshConfig"
        Write-Host "  bastion = $($out.bastion_public_ip.value)"
        Write-Host "  node    = $($out.gpu_private_ip.value)"
    }
    "upload" {
        Assert-Config
        Invoke-Node "mkdir -p ~/ml-benchmark ~/.kaggle"
        & $Scp -F $SshConfig (Join-Path $Root "benchmark\benchmark.py") "lab-node:ml-benchmark/benchmark.py"
        $kaggle = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
        if (Test-Path $kaggle) {
            & $Scp -F $SshConfig $kaggle "lab-node:.kaggle/kaggle.json"
            Invoke-Node "chmod 600 ~/.kaggle/kaggle.json"
            Write-Host "Uploaded benchmark.py + kaggle.json"
        } else {
            Write-Host "Uploaded benchmark.py (no $kaggle found - benchmark will fall back to OpenML)"
        }
    }
    "run" {
        Assert-Config
        $remote = @'
set -e
echo "Waiting for user_data to finish installing packages..."
until python3 -c "import lightgbm, sklearn, pandas, numpy" 2>/dev/null; do sleep 10; done
echo "ML environment OK"
cd ~/ml-benchmark
if [ ! -f creditcard.csv ] && [ -f ~/.kaggle/kaggle.json ]; then
  kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/
fi
python3 benchmark.py 2>&1 | tee benchmark_output.log
'@
        Invoke-Node ($remote -replace "`r", "")
    }
    "stats" {
        Assert-Config
        Invoke-Node "echo '=== nproc'; nproc; echo '=== free -h'; free -h; echo '=== top'; top -bn1 | head -20; echo '=== ip -s link'; ip -s link"
    }
    "fetch" {
        Assert-Config
        New-Item -ItemType Directory -Force $Results | Out-Null
        & $Scp -F $SshConfig "lab-node:ml-benchmark/benchmark_result.json" "lab-node:ml-benchmark/benchmark_output.log" $Results
        Write-Host "Saved to $Results"
    }
    "ssh" { Assert-Config; & $Ssh -F $SshConfig lab-node }
    "bastion" { Assert-Config; & $Ssh -F $SshConfig lab-bastion }
}
