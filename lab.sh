#!/usr/bin/env bash
# Lab 16 helper (Linux / WSL). Run from the repo root after `terraform apply`.
#
#   ./lab.sh config   # generate ~/.ssh/lab16_config from terraform outputs
#   ./lab.sh upload   # copy benchmark.py (+ ~/.kaggle/kaggle.json if present) to the compute node
#   ./lab.sh run      # wait for user_data, download dataset, run benchmark (output -> results/)
#   ./lab.sh stats    # print CPU / RAM / network usage of the compute node
#   ./lab.sh fetch    # download benchmark_result.json + log into results/
#   ./lab.sh ssh      # interactive SSH into the compute node (via bastion)
#   ./lab.sh bastion  # interactive SSH into the bastion host
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="$ROOT/terraform"
RESULTS="$ROOT/results"
# Key + config live in ~/.ssh: files under /mnt/c are 0777 in WSL and ssh rejects such keys.
KEY="$HOME/.ssh/lab16-key"
SSH_CONFIG="$HOME/.ssh/lab16_config"
KNOWN="$HOME/.ssh/lab16_known_hosts"

node() { ssh -F "$SSH_CONFIG" lab-node "$@"; }

need_config() {
  [ -f "$SSH_CONFIG" ] || { echo "Missing $SSH_CONFIG - run './lab.sh config' first" >&2; exit 1; }
}

case "${1:-}" in
  config)
    mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
    install -m 600 "$TF_DIR/lab-key" "$KEY"
    bastion_ip=$(terraform -chdir="$TF_DIR" output -raw bastion_public_ip)
    node_ip=$(terraform -chdir="$TF_DIR" output -raw gpu_private_ip)
    if [ -z "$bastion_ip" ] || [ -z "$node_ip" ]; then
      echo "No terraform outputs found - run 'terraform -chdir=terraform apply' first" >&2
      exit 1
    fi
    rm -f "$KNOWN"
    cat > "$SSH_CONFIG" <<EOF
Host lab-bastion
    HostName $bastion_ip
    User ubuntu
    IdentityFile $KEY
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    UserKnownHostsFile $KNOWN

Host lab-node
    HostName $node_ip
    User ubuntu
    IdentityFile $KEY
    IdentitiesOnly yes
    ProxyJump lab-bastion
    StrictHostKeyChecking accept-new
    UserKnownHostsFile $KNOWN
EOF
    echo "Wrote $SSH_CONFIG (bastion=$bastion_ip, node=$node_ip)"
    ;;
  upload)
    need_config
    node "mkdir -p ~/ml-benchmark ~/.kaggle"
    scp -F "$SSH_CONFIG" "$ROOT/benchmark/benchmark.py" lab-node:ml-benchmark/benchmark.py
    if [ -f "$HOME/.kaggle/kaggle.json" ]; then
      scp -F "$SSH_CONFIG" "$HOME/.kaggle/kaggle.json" lab-node:.kaggle/kaggle.json
      node "chmod 600 ~/.kaggle/kaggle.json"
      echo "Uploaded benchmark.py + kaggle.json"
    else
      echo "Uploaded benchmark.py (no ~/.kaggle/kaggle.json - benchmark will fall back to OpenML)"
    fi
    ;;
  run)
    need_config
    node 'bash -s' <<'EOF'
set -e
echo "Waiting for user_data to finish installing packages..."
until python3 -c "import lightgbm, sklearn, pandas, numpy" 2>/dev/null; do sleep 10; done
echo "ML environment OK"
cd ~/ml-benchmark
if [ ! -f creditcard.csv ] && [ -f ~/.kaggle/kaggle.json ]; then
  kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/
fi
python3 benchmark.py 2>&1 | tee benchmark_output.log
EOF
    ;;
  stats)
    need_config
    node "echo '=== nproc'; nproc; echo '=== free -h'; free -h; echo '=== top'; top -bn1 | head -20; echo '=== ip -s link'; ip -s link"
    ;;
  fetch)
    need_config
    mkdir -p "$RESULTS"
    scp -F "$SSH_CONFIG" lab-node:ml-benchmark/benchmark_result.json lab-node:ml-benchmark/benchmark_output.log "$RESULTS/"
    echo "Saved to $RESULTS"
    ;;
  ssh)     need_config; ssh -F "$SSH_CONFIG" lab-node ;;
  bastion) need_config; ssh -F "$SSH_CONFIG" lab-bastion ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
