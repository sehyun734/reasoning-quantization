#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

session="$(openssl rand -hex 3)"
tar_path="/tmp/$session.tar.gz"
argv="$(python3 -c "import sys; print(sys.argv[1:])" "$@")"
# vllm can't install on mac, so pin it here
packages="$(grep -E "^(accelerate|datasets|huggingface_hub|math-verify|safetensors|simple-parsing|transformers)==" requirements.txt | python3 -c "import sys; print([line.strip() for line in sys.stdin] + ['vllm==0.30.0'])")"
log_path="logs/$(basename "$1" .py)/$(date +%y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$log_path")"

# on exit, remove tarball and stop session
trap 'rm -f "$tar_path"; colab stop -s "$session" || true' EXIT

colab new -s "$session" --gpu G4
COPYFILE_DISABLE=1 tar -czf "$tar_path" --exclude .venv --exclude __pycache__ --exclude logs .
colab upload -s "$session" "$tar_path" content/project.tar.gz

colab exec -s "$session" --timeout 86400 <<EOF 2>&1 | tee "$log_path"
import subprocess
import tarfile
tarfile.open("project.tar.gz").extractall(filter="data")
subprocess.run(["uv", "pip", "install", "--system", *$packages], check=True)
proc = subprocess.Popen(
    ["bash", "-c", 'set -a && . ./.env && set +a && PYTHONPATH=. exec python -u "\$@"', "bash", *$argv],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
for line in proc.stdout:
    print(line, end="", flush=True)
if proc.wait() != 0:
    print("fail", flush=True)
EOF
