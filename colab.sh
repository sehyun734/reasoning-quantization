#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

session="$(openssl rand -hex 3)"
tar_path="/tmp/$session.tar.gz"
argv="$(python3 -c "import sys; print(sys.argv[1:])" "$@")"
# skip torch (colab has it) and local-only packages
# vllm can't install on mac, so pin it here instead of requirements.txt
packages="$(grep -E "^(accelerate|datasets|huggingface_hub|math-verify|safetensors|simple-parsing|transformers)==" requirements.txt | python3 -c "import sys; print([line.strip() for line in sys.stdin] + ['vllm==0.30.0'])")"

log_path="logs/$(basename "$1" .py)/$(date +%y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$log_path")"

# on exit, delete tarball and stop colab session
trap 'rm -f "$tar_path"; colab stop -s "$session" || true' EXIT

# start colab session
colab new -s "$session" --gpu G4
# upload project without local-only files
COPYFILE_DISABLE=1 tar -czf "$tar_path" --exclude .venv --exclude __pycache__ --exclude logs .
colab upload -s "$session" "$tar_path" content/project.tar.gz

# extract, install packages, and run entry
# output goes to both terminal and log
colab exec -s "$session" --timeout 86400 <<EOF 2>&1 | tee "$log_path"
import subprocess
import tarfile
tarfile.open("project.tar.gz").extractall(filter="data")
subprocess.run(["uv", "pip", "install", "--system", *$packages], check=True)
proc = subprocess.Popen(
    # export .env vars to python
    # PYTHONPATH=. to import primitives
    # -u to disable output buffering
    ["bash", "-c", 'set -a && . ./.env && set +a && PYTHONPATH=. exec python -u "\$@"', "bash", *$argv],
    # one stream, so errors show up in order and in the log
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
for line in proc.stdout:
    print(line, end="", flush=True)
if proc.wait() != 0:
    print("fail", flush=True)
EOF
