#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${runtime_dir}/.." && pwd)"

if [[ -x "${repo_dir}/.venv/bin/python" ]]; then
  python_bin="${repo_dir}/.venv/bin/python"
elif [[ -x "${repo_dir}/.venv/Scripts/python.exe" ]]; then
  python_bin="${repo_dir}/.venv/Scripts/python.exe"
else
  python_bin="${PYTHON:-python}"
fi

exec "${python_bin}" "${runtime_dir}/run.py" "$@"

