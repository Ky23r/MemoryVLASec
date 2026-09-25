#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "A-MemGuard real evaluation is defined on the attacked MemoryVLA checkpoint." >&2
exec bash "${SCRIPT_DIR}/defense_eval.sh" "$@"
