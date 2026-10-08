#!/usr/bin/env bash
# Each phase is independently callable, logged and GPU scheduled.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_ROOT="${DROPVLA_RUN_ROOT:-${PROJECT_ROOT}/output/dropvla_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$RUN_ROOT" ]]; then
    echo "ERROR: RUN_ROOT already exists: $RUN_ROOT; use individual phases for a prepared run." >&2
    exit 1
fi
for phase in prepare preflight train baseline clean trigger summary; do
    bash "${SCRIPT_DIR}/wait_dropvla.sh" "$phase" "$RUN_ROOT"
done
echo "DropVLA workflow complete: $RUN_ROOT"
