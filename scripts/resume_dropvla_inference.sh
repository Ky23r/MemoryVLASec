#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/dropvla/resume_dropvla_inference.sh" "$@"
