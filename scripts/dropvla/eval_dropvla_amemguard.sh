#!/usr/bin/env bash
set -euo pipefail
echo "ERROR: the obsolete DropVLA defense workflow was removed. The completed DropVLA run used --defense none. Current A-MemGuard lives in defenses/amemguard/; integration with the pinned DropVLA model requires separate validation." >&2
exit 2
