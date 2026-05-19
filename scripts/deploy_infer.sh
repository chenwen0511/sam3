#!/usr/bin/env bash
# Deploy infer.py to the path used by sam3_seg_backend (default: /home/mui/sam3/scripts/infer.py).
# Run as a user that can write the target, e.g.:
#   sudo -u mui bash scripts/deploy_infer.sh
#   SAM3_INFER_DEPLOY_TARGET=/path/to/infer.py bash scripts/deploy_infer.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/scripts/infer.py"
TARGET="${SAM3_INFER_DEPLOY_TARGET:-/home/mui/sam3/scripts/infer.py}"

if [[ ! -f "${SRC}" ]]; then
  echo "Source not found: ${SRC}" >&2
  exit 1
fi

mkdir -p "$(dirname "${TARGET}")"
cp "${SRC}" "${TARGET}"
chmod +x "${TARGET}" 2>/dev/null || true

echo "Deployed:"
echo "  ${SRC}"
echo "  -> ${TARGET}"
echo ""
echo "Verify:"
echo "  ${SAM3_PYTHON:-python} ${TARGET} --help | head -5"
echo "  (should list --threshold and --mask-threshold)"
