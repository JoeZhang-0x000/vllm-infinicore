#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="root@ssh.v5000-prod-gw.nhss.zhejianglab.com"
REMOTE_PORT="30358"
REMOTE_DIR="/root/vllm-infinicore"

if [ "${VLLM_INFINICORE_SYNC_MODE:-tar}" = "rsync" ] && command -v rsync >/dev/null 2>&1; then
  rsync -az --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    -e "ssh -p ${REMOTE_PORT}" \
    ./ "${REMOTE_HOST}:${REMOTE_DIR}"
else
  COPYFILE_DISABLE=1 tar \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='._*' \
    -czf - . \
    | ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" \
      "rm -rf '${REMOTE_DIR}' && mkdir -p '${REMOTE_DIR}' && tar -xzf - -C '${REMOTE_DIR}'"
fi
