#!/usr/bin/env bash
# Download the PaySim synthetic mobile money dataset from Kaggle.
#
# Prerequisites:
#   1. Kaggle account with API token: https://www.kaggle.com/settings/account
#   2. Either:
#      - Set KAGGLE_USERNAME and KAGGLE_KEY in .env, OR
#      - Place kaggle.json at ~/.kaggle/kaggle.json
#   3. uv-managed environment with kaggle installed (we'll install it inline if not)
#
# Usage: bash scripts/download_data.sh

set -euo pipefail

DATA_DIR="data/raw"
DATASET_NAME="ealaxi/paysim1"
TARGET_FILE="${DATA_DIR}/PS_20174392719_1491204439457_log.csv"

mkdir -p "${DATA_DIR}"

if [[ -f "${TARGET_FILE}" ]]; then
    echo "✓ Dataset already present at ${TARGET_FILE}"
    echo "  Size: $(du -h "${TARGET_FILE}" | cut -f1)"
    exit 0
fi

# Load .env if it exists
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

echo "Downloading PaySim dataset from Kaggle..."
echo "Dataset: ${DATASET_NAME}"
echo "Target:  ${DATA_DIR}/"

# Use uv to run kaggle without polluting the global env
uv run --with kaggle kaggle datasets download \
    -d "${DATASET_NAME}" \
    -p "${DATA_DIR}" \
    --unzip

echo ""
echo "✓ Download complete."
ls -lh "${DATA_DIR}"
