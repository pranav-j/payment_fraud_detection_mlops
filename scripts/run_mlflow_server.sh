#!/usr/bin/env bash
set -euo pipefail

# Load .env from project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
  echo "ERROR: .env not found at $PROJECT_ROOT/.env" >&2
  exit 1
fi

# Export all variables from .env
set -a
# shellcheck disable=SC1091
source "$PROJECT_ROOT/.env"
set +a

# Sanity-check required vars
: "${RDS_HOST:?RDS_HOST not set in .env}"
: "${RDS_PORT:?RDS_PORT not set in .env}"
: "${RDS_DB:?RDS_DB not set in .env}"
: "${RDS_USER:?RDS_USER not set in .env}"
: "${RDS_PASSWORD:?RDS_PASSWORD not set in .env}"
: "${MLFLOW_ARTIFACT_ROOT:?MLFLOW_ARTIFACT_ROOT not set in .env}"

BACKEND_URI="postgresql://${RDS_USER}:${RDS_PASSWORD}@${RDS_HOST}:${RDS_PORT}/${RDS_DB}"

echo "Starting MLflow server"
echo "  Backend: postgresql://${RDS_USER}:***@${RDS_HOST}:${RDS_PORT}/${RDS_DB}"
echo "  Artifacts: ${MLFLOW_ARTIFACT_ROOT}"
echo "  Listening on http://127.0.0.1:5000"
echo

exec mlflow server \
  --backend-store-uri "$BACKEND_URI" \
  --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
  --host 127.0.0.1 \
  --port 5000
