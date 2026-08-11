#!/usr/bin/env bash
set -euo pipefail

# Wait for Postgres to be available and seed minimal data, then run RFM pipeline
PGHOST=${POSTGRES_HOST:-localhost}
PGPORT=${POSTGRES_PORT:-5432}
PGUSER=${POSTGRES_USER:-olist_user}
PGDB=${POSTGRES_DB:-olist_db}

echo "Waiting for Postgres at ${PGHOST}:${PGPORT}..."
for i in {1..30}; do
  if pg_isready -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" >/dev/null 2>&1; then
    echo "Postgres is ready"
    break
  fi
  sleep 1
done

echo "Seeding database..."
psql "postgresql://${PGUSER}@${PGHOST}:${PGPORT}/${PGDB}" -f scripts/ci_seed_postgres.sql

echo "Running RFM feature pipeline (build_rfm_dataset)..."
python - <<PY
from src.features.sql_features import build_rfm_dataset
build_rfm_dataset()
print('RFM pipeline completed')
PY