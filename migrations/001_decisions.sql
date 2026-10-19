-- The decision audit log: one row per scored transaction.
--
-- This table already existed in RDS, created by hand during the build. The DDL
-- below was reconstructed from the INSERT statements that write to it
-- (lambda/handler.py, scripts/seed_decisions.py, scripts/inject_drift.py) and is
-- written as IF NOT EXISTS so that applying it against the live database is a
-- no-op rather than a surprise.
--
-- Deliberately append-only: nothing updates or deletes a decision. It is the
-- record of what the model said at the time, which is what makes it usable as
-- an audit trail and as the input to drift and performance monitoring.

CREATE TABLE IF NOT EXISTS decisions (
    id              BIGSERIAL PRIMARY KEY,
    transaction_id  TEXT        NOT NULL,
    step            INTEGER,
    type            TEXT,
    amount          DOUBLE PRECISION,
    is_fraud        BOOLEAN     NOT NULL,
    probability     DOUBLE PRECISION NOT NULL,
    threshold       DOUBLE PRECISION NOT NULL,
    model_version   TEXT,
    feature_set     TEXT,
    scored_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- The drift flow filters on scored_at; the performance flow joins on
-- transaction_id. Both are sequential scans without these.
CREATE INDEX IF NOT EXISTS decisions_scored_at_idx ON decisions (scored_at);
CREATE INDEX IF NOT EXISTS decisions_transaction_id_idx ON decisions (transaction_id);
