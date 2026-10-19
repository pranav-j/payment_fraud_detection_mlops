-- Outcome labels, and when the business is allowed to know them.
--
-- The point of this table is the gap between `event_time` and
-- `label_available_at`. In production a fraud label is a chargeback or a
-- dispute: the transaction happened, the model decided, and the truth arrives
-- days or weeks later. Monitoring that ignores this reads yesterday's recall
-- off labels that do not exist yet.
--
--   event_time         when the transaction actually happened
--   label_available_at when the label became knowable (decision + delay)
--
-- The performance flow only ever reads rows with label_available_at <= now().
-- Everything newer is counted as "not yet matured" and excluded from the
-- metrics rather than silently treated as a miss.

CREATE TABLE IF NOT EXISTS labels (
    transaction_id      TEXT        PRIMARY KEY,
    is_fraud            BOOLEAN     NOT NULL,
    event_time          TIMESTAMPTZ NOT NULL,
    label_available_at  TIMESTAMPTZ NOT NULL,
    source              TEXT        NOT NULL DEFAULT 'paysim_replay',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS labels_available_at_idx ON labels (label_available_at);

-- Ground truth as the *simulator* knows it, recorded by the producer at send
-- time. Separate from `labels` on purpose: this is what is true, `labels` is
-- what has become knowable. The replay script moves rows from here to there
-- once the configured delay has elapsed.
CREATE TABLE IF NOT EXISTS ground_truth (
    transaction_id  TEXT        PRIMARY KEY,
    is_fraud        BOOLEAN     NOT NULL,
    produced_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source          TEXT        NOT NULL DEFAULT 'producer'
);
