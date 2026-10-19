-- Ingest-to-decision latency for the streaming path.
--
-- `scored_at` alone cannot answer "how long after the transaction arrived did
-- we decide?", because it has no arrival time to subtract. Kinesis stamps every
-- record with ApproximateArrivalTimestamp; the handler now records it and the
-- difference.
--
-- latency_ms is stored rather than computed on read so that a dashboard query
-- is a plain aggregate, and so the number survives even if the two timestamps
-- are later found to disagree about clocks.

ALTER TABLE decisions ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS latency_ms DOUBLE PRECISION;

COMMENT ON COLUMN decisions.ingested_at IS
    'Kinesis ApproximateArrivalTimestamp for the record that produced this decision';
COMMENT ON COLUMN decisions.latency_ms IS
    'scored_at - ingested_at, in milliseconds; NULL for rows written before this column existed';
