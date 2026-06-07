# Run this once to populate decisions table
import os
import random
import uuid
from datetime import datetime, timedelta

import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.environ["RDS_HOST"],
    port=int(os.environ.get("RDS_PORT", 5432)),
    dbname=os.environ.get("RDS_DB", "mlflow"),
    user=os.environ["RDS_USER"],
    password=os.environ["RDS_PASSWORD"],
    sslmode="require",
)

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
now = datetime.utcnow()

rows = []
for _ in range(2000):
    # Mostly normal, 13% fraud (matching PaySim distribution)
    is_fraud = random.random() < 0.0013
    amount = random.lognormvariate(7.0, 1.5) if not is_fraud else random.lognormvariate(9.0, 1.0)
    probability = random.betavariate(1, 8) if not is_fraud else random.betavariate(8, 1)
    threshold = 0.4234
    scored_at = now - timedelta(hours=random.randint(0, 168))  # last 7 days

    rows.append(
        (
            str(uuid.uuid4()),
            random.randint(1, 743),
            random.choice(TYPES),
            round(amount, 2),
            is_fraud,
            round(probability, 6),
            threshold,
            "2",
            "enriched_v1",
            scored_at,
        )
    )

with conn.cursor() as cur:
    cur.executemany(
        """
        INSERT INTO decisions
            (transaction_id, step, type, amount, is_fraud, probability,
             threshold, model_version, feature_set, scored_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """,
        rows,
    )
conn.commit()
conn.close()
print(f"Inserted {len(rows)} decisions")
