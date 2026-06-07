# Run this in a Python shell to inject drifted decisions
import os
import random
import uuid
from datetime import datetime, timedelta

import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.environ["RDS_HOST"],
    port=5432,
    dbname="mlflow",
    user=os.environ["RDS_USER"],
    password=os.environ["RDS_PASSWORD"],
    sslmode="require",
)

# Insert 200 decisions with drifted amounts (100x normal) and unusual types
rows = []
for _ in range(200):
    rows.append(
        (
            str(uuid.uuid4()),
            1,
            "CASH_OUT",
            round(random.lognormvariate(14.0, 0.5), 2),  # ~1M vs normal ~1K
            False,
            round(random.random() * 0.1, 6),
            0.4234,
            "2",
            "enriched_v1",
            datetime.utcnow() - timedelta(minutes=random.randint(0, 60)),
        )
    )

with conn.cursor() as cur:
    cur.executemany(
        """
        INSERT INTO decisions
            (transaction_id, step, type, amount, is_fraud, probability,
             threshold, model_version, feature_set, scored_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """,
        rows,
    )
conn.commit()
conn.close()
print("Injected 200 drifted decisions")
