"""Feast feature materialization flow.

Runs daily to update the Redis online store with the latest
sender features from the S3 parquet offline store.
"""

from __future__ import annotations

import os
from datetime import datetime

from dotenv import load_dotenv
from prefect import flow, get_run_logger, task

load_dotenv()


@task(retries=2, retry_delay_seconds=30)
def check_redis_connection(redis_host: str, redis_port: int = 6379) -> bool:
    """Verify Redis is reachable before attempting materialization."""
    import redis

    logger = get_run_logger()
    client = redis.Redis(host=redis_host, port=redis_port, socket_connect_timeout=5)
    client.ping()
    count = client.dbsize()
    logger.info("Redis connected. Current key count: %d", count)
    return True


@task(retries=1, retry_delay_seconds=60)
def run_feast_materialize(repo_path: str, end_date: str) -> int:
    """Run feast materialize for the PaySim date range."""
    import os

    from dotenv import load_dotenv
    from feast import FeatureStore

    logger = get_run_logger()
    load_dotenv()

    store = FeatureStore(repo_path=repo_path)

    # PaySim data spans 2024-01-01 to 2024-01-31
    # We always materialize the full range since it's a static dataset
    start = datetime(2024, 1, 1)
    end = datetime(2024, 2, 1)

    logger.info("Materializing features from %s to %s", start, end)
    store.materialize(start_date=start, end_date=end)

    # Count keys after materialization
    import redis

    conn_str = os.environ.get("REDIS_CONNECTION_STRING", "172.30.0.198:6379")
    host, port = conn_str.split(":")
    client = redis.Redis(host=host, port=int(port), socket_connect_timeout=5)
    count = client.dbsize()
    logger.info("Materialization complete. Redis key count: %d", count)
    return count


@task
def verify_feature_lookup(repo_path: str, sample_sender: str) -> dict:
    """Spot-check that features are readable after materialization."""
    from dotenv import load_dotenv
    from feast import FeatureStore

    load_dotenv()
    logger = get_run_logger()

    store = FeatureStore(repo_path=repo_path)
    result = store.get_online_features(
        features=[
            "sender_transaction_features:sender_txn_count_1h",
            "sender_transaction_features:sender_amount_mean_historical",
        ],
        entity_rows=[{"sender": sample_sender}],
    ).to_dict()

    logger.info("Spot check for sender %s: %s", sample_sender, result)
    return result


@flow(name="feast-materialization", log_prints=True)
def materialize_features_flow(
    redis_host: str = os.environ.get("REDIS_CONNECTION_STRING", "172.30.0.198:6379").split(":")[0],
    repo_path: str = "feature_repo",
) -> None:
    """Daily flow: refresh Redis online store from S3 parquet offline store."""
    logger = get_run_logger()
    logger.info("Starting Feast materialization flow")

    # Step 1: verify Redis is up
    check_redis_connection(redis_host=redis_host)

    # Step 2: materialize
    end_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    key_count = run_feast_materialize(repo_path=repo_path, end_date=end_date)

    # Step 3: spot check
    verify_feature_lookup(
        repo_path=repo_path,
        sample_sender="C1892869131",
    )

    logger.info("Materialization flow complete. %d keys in Redis.", key_count)


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    materialize_features_flow(redis_host=os.environ.get("REDIS_PUBLIC_IP", "172.30.0.198"))
