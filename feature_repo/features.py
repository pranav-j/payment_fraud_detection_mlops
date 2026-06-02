from datetime import timedelta

from data_sources import sender_features_source
from entities import sender
from feast import FeatureView, Field
from feast.types import Float64, Int32

sender_transaction_features = FeatureView(
    name="sender_transaction_features",
    entities=[sender],
    ttl=timedelta(days=7),
    schema=[
        Field(name="sender_txn_count_1h", dtype=Int32),
        Field(name="sender_txn_count_24h", dtype=Int32),
        Field(name="sender_amount_sum_24h", dtype=Float64),
        Field(name="sender_amount_mean_historical", dtype=Float64),
        Field(name="sender_time_since_last_txn", dtype=Float64),
    ],
    source=sender_features_source,
    online=True,
)
