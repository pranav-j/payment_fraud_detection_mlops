from feast import FileSource
from feast.data_format import ParquetFormat

sender_features_source = FileSource(
    name="sender_features_source",
    path="s3://fraud-mlops-kidiloski/feast/paysim_repeat_senders.parquet",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
)
