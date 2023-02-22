from datetime import timedelta

from feast import Entity, Field, FeatureView,FileSource,ValueType
from feast.types import Int64, String

msg_id = Entity(name="msg_id", value_type=ValueType.INT64)

msg_hourly_datas = FileSource(
    path="/workspace/feature_store/data/msg_data.parquet",
    event_timestamp_column="event_timestamp"
)


msg_hourly_datas_view = FeatureView(
    name="msg_datas",
    entities= [msg_id],
    ttl=timedelta(days=0),
    schema=[
        Field(name="msg_body", dtype=String),
        Field(name="category", dtype=Int64),
    ],
    online=False,
    source=msg_hourly_datas,
)