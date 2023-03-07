from datetime import timedelta


from feast import Entity, Field, FeatureView,FileSource,ValueType, FeatureService
from feast.types import String

msg_id = Entity(name="id", value_type=ValueType.INT64)

msg_hourly_datas = FileSource(
    path="/app/datas/msg_data.parquet",
    event_timestamp_column="event_timestamp"
)


msg_hourly_datas_view = FeatureView(
    name="msg_datas",
    entities= [msg_id],
    ttl=timedelta(days=0),
    schema=[
        Field(name="msg_body", dtype=String),
    ],
    online=False,
    source=msg_hourly_datas,
)

msg_fs = FeatureService(
    name = "msg_svc",
    features=[msg_hourly_datas_view],
    tags={"description": "Used for training NLP model"}
)