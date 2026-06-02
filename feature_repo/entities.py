from feast import Entity, ValueType

sender = Entity(
    name="sender",
    value_type=ValueType.STRING,
    description="Sender account ID (nameOrig in PaySim)",
)
