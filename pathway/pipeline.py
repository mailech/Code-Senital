from __future__ import annotations

import pathway as pw


class Event(pw.Schema):
    source: str
    event_type: str
    payload: pw.Json


def build_pipeline() -> pw.Table[Event]:
    # Placeholder in-memory stream that can be replaced by webhook inputs
    events = pw.debug.table_from_rows(
        Event,
        [("bootstrap", "startup", {"message": "pipeline ready"})],
    )
    return events
