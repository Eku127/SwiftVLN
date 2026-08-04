"""Shared timestamp helpers for SatNav API responses."""

from datetime import datetime
from zoneinfo import ZoneInfo

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def now_shanghai_iso() -> str:
    """Return an ISO-8601 timestamp in Asia/Shanghai."""
    return datetime.now(SHANGHAI_TZ).isoformat()
