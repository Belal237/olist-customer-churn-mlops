"""
Logging setup shared by the CLI pipeline and the API.

Modules only create loggers (logging.getLogger(__name__)); handlers are
configured once, by the entrypoint, through setup_logging(). Configuring
logging at import time would let whichever module is imported first decide
the format for the whole process.

- CLI scripts use a human-readable format.
- The API uses one JSON object per line, ready for a log aggregator. Structured
  fields are passed with log_event() and land at the top level of the JSON.
"""

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

# Set by the API middleware for the duration of a request
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(json_format: bool = False, level: int = logging.INFO) -> None:
    """Configure the root logger once, replacing any handler already installed."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if json_format else logging.Formatter(TEXT_FORMAT))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    """Log a named event with structured fields (rendered as JSON keys by the API)."""
    logger.log(level, event, extra={"fields": {"event": event, **fields}})
