# ============================================================
# utils/logger.py — Enterprise Logging & Validation
# ============================================================

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


# ─────────────────────────────────────────────
# ANSI Color Codes for Console
# ─────────────────────────────────────────────
COLORS = {
    "DEBUG":    "\033[36m",    # Cyan
    "INFO":     "\033[32m",    # Green
    "WARNING":  "\033[33m",    # Yellow
    "ERROR":    "\033[31m",    # Red
    "CRITICAL": "\033[35m",    # Magenta
    "RESET":    "\033[0m",
    "BOLD":     "\033[1m",
}

ICONS = {
    "DEBUG":    "🔍",
    "INFO":     "✅",
    "WARNING":  "⚠️ ",
    "ERROR":    "❌",
    "CRITICAL": "🔥",
}


class ColoredFormatter(logging.Formatter):
    """Colored + icon-based console formatter."""

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color     = COLORS.get(levelname, COLORS["RESET"])
        icon      = ICONS.get(levelname, "  ")
        reset     = COLORS["RESET"]
        bold      = COLORS["BOLD"]

        # Format: TIME | LEVEL | MODULE | MESSAGE
        time_str  = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        module    = record.name.split(".")[-1][:20]
        msg       = record.getMessage()

        return (
            f"{color}{bold}{time_str}{reset} "
            f"{icon} {color}{levelname:<8}{reset} "
            f"│ {bold}{module:<20}{reset} │ {msg}"
        )


class JSONFormatter(logging.Formatter):
    """Structured JSON formatter for file logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp":  datetime.fromtimestamp(record.created).isoformat(),
            "level":      record.levelname,
            "module":     record.name,
            "function":   record.funcName,
            "line":       record.lineno,
            "message":    record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


class AuditLogger:
    """
    Specialized audit logger for security events.
    Writes to a separate audit log file with full context.
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.log_dir / "audit.jsonl"

    def log_event(
        self,
        event_type: str,
        user_input: str,
        threat_level: str,
        details: Dict[str, Any],
        blocked: bool = False,
    ) -> None:
        """Write a single audit event as a JSON line."""
        entry = {
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            "event_type":  event_type,
            "threat_level": threat_level,
            "blocked":     blocked,
            "input_preview": user_input[:200],
            "details":     details,
        }
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_query(
        self,
        query: str,
        retrieved_count: int,
        execution_time: float,
        model_used: str,
    ) -> None:
        """Log every query with performance metrics."""
        entry = {
            "timestamp":       datetime.utcnow().isoformat() + "Z",
            "event_type":      "query",
            "query_preview":   query[:200],
            "retrieved_count": retrieved_count,
            "execution_time_s": round(execution_time, 4),
            "model_used":      model_used,
        }
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_logger(
    name: str,
    log_dir: str   = "./logs",
    log_level: str = "INFO",
    enable_file: bool = True,
    enable_console: bool = True,
) -> logging.Logger:
    """
    Factory: returns a fully-configured logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("System started")
        logger.warning("Low memory")
        logger.error("Failed to load PDF", exc_info=True)
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    # ── Console Handler ─────────────────────────────────────
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)

    # ── File Handler (rotating JSON) ─────────────────────────
    if enable_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename     = log_path / "rag_system.log",
            maxBytes     = 10_000_000,   # 10 MB
            backupCount  = 5,
            encoding     = "utf-8",
        )
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger


# ─────────────────────────────────────────────
# Performance Timer (context manager)
# ─────────────────────────────────────────────
class Timer:
    """
    Context manager to measure execution time.

    Usage:
        with Timer("embedding generation") as t:
            embeddings = model.encode(texts)
        logger.info(f"Took {t.elapsed:.3f}s")
    """

    def __init__(self, label: str = ""):
        self.label   = label
        self.elapsed = 0.0

    def __enter__(self):
        import time
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time
        self.elapsed = time.perf_counter() - self._start


# ─────────────────────────────────────────────
# Validation Helpers
# ─────────────────────────────────────────────
def validate_pdf_path(path: str) -> bool:
    """Check PDF file exists and is readable."""
    p = Path(path)
    return p.exists() and p.is_file() and p.suffix.lower() == ".pdf"


def validate_api_key(key: str, name: str = "API Key") -> bool:
    """Basic API key validation."""
    if not key or key.strip() == "":
        return False
    if len(key) < 10:
        return False
    return True


def truncate_text(text: str, max_chars: int = 100) -> str:
    """Truncate text for display/logging."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [{len(text) - max_chars} more chars]"
