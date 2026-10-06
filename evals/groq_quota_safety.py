"""Evaluation-only Groq TPM pacing and explicit TPD reset gate.

The token ledger never stores prompts or evidence. It keeps only estimates,
provider-reported usage, timestamps and wait durations for the Groq model.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from pathlib import Path

TPM_LIMIT = 8_000
TPD_LIMIT = 200_000
MODEL = "openai/gpt-oss-120b"


class QuotaSafetyStop(BaseException):
    """Fail-closed local guard; intentionally bypasses application fallbacks."""

    def __init__(self, dimension: str, detail: str):
        self.dimension = dimension
        self.detail = detail
        super().__init__(f"Evaluation stopped by {dimension} quota safety: {detail}")


def estimate_message_tokens(messages) -> int:
    """Conservative local estimate; content is transient and never persisted."""
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("o200k_base")
    except Exception as exc:  # Fail closed: no unpaced provider call.
        raise QuotaSafetyStop("TPM", "token estimator unavailable") from exc

    def content_text(value):
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return "\n".join(content_text(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        return str(value)

    raw = sum(len(encoder.encode(content_text(message))) for message in messages)
    # 25% margin plus per-message framing; output reserve is added separately.
    return math.ceil(raw * 1.25) + 12 * len(messages)


class GroqTPMPacer:
    """Cross-process, rolling-window token reservation ledger for eval calls."""

    def __init__(self, database: Path, *, tpm_limit=TPM_LIMIT, tpd_limit=TPD_LIMIT,
                 clock=time.time, sleeper=time.sleep, estimator=estimate_message_tokens):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.tpm_limit = int(tpm_limit)
        self.tpd_limit = int(tpd_limit)
        self.clock = clock
        self.sleeper = sleeper
        self.estimator = estimator
        db = self._connect()
        try:
            db.execute("""CREATE TABLE IF NOT EXISTS groq_usage (
                request_id TEXT PRIMARY KEY, model TEXT NOT NULL, created REAL NOT NULL,
                reserved INTEGER NOT NULL, actual INTEGER, outcome TEXT NOT NULL
            )""")
        finally:
            db.close()

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        return db

    def before_call(self, messages, *, max_output_tokens=2048):
        input_estimate = self.estimator(messages)
        reserve = input_estimate + max(0, int(max_output_tokens))
        if reserve > self.tpm_limit:
            raise QuotaSafetyStop("TPM", "single call reservation exceeds configured TPM limit")
        total_wait = 0.0
        while True:
            now = self.clock()
            db = self._connect()
            try:
                db.execute("BEGIN IMMEDIATE")
                db.execute("DELETE FROM groq_usage WHERE created < ?", (now - 86_400,))
                day = db.execute(
                    "SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) n FROM groq_usage WHERE created >= ?",
                    (now - 86_400,),
                ).fetchone()["n"]
                if day + reserve > self.tpd_limit:
                    db.execute("ROLLBACK")
                    raise QuotaSafetyStop("TPD", "local rolling daily token budget exhausted")
                recent = db.execute(
                    "SELECT created,COALESCE(actual,reserved) tokens FROM groq_usage WHERE created >= ? ORDER BY created",
                    (now - 60,),
                ).fetchall()
                used = sum(row["tokens"] for row in recent)
                if used + reserve <= self.tpm_limit:
                    request_id = str(uuid.uuid4())
                    db.execute(
                        "INSERT INTO groq_usage VALUES (?,?,?,?,?,?)",
                        (request_id, MODEL, now, reserve, None, "reserved"),
                    )
                    db.execute("COMMIT")
                    return {"request_id": request_id, "estimated_input_tokens": input_estimate,
                            "reserved_tokens": reserve, "pacing_wait_seconds": round(total_wait, 3)}
                # Wait until enough earliest reservations age out; recheck under lock.
                cumulative = 0
                wait_until = now + 1.0
                for row in recent:
                    cumulative += row["tokens"]
                    if used - cumulative + reserve <= self.tpm_limit:
                        wait_until = row["created"] + 60.0
                        break
                db.execute("ROLLBACK")
            finally:
                db.close()
            delay = max(0.05, wait_until - self.clock())
            self.sleeper(delay)
            total_wait += delay

    def finish_call(self, request_id, *, actual_tokens=None, outcome="completed"):
        if not request_id:
            return
        db = self._connect()
        try:
            if actual_tokens is None:
                db.execute("UPDATE groq_usage SET created=?, outcome=? WHERE request_id=?",
                           (self.clock(), outcome, request_id))
            else:
                db.execute("UPDATE groq_usage SET created=?, actual=?, outcome=? WHERE request_id=?",
                           (self.clock(), max(0, int(actual_tokens)), outcome, request_id))
        finally:
            db.close()


def assert_tpd_reset_confirmed(*, confirmed: bool):
    """Current observed TPD exhaustion is a hard default block for all new calls."""
    if not confirmed:
        raise QuotaSafetyStop("TPD", "operator confirmation of provider daily quota reset required")

