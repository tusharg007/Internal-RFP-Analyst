"""Simple JSON-backed ingestion registry."""

from __future__ import annotations

import json
from pathlib import Path

from rfp_analyst.schemas import IngestionRecord


class IngestionRegistry:
    """Track ingested files so repeat runs can skip duplicates safely."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path

    def _read(self) -> dict[str, dict]:
        if not self.registry_path.exists():
            return {}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, dict]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def contains_hash(self, file_hash: str) -> bool:
        return file_hash in self._read()

    def get_record(self, file_hash: str) -> IngestionRecord | None:
        payload = self._read().get(file_hash)
        return IngestionRecord.from_dict(payload) if payload else None

    def register(self, record: IngestionRecord) -> None:
        payload = self._read()
        payload[record.file_hash] = record.to_dict()
        self._write(payload)

    def records(self) -> list[IngestionRecord]:
        return [IngestionRecord.from_dict(item) for item in self._read().values()]
