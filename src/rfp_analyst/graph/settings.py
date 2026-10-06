"""Instance-local configuration. No driver imports or connection side effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal
from urllib.parse import urlparse

GraphRole = Literal["reader", "writer", "admin"]


@dataclass(frozen=True)
class GraphSettings:
    enabled: bool = False
    uri: str = field(default="", repr=False)
    username: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    database: str = "neo4j"
    role: GraphRole = "reader"
    connection_timeout_seconds: float = 1.0
    acquisition_timeout_seconds: float = 2.0
    transaction_timeout_seconds: float = 2.0
    max_connection_pool_size: int = 10
    max_batch_size: int = 500

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("Graph enabled setting must be boolean")
        if self.role not in {"reader", "writer", "admin"}:
            raise ValueError("Unsupported graph access role")
        if not self.enabled:
            return
        for value in (
            self.connection_timeout_seconds,
            self.acquisition_timeout_seconds,
            self.transaction_timeout_seconds,
        ):
            if not isfinite(value) or not 0 < value <= 60:
                raise ValueError("Graph timeouts must be finite and between 0 and 60 seconds")
        for value, ceiling in ((self.max_connection_pool_size, 50), (self.max_batch_size, 1000)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
                raise ValueError("Graph pool/batch limits are outside the supported range")
        if not self.database.strip():
            raise ValueError("Graph database must be explicit")
        if self.uri:
            parsed = urlparse(self.uri)
            if parsed.scheme not in {"bolt", "bolt+s", "neo4j", "neo4j+s"} or not parsed.hostname:
                raise ValueError("Unsupported graph connection URI")
            if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path:
                raise ValueError(
                    "Graph URI must not contain credentials, paths or query parameters"
                )
            if parsed.scheme in {"bolt", "neo4j"} and parsed.hostname not in {
                "localhost",
                "127.0.0.1",
                "::1",
            }:
                raise ValueError("Remote graph connections require verified TLS (+s)")
            if parsed.port is not None and not 1 <= parsed.port <= 65535:
                raise ValueError("Invalid graph port")

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.uri and self.username and self.password)

    @classmethod
    def from_config(cls, role: GraphRole = "reader") -> GraphSettings:
        # Resolve dynamically at the composition boundary, not module import.
        from config import get_neo4j_settings

        return cls(**get_neo4j_settings(role))
