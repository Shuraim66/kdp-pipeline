"""Deterministic hashing helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_hash(data: dict[str, Any]) -> str:
    """SHA256 hex digest of a dict's canonical (sorted-key, compact) JSON.

    Deterministic for equal dicts regardless of key insertion order.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """SHA256 hex digest of raw bytes — used to fingerprint generated images."""
    return hashlib.sha256(data).hexdigest()
