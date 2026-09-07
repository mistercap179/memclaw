"""Discover which Claude models the user's credential can actually reach.

Anthropic's /v1/models endpoint returns every model available to the
configured credential, each with a capability tree describing which effort
levels it accepts. The setup wizard builds its model picker from this, so a
model Anthropic releases shows up with no code change here.

The result is cached in ~/.memclaw/models_cache.json for a day, so repeated
`memclaw configure` runs don't re-hit the API.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx

from .backends.claude import _claude_auth_mode

if TYPE_CHECKING:
    from pathlib import Path

    from .config import MemclawConfig


MODELS_URL = "https://api.anthropic.com/v1/models"
API_VERSION = "2023-06-01"

# Not required by /v1/models, but other OAuth-authenticated endpoints reject
# the token without it. Sent for consistency so this header never becomes a
# surprise when the same auth path is reused elsewhere.
OAUTH_BETA = "oauth-2025-04-20"

REQUEST_TIMEOUT = 10.0

# The endpoint pages at 20 by default; ask for more so a long model list
# doesn't get silently truncated.
PAGE_LIMIT = 100

# Effort levels claude-agent-sdk accepts — ClaudeAgentOptions.effort is
# Optional[Literal["low", "medium", "high", "xhigh", "max"]]. The API may
# report a level a newer SDK understands and ours does not, so anything
# outside this set is dropped. Widen the tuple when the SDK gains a level.
SDK_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

CACHE_FILENAME = "models_cache.json"
CACHE_TTL_SECONDS = 24 * 60 * 60


@dataclass
class ModelInfo:
    id: str
    display_name: str
    created_at: str
    effort_levels: list[str]  # empty when the model has no effort support


# ── Request building ────────────────────────────────────────────────

def _build_headers(config: "MemclawConfig") -> dict[str, str]:
    """Pick the auth header matching the configured credential.

    The two credential types are not interchangeable: an OAuth token sent as
    `x-api-key` comes back 401, so these cases must stay separate.
    """
    headers = {"anthropic-version": API_VERSION}
    mode = _claude_auth_mode(config)
    if mode == "subscription":
        headers["Authorization"] = f"Bearer {config.claude_code_oauth_token}"
        headers["anthropic-beta"] = OAUTH_BETA
    elif mode == "api_key":
        headers["x-api-key"] = config.anthropic_api_key
    else:
        raise RuntimeError("No Claude credential is configured.")
    return headers


# ── Response parsing ────────────────────────────────────────────────

def _effort_levels(capabilities: dict[str, Any]) -> list[str]:
    """Read the supported effort levels off one model's capability tree.

    Iterating SDK_EFFORT_LEVELS rather than the API's own keys is what makes
    an unknown future level drop out on its own.
    """
    effort = capabilities.get("effort")
    if not isinstance(effort, dict) or not effort.get("supported"):
        return []
    return [
        level
        for level in SDK_EFFORT_LEVELS
        if isinstance(effort.get(level), dict) and effort[level].get("supported")
    ]


def _created_key(created_at: str) -> datetime:
    """Parse an ISO-8601 timestamp into something sortable.

    Python 3.10's fromisoformat rejects a trailing "Z", and aware and naive
    datetimes can't be compared to each other, so both are normalised here.
    A value we can't parse sorts last instead of breaking the whole list.
    """
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_models(payload: dict[str, Any]) -> list[ModelInfo]:
    models = [
        ModelInfo(
            id=entry["id"],
            display_name=entry.get("display_name") or entry["id"],
            created_at=entry.get("created_at") or "",
            effort_levels=_effort_levels(entry.get("capabilities") or {}),
        )
        for entry in payload.get("data") or []
        if entry.get("id")
    ]
    models.sort(key=lambda m: _created_key(m.created_at), reverse=True)
    return models


# ── Cache ───────────────────────────────────────────────────────────

def _cache_path(config: "MemclawConfig") -> "Path":
    return config.memory_dir / CACHE_FILENAME


def _read_cache(config: "MemclawConfig") -> list[ModelInfo] | None:
    """Return cached models while they're fresh, else None.

    A missing, unreadable or malformed cache is not an error — we just go
    back to the network. The wizard must never break over a bad cache file.
    """
    try:
        raw = json.loads(_cache_path(config).read_text(encoding="utf-8"))
        age = time.time() - float(raw["fetched_at"])
        # A negative age means the clock moved; refetch rather than trust it.
        if not 0 <= age <= CACHE_TTL_SECONDS:
            return None
        return [
            ModelInfo(
                id=str(entry["id"]),
                display_name=str(entry["display_name"]),
                created_at=str(entry["created_at"]),
                effort_levels=[str(level) for level in entry["effort_levels"]],
            )
            for entry in raw["models"]
        ]
    except Exception:
        return None


def _write_cache(config: "MemclawConfig", models: list[ModelInfo]) -> None:
    """Persist a freshly fetched list. Failing to write is never fatal."""
    payload = {"fetched_at": time.time(), "models": [asdict(m) for m in models]}
    try:
        _cache_path(config).write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
    except OSError:
        pass


# ── Entry point ─────────────────────────────────────────────────────

async def fetch_models(config: "MemclawConfig") -> list[ModelInfo]:
    """Fetch models from /v1/models, newest first.

    Serves a cached list when one is younger than CACHE_TTL_SECONDS.
    Raises on network or auth failure — the caller decides what to do.
    """
    cached = _read_cache(config)
    if cached is not None:
        return cached

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.get(
            MODELS_URL,
            headers=_build_headers(config),
            params={"limit": PAGE_LIMIT},
        )
        response.raise_for_status()
        payload = response.json()

    models = _parse_models(payload)
    _write_cache(config, models)
    return models
