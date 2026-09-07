"""Tests for the Anthropic /v1/models discovery module.

Every HTTP call is mocked — the suite runs with no API keys and no network.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from memclaw import anthropic_models
from memclaw.anthropic_models import ModelInfo, fetch_models
from memclaw.config import MemclawConfig


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_credentials(monkeypatch):
    """Keep the developer shell env out of MemclawConfig.

    __post_init__ falls back to os.environ for blank fields, so a real token
    in the parent shell would override what these tests configure.
    """
    for name in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY",
                 "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _make_config(tmp_path: Path, *, oauth: str = "", api_key: str = "") -> MemclawConfig:
    return MemclawConfig(
        memory_dir=tmp_path / "m",
        openai_api_key="test-openai-key",
        anthropic_api_key=api_key,
        claude_code_oauth_token=oauth,
    )


def _effort(*levels: str, supported: bool = True) -> dict:
    """Build an `effort` capability node advertising *levels*."""
    node: dict = {"supported": supported}
    for level in ("low", "medium", "high", "xhigh", "max"):
        node[level] = {"supported": level in levels}
    return node


def _entry(model_id: str, created_at: str, *, effort: dict | None = None,
           display_name: str | None = None) -> dict:
    return {
        "id": model_id,
        "display_name": display_name or model_id,
        "created_at": created_at,
        "capabilities": {"effort": effort} if effort is not None else {},
    }


def _mock_httpx(payload: dict | None = None, *, calls: list | None = None,
                get_error: Exception | None = None,
                status_error: Exception | None = None):
    """Patch httpx.AsyncClient with a fake that records the request it got.

    Returns the patcher; use it as a context manager.
    """
    response = MagicMock()
    response.raise_for_status = MagicMock(side_effect=status_error)
    response.json = MagicMock(return_value=payload or {"data": []})

    async def _get(url, headers=None, params=None):
        if calls is not None:
            calls.append({"url": url, "headers": headers, "params": params})
        if get_error is not None:
            raise get_error
        return response

    client = MagicMock()
    client.get = _get

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=None)

    return patch("memclaw.anthropic_models.httpx.AsyncClient",
                 MagicMock(return_value=ctx))


# ────────────────────────────────────────────────────────────────────
# Authentication headers (spec 4.2)
# ────────────────────────────────────────────────────────────────────

class TestAuthHeaders:
    @pytest.mark.asyncio
    async def test_api_key_sends_x_api_key(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="sk-ant-test")
        calls: list = []
        with _mock_httpx(calls=calls):
            await fetch_models(cfg)

        headers = calls[0]["headers"]
        assert headers["x-api-key"] == "sk-ant-test"
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_oauth_sends_bearer_not_x_api_key(self, tmp_path: Path):
        """An OAuth token sent as x-api-key gets a 401 — it must never happen."""
        cfg = _make_config(tmp_path, oauth="oat-token")
        calls: list = []
        with _mock_httpx(calls=calls):
            await fetch_models(cfg)

        headers = calls[0]["headers"]
        assert headers["Authorization"] == "Bearer oat-token"
        assert headers["anthropic-beta"] == anthropic_models.OAUTH_BETA
        assert "x-api-key" not in headers

    @pytest.mark.asyncio
    async def test_api_version_always_sent(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="sk-ant-test")
        calls: list = []
        with _mock_httpx(calls=calls):
            await fetch_models(cfg)

        assert calls[0]["headers"]["anthropic-version"] == anthropic_models.API_VERSION

    @pytest.mark.asyncio
    async def test_no_credential_raises(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        with _mock_httpx(), pytest.raises(RuntimeError, match="No Claude credential"):
            await fetch_models(cfg)


# ────────────────────────────────────────────────────────────────────
# Parsing the capability tree (spec 4.3)
# ────────────────────────────────────────────────────────────────────

class TestEffortLevels:
    @pytest.mark.asyncio
    async def test_supported_levels_come_back_in_sdk_order(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="k")
        payload = {"data": [
            _entry("claude-opus-5", "2026-07-24T00:00:00Z",
                   effort=_effort("low", "medium", "high", "xhigh", "max")),
        ]}
        with _mock_httpx(payload):
            models = await fetch_models(cfg)

        assert models[0].effort_levels == ["low", "medium", "high", "xhigh", "max"]

    @pytest.mark.asyncio
    async def test_partial_support_keeps_only_supported_levels(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="k")
        payload = {"data": [
            _entry("claude-sonnet-5", "2026-05-01T00:00:00Z",
                   effort=_effort("low", "high")),
        ]}
        with _mock_httpx(payload):
            models = await fetch_models(cfg)

        assert models[0].effort_levels == ["low", "high"]

    @pytest.mark.asyncio
    async def test_model_without_effort_support_has_no_levels(self, tmp_path: Path):
        """Haiku 4.5 reports effort.supported=false — never offer it a level."""
        cfg = _make_config(tmp_path, api_key="k")
        payload = {"data": [
            _entry("claude-haiku-4-5", "2025-10-01T00:00:00Z",
                   effort=_effort(supported=False)),
        ]}
        with _mock_httpx(payload):
            models = await fetch_models(cfg)

        assert models[0].effort_levels == []

    @pytest.mark.asyncio
    async def test_missing_effort_node_has_no_levels(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="k")
        payload = {"data": [_entry("claude-old", "2024-01-01T00:00:00Z")]}
        with _mock_httpx(payload):
            models = await fetch_models(cfg)

        assert models[0].effort_levels == []

    @pytest.mark.asyncio
    async def test_level_the_sdk_does_not_know_is_dropped(self, tmp_path: Path):
        """A level Anthropic ships before our SDK understands it must not leak."""
        cfg = _make_config(tmp_path, api_key="k")
        effort = _effort("high")
        effort["ultra"] = {"supported": True}  # hypothetical future level
        payload = {"data": [_entry("claude-future", "2027-01-01T00:00:00Z", effort=effort)]}
        with _mock_httpx(payload):
            models = await fetch_models(cfg)

        assert models[0].effort_levels == ["high"]


# ────────────────────────────────────────────────────────────────────
# Ordering and shape
# ────────────────────────────────────────────────────────────────────

class TestParsing:
    @pytest.mark.asyncio
    async def test_sorted_newest_first(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="k")
        payload = {"data": [
            _entry("old", "2024-01-01T00:00:00Z"),
            _entry("newest", "2026-07-24T00:00:00Z"),
            _entry("middle", "2025-06-01T00:00:00Z"),
        ]}
        with _mock_httpx(payload):
            models = await fetch_models(cfg)

        assert [m.id for m in models] == ["newest", "middle", "old"]

    @pytest.mark.asyncio
    async def test_unparseable_timestamp_sorts_last(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="k")
        payload = {"data": [
            _entry("broken", "not-a-date"),
            _entry("fine", "2025-01-01T00:00:00Z"),
        ]}
        with _mock_httpx(payload):
            models = await fetch_models(cfg)

        assert [m.id for m in models] == ["fine", "broken"]

    @pytest.mark.asyncio
    async def test_display_name_falls_back_to_id(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="k")
        payload = {"data": [
            {"id": "claude-bare", "created_at": "2026-01-01T00:00:00Z"},
        ]}
        with _mock_httpx(payload):
            models = await fetch_models(cfg)

        assert models[0].display_name == "claude-bare"

    @pytest.mark.asyncio
    async def test_empty_data_gives_empty_list(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="k")
        with _mock_httpx({"data": []}):
            models = await fetch_models(cfg)

        assert models == []


# ────────────────────────────────────────────────────────────────────
# Failures propagate — the caller decides what to do
# ────────────────────────────────────────────────────────────────────

class TestFailures:
    @pytest.mark.asyncio
    async def test_network_error_propagates(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="k")
        with _mock_httpx(get_error=httpx.ConnectError("no network")):
            with pytest.raises(httpx.ConnectError):
                await fetch_models(cfg)

    @pytest.mark.asyncio
    async def test_http_status_error_propagates(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="bad-key")
        error = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=MagicMock(),
        )
        with _mock_httpx(status_error=error):
            with pytest.raises(httpx.HTTPStatusError):
                await fetch_models(cfg)


# ────────────────────────────────────────────────────────────────────
# Cache
# ────────────────────────────────────────────────────────────────────

class TestCache:
    def _write_cache_file(self, cfg: MemclawConfig, *, age_seconds: float,
                          models: list[dict] | None = None) -> Path:
        path = cfg.memory_dir / anthropic_models.CACHE_FILENAME
        path.write_text(json.dumps({
            "fetched_at": time.time() - age_seconds,
            "models": models if models is not None else [
                {"id": "cached-model", "display_name": "Cached Model",
                 "created_at": "2026-01-01T00:00:00Z", "effort_levels": ["high"]},
            ],
        }), encoding="utf-8")
        return path

    @pytest.mark.asyncio
    async def test_fetch_writes_the_cache(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="k")
        payload = {"data": [_entry("claude-opus-5", "2026-07-24T00:00:00Z",
                                   effort=_effort("high", "max"))]}
        with _mock_httpx(payload):
            await fetch_models(cfg)

        raw = json.loads(
            (cfg.memory_dir / anthropic_models.CACHE_FILENAME).read_text(encoding="utf-8")
        )
        assert raw["models"][0]["id"] == "claude-opus-5"
        assert raw["models"][0]["effort_levels"] == ["high", "max"]
        assert raw["fetched_at"] <= time.time()

    @pytest.mark.asyncio
    async def test_fresh_cache_is_served_without_http(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="k")
        self._write_cache_file(cfg, age_seconds=60)

        calls: list = []
        with _mock_httpx(calls=calls):
            models = await fetch_models(cfg)

        assert calls == []
        assert models == [ModelInfo("cached-model", "Cached Model",
                                    "2026-01-01T00:00:00Z", ["high"])]

    @pytest.mark.asyncio
    async def test_stale_cache_is_refetched(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="k")
        self._write_cache_file(
            cfg, age_seconds=anthropic_models.CACHE_TTL_SECONDS + 60,
        )

        calls: list = []
        payload = {"data": [_entry("fresh-model", "2026-08-01T00:00:00Z")]}
        with _mock_httpx(payload, calls=calls):
            models = await fetch_models(cfg)

        assert len(calls) == 1
        assert [m.id for m in models] == ["fresh-model"]

    @pytest.mark.asyncio
    async def test_cache_from_the_future_is_refetched(self, tmp_path: Path):
        """A clock jump must not pin a stale list forever."""
        cfg = _make_config(tmp_path, api_key="k")
        self._write_cache_file(cfg, age_seconds=-3600)

        calls: list = []
        with _mock_httpx({"data": [_entry("fresh", "2026-08-01T00:00:00Z")]}, calls=calls):
            models = await fetch_models(cfg)

        assert len(calls) == 1
        assert [m.id for m in models] == ["fresh"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("content", [
        "not json at all",
        '{"models": []}',
        '{"fetched_at": "yesterday", "models": []}',
        '{"fetched_at": 0, "models": [{"id": "x"}]}',
    ])
    async def test_broken_cache_never_crashes(self, tmp_path: Path, content: str):
        cfg = _make_config(tmp_path, api_key="k")
        (cfg.memory_dir / anthropic_models.CACHE_FILENAME).write_text(
            content, encoding="utf-8",
        )

        calls: list = []
        with _mock_httpx({"data": [_entry("fresh", "2026-08-01T00:00:00Z")]}, calls=calls):
            models = await fetch_models(cfg)

        assert len(calls) == 1
        assert [m.id for m in models] == ["fresh"]
