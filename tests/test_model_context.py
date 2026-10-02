"""Tests for model_context.py — local endpoint detection, token estimation, known model lookup."""

import sys
import types

import pytest

import src.model_context as model_context
from src.model_context import is_local_endpoint, estimate_tokens, _lookup_known


class _Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return ("eq", self.name, value)


class _ModelEndpoint:
    is_enabled = _Column("is_enabled")


class _Query:
    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, *conditions):
        for condition in conditions:
            if isinstance(condition, tuple) and condition[0] == "eq":
                _, field, value = condition
                self.rows = [row for row in self.rows if getattr(row, field) == value]
        return self

    def all(self):
        return list(self.rows)


class _Db:
    def __init__(self, rows):
        self.rows = rows

    def query(self, model):
        return _Query(self.rows)

    def close(self):
        pass


def _install_endpoint_db(monkeypatch, rows):
    mod = types.ModuleType("core.database")
    mod.ModelEndpoint = _ModelEndpoint
    mod.SessionLocal = lambda: _Db(rows)
    monkeypatch.setitem(sys.modules, "core.database", mod)


class TestIsLocalEndpoint:
    def test_localhost(self):
        assert is_local_endpoint("http://localhost:5000/v1/chat/completions") is True

    def test_loopback_ipv4(self):
        assert is_local_endpoint("http://127.0.0.1:8080/v1/chat/completions") is True

    def test_private_192_168(self):
        assert is_local_endpoint("http://192.168.1.1:11434/v1/chat/completions") is True

    def test_private_10(self):
        assert is_local_endpoint("http://10.0.0.5:8000/v1/chat/completions") is True

    @pytest.mark.parametrize("host", [
        "10.example-cloud.com",
        "172.16.example-cloud.com",
        "192.168.example-cloud.com",
    ])
    def test_private_prefix_dns_names_are_remote(self, host):
        assert is_local_endpoint(f"https://{host}/v1/chat/completions") is False

    def test_tailscale_100(self):
        # 100.64.0.0/10 is the CGNAT range Tailscale uses.
        assert is_local_endpoint("http://100.64.0.1:5000/v1/chat/completions") is True

    def test_configured_tailscale_proxy_is_remote(self, monkeypatch):
        _install_endpoint_db(monkeypatch, [
            types.SimpleNamespace(
                base_url="http://100.117.136.97:34521/v1",
                endpoint_kind="proxy",
                api_key="fake-key",
                is_enabled=True,
            )
        ])

        assert is_local_endpoint("http://100.117.136.97:34521/v1/chat/completions") is False

    def test_openai_is_remote(self):
        assert is_local_endpoint("https://api.openai.com/v1/chat/completions") is False

    def test_anthropic_is_remote(self):
        assert is_local_endpoint("https://api.anthropic.com/v1/messages") is False

    def test_empty_url(self):
        assert is_local_endpoint("") is False

    def test_malformed_url(self):
        assert is_local_endpoint("not-a-url") is False


class TestEstimateTokens:
    def test_empty_list(self):
        assert estimate_tokens([]) == 0

    def test_single_short_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        tokens = estimate_tokens(messages)
        # 4 overhead + int(5 * 0.3) = 4 + 1 = 5
        assert tokens == 5

    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there"},
        ]
        tokens = estimate_tokens(messages)
        assert tokens > 0
        # Each message adds 4 overhead + chars * 0.3
        assert tokens == 4 + int(16 * 0.3) + 4 + int(8 * 0.3)

    def test_multimodal_content_list(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        tokens = estimate_tokens(messages)
        # 4 overhead + int(19 * 0.3) for the text item; image_url is ignored
        assert tokens == 4 + int(19 * 0.3)

    def test_missing_content_key(self):
        messages = [{"role": "assistant"}]
        tokens = estimate_tokens(messages)
        # 4 overhead + 0 content
        assert tokens == 4

    def test_scales_with_length(self):
        short = estimate_tokens([{"role": "user", "content": "short"}])
        long_text = "a" * 10000
        long = estimate_tokens([{"role": "user", "content": long_text}])
        assert long > short * 10


class TestLookupKnown:
    def test_claude_sonnet(self):
        assert _lookup_known("claude-sonnet-4-5") == 200000

    def test_gpt4o(self):
        assert _lookup_known("gpt-4o") == 128000

    def test_deepseek_r1(self):
        assert _lookup_known("deepseek-r1") == 64000

    def test_gemini_pro(self):
        assert _lookup_known("gemini-2.5-pro") == 1048576

    def test_unknown_model(self):
        assert _lookup_known("totally-unknown-model-xyz") is None

    def test_namespaced_model(self):
        """Models prefixed with provider/ should still match."""
        result = _lookup_known("openrouter/deepseek-r1")
        assert result == 64000

    def test_model_with_tag(self):
        """Models with :free or :extended suffixes should still match."""
        result = _lookup_known("deepseek-r1:free")
        assert result == 64000

    def test_o1_mini_not_shadowed_by_o1(self):
        """'o1' (200k) precedes 'o1-mini' (128k) in the table; longest match wins."""
        assert _lookup_known("o1-mini") == 128000

    def test_o1_full(self):
        assert _lookup_known("o1") == 200000

    def test_gpt4o_mini_not_shadowed_by_gpt4(self):
        assert _lookup_known("gpt-4o-mini") == 128000

    def test_gpt4_base(self):
        assert _lookup_known("gpt-4") == 8192


class _FakeResp:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.is_success = ok

    def json(self):
        return self._payload


class TestGetContextLength:
    def setup_method(self):
        model_context._context_cache.clear()
        model_context._catalog_ctx_cache.clear()

    def test_local_endpoint_requeries_same_model_after_restart(self, monkeypatch):
        calls = []

        def fake_query(endpoint_url, model):
            calls.append((endpoint_url, model))
            return (8192, True) if len(calls) == 1 else (27000, True)

        monkeypatch.setattr(model_context, "_query_context_length", fake_query)

        endpoint = "http://127.0.0.1:8000/v1/chat/completions"
        model = "Qwen/Qwen3-14B"

        first = model_context.get_context_length(endpoint, model)
        second = model_context.get_context_length(endpoint, model)

        assert first == 8192
        assert second == 27000
        assert len(calls) == 2

    def test_remote_endpoint_keeps_cached_context(self, monkeypatch):
        calls = []

        def fake_query(endpoint_url, model):
            calls.append((endpoint_url, model))
            return (200000, True) if len(calls) == 1 else (12345, True)

        monkeypatch.setattr(model_context, "_query_context_length", fake_query)

        endpoint = "https://api.openai.com/v1/chat/completions"
        model = "gpt-5"

        first = model_context.get_context_length(endpoint, model)
        second = model_context.get_context_length(endpoint, model)

        assert first == 200000
        assert second == 200000
        assert len(calls) == 1

    def _proxy_db(self, monkeypatch):
        _install_endpoint_db(monkeypatch, [
            types.SimpleNamespace(
                base_url="http://100.117.136.97:34521/v1",
                endpoint_kind="proxy",
                api_key="fake-key",
                is_enabled=True,
            )
        ])

    def test_configured_proxy_known_model_skips_model_listing(self, monkeypatch):
        # A model covered by the known-context table must still resolve without
        # touching /models — the cheap path the proxy short-circuit exists for.
        self._proxy_db(monkeypatch)

        def fake_get(*args, **kwargs):
            raise AssertionError("/models must not be queried for a known proxy model")

        monkeypatch.setattr(model_context.httpx, "get", fake_get)

        endpoint = "http://100.117.136.97:34521/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "gpt-4o") == 128000

    def test_configured_proxy_unknown_model_reads_catalog_context(self, monkeypatch):
        # A model missing from the known table (e.g. a new OpenRouter model)
        # must report the catalog's real window, not the bare default (#4886).
        # The catalog is fetched once per endpoint and reused for other models.
        self._proxy_db(monkeypatch)
        fetches = []

        def fake_get(url, *args, **kwargs):
            fetches.append(url)
            return _FakeResp({"data": [
                {"id": "owl-alpha", "context_length": 1048576},
                {"id": "tiny-proxy-model", "context_length": 8192},
            ]})

        monkeypatch.setattr(model_context.httpx, "get", fake_get)

        endpoint = "http://100.117.136.97:34521/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "owl-alpha") == 1048576
        # A second unknown model on the same endpoint reuses the cached catalog.
        assert model_context.get_context_length(endpoint, "tiny-proxy-model") == 8192
        assert len(fetches) == 1

    def test_configured_proxy_unknown_model_falls_back_to_default(self, monkeypatch):
        # If the catalog can be read but doesn't list the model, keep the
        # conservative default rather than guessing.
        self._proxy_db(monkeypatch)

        def fake_get(url, *args, **kwargs):
            return _FakeResp({"data": [{"id": "some-other-model", "context_length": 4096}]})

        monkeypatch.setattr(model_context.httpx, "get", fake_get)

        endpoint = "http://100.117.136.97:34521/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "absent-model") == model_context.DEFAULT_CONTEXT

    def test_configured_proxy_catalog_fetch_failure_uses_default(self, monkeypatch):
        # A failed/unreachable catalog must not raise — fall back to the default.
        self._proxy_db(monkeypatch)

        def fake_get(url, *args, **kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr(model_context.httpx, "get", fake_get)

        endpoint = "http://100.117.136.97:34521/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "unknown-proxy-model") == model_context.DEFAULT_CONTEXT


class TestConfiguredContextOverride:
    """Admin-set per-model num_ctx override (Settings > Models) must win over
    discovery, the known-models table, and the cache — and apply immediately."""

    def setup_method(self):
        model_context._context_cache.clear()
        model_context._catalog_ctx_cache.clear()

    def _install_ollama_endpoint(self, monkeypatch, overrides):
        # Convenience wrapper: tests pass the flat {model_id: num_ctx} shape;
        # the DB column actually stores the generic nested
        # {model_id: {field: value}} shape.
        import json
        nested = {k: {"num_ctx": v} for k, v in (overrides or {}).items()}
        _install_endpoint_db(monkeypatch, [
            types.SimpleNamespace(
                base_url="http://localhost:11434",
                endpoint_kind="local",
                api_key=None,
                is_enabled=True,
                model_overrides=json.dumps(nested) if nested else None,
            )
        ])

    def test_override_wins_over_known_table(self, monkeypatch):
        # "gpt-4o" would normally resolve to 128000 from the known table; an
        # explicit override must take priority.
        self._install_ollama_endpoint(monkeypatch, {"gpt-4o": 32768})
        endpoint = "http://localhost:11434/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "gpt-4o") == 32768
        assert model_context.get_context_length_known(endpoint, "gpt-4o") == (32768, True)

    def test_override_wins_over_live_query(self, monkeypatch):
        def fake_query(endpoint_url, model):
            raise AssertionError("discovery must be skipped when an override is set")

        monkeypatch.setattr(model_context, "_query_context_length", fake_query)
        self._install_ollama_endpoint(monkeypatch, {"qwen3:14b": 65536})
        endpoint = "http://localhost:11434/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "qwen3:14b") == 65536

    def test_no_override_falls_back_to_normal_lookup(self, monkeypatch):
        self._install_ollama_endpoint(monkeypatch, {"qwen3:14b": 65536})

        def fake_query(endpoint_url, model):
            return (131072, True)

        monkeypatch.setattr(model_context, "_query_context_length", fake_query)
        endpoint = "http://localhost:11434/v1/chat/completions"
        # A model without an override still resolves via the normal path.
        assert model_context.get_context_length(endpoint, "llama-3.1") == 131072

    def test_override_matches_prefixed_catalog_id(self, monkeypatch):
        # Overrides keyed on the bare model id still apply when the caller's
        # id carries a provider prefix (or vice versa).
        self._install_ollama_endpoint(monkeypatch, {"some/qwen3:14b": 16384})
        endpoint = "http://localhost:11434/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "qwen3:14b") == 16384

    def test_override_takes_effect_immediately_no_cache(self, monkeypatch):
        # Changing the override must apply on the very next call, without
        # needing to invalidate any cache.
        endpoint = "http://localhost:11434/v1/chat/completions"
        self._install_ollama_endpoint(monkeypatch, {"devstral:24b": 32768})
        assert model_context.get_context_length(endpoint, "devstral:24b") == 32768
        self._install_ollama_endpoint(monkeypatch, {"devstral:24b": 8192})
        assert model_context.get_context_length(endpoint, "devstral:24b") == 8192

    def test_zero_or_missing_override_is_ignored(self, monkeypatch):
        self._install_ollama_endpoint(monkeypatch, {"some-model": 0})

        def fake_query(endpoint_url, model):
            return (model_context.DEFAULT_CONTEXT, False)

        monkeypatch.setattr(model_context, "_query_context_length", fake_query)
        endpoint = "http://localhost:11434/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "some-model") == model_context.DEFAULT_CONTEXT

    def test_malformed_overrides_json_is_ignored(self, monkeypatch):
        _install_endpoint_db(monkeypatch, [
            types.SimpleNamespace(
                base_url="http://localhost:11434",
                endpoint_kind="local",
                api_key=None,
                is_enabled=True,
                model_overrides="not valid json",
            )
        ])

        def fake_query(endpoint_url, model):
            return (model_context.DEFAULT_CONTEXT, False)

        monkeypatch.setattr(model_context, "_query_context_length", fake_query)
        endpoint = "http://localhost:11434/v1/chat/completions"
        assert model_context.get_context_length(endpoint, "some-model") == model_context.DEFAULT_CONTEXT

    def test_generic_lookup_reads_arbitrary_field(self, monkeypatch):
        # Locks in that _configured_model_override is a generic {field: value}
        # reader, not hardcoded to num_ctx — future override fields (e.g.
        # supports_tools, keep_alive) reuse this same lookup unmodified.
        import json
        _install_endpoint_db(monkeypatch, [
            types.SimpleNamespace(
                base_url="http://localhost:11434",
                endpoint_kind="local",
                api_key=None,
                is_enabled=True,
                model_overrides=json.dumps({"some-model": {"num_ctx": 8192, "keep_alive": "30m"}}),
            )
        ])
        endpoint = "http://localhost:11434/v1/chat/completions"
        assert model_context._configured_model_override(endpoint, "some-model", "keep_alive") == "30m"
        assert model_context._configured_model_override(endpoint, "some-model", "num_ctx") == 8192
        assert model_context._configured_model_override(endpoint, "some-model", "nonexistent_field") is None
