"""Model discovery through ``LiteLLMModel.available_models()``."""

from __future__ import annotations

import pytest

from lamssi_agents import LiteLLMModel
from lamssi_agents.providers import model_catalog as mc
from lamssi_agents.providers.model_catalog import (
    fetch_anthropic,
    fetch_local,
    fetch_openai,
    fetch_openrouter,
    fetch_unlisted,
    models_endpoint,
    source_for_model,
)

@pytest.fixture(autouse=True)
def forget_test_keys():
    """Clear process-wide secret registrations after each test."""
    from lamssi_agents.redaction import forget_secrets

    yield
    forget_secrets()

@pytest.fixture
def no_network(monkeypatch):
    """Reject unexpected network access in catalog tests."""
    def explode(*a, **kw):
        raise AssertionError("test attempted a live HTTP call")

    monkeypatch.setattr(mc, "http_get_json", explode)

# the thing that was asked for

def test_a_model_adapter_can_list_its_own_models(no_network):
    """The headline: no config plumbing, no second object to construct."""
    model = LiteLLMModel("claude-sonnet-4-5", api_key="")

    models = model.available_models()

    assert isinstance(models, list)
    assert "claude-sonnet-4-5" in models

def test_the_label_says_whether_the_list_is_real():
    """Catalog labels distinguish live results from fallbacks."""
    model = LiteLLMModel("claude-sonnet-4-5", api_key="")

    _, label = model.model_catalog()

    assert "fallback" in label.lower()
    assert "no API key" in label

def test_constructing_a_model_adapter_does_not_call_the_network(no_network):
    """Model-adapter construction does not fetch a catalog."""
    LiteLLMModel("gpt-4o", api_key="sk-something-longer-than-nothing")

def test_the_answer_is_remembered(monkeypatch):
    """The obvious caller is a dropdown that reopens."""
    calls = []

    def counted(url, headers, timeout=5.0):
        calls.append(url)
        return {"data": [{"id": "claude-x"}]}

    monkeypatch.setattr(mc, "http_get_json", counted)
    model = LiteLLMModel("claude-sonnet-4-5", api_key="sk-ant-testkeytestkey1234")

    model.available_models()
    model.available_models()

    assert len(calls) == 1

def test_refresh_asks_again(monkeypatch):
    """Bypass a cached catalog when ``refresh=True``."""
    calls = []

    def counted(url, headers, timeout=5.0):
        calls.append(url)
        return {"data": [{"id": "claude-x"}]}

    monkeypatch.setattr(mc, "http_get_json", counted)
    model = LiteLLMModel("claude-sonnet-4-5", api_key="sk-ant-testkeytestkey1234")

    model.available_models()
    model.available_models(refresh=True)

    assert len(calls) == 2

# which source a configured backend gets

@pytest.mark.parametrize("model, expected", [
    ("claude-sonnet-4-5", fetch_anthropic),
    ("claude-3-5-haiku-20241022", fetch_anthropic),
    ("anthropic/claude-sonnet-4-5", fetch_anthropic),
    ("gpt-4o", fetch_openai),
    ("o1-mini", fetch_openai),
    ("chatgpt-4o-latest", fetch_openai),
    ("openrouter/anthropic/claude-3.5-sonnet", fetch_openrouter),
    ("gemini/gemini-2.0-flash", fetch_unlisted),
    ("deepseek/deepseek-chat", fetch_unlisted),
])
def test_a_model_id_picks_its_catalogue(model, expected):
    assert source_for_model(model).func is expected

def test_openrouter_is_matched_before_the_families_inside_it():
    """OpenRouter routing wins over vendor substrings in its model id."""
    assert source_for_model("openrouter/anthropic/claude-3.5-sonnet").func is fetch_openrouter

def test_a_configured_endpoint_beats_the_model_name():
    """A configured local endpoint takes precedence over model-name routing."""
    picked = source_for_model("openai/qwen2.5-14b", api_base="http://127.0.0.1:1234/v1")

    assert picked.func is fetch_local

def test_a_backend_with_no_catalogue_says_so_instead_of_guessing():
    """Backends without catalogs return an explicit empty result."""
    models, label = source_for_model("gemini/gemini-2.0-flash")()

    assert models == []
    assert "gemini" in label
    assert "no catalogue" in label

# the endpoint both callers build

@pytest.mark.parametrize("base", [
    "http://127.0.0.1:1234",
    "http://127.0.0.1:1234/",
    "http://127.0.0.1:1234/v1",
    "http://127.0.0.1:1234/v1/",
])
def test_the_models_url_has_exactly_one_v1(base):
    """The models endpoint contains exactly one ``/v1`` segment."""
    assert models_endpoint(base) == "http://127.0.0.1:1234/v1/models"

def test_the_local_source_uses_it(monkeypatch):
    """Asserted on the URL actually requested, not on the helper in isolation."""
    seen = {}

    def capture(url, headers, timeout=5.0):
        seen["url"] = url
        return {"data": [{"id": "qwen"}]}

    monkeypatch.setattr(mc, "http_get_json", capture)
    fetch_local("http://127.0.0.1:1234/v1")

    assert seen["url"] == "http://127.0.0.1:1234/v1/models"

# parsing a live answer

def test_anthropic_ids_are_extracted_and_sorted(monkeypatch):
    monkeypatch.setattr(mc, "http_get_json", lambda *a, **k: {
        "data": [{"id": "claude-sonnet-4-5"}, {"id": "claude-opus-4-8"}]
    })

    models, label = fetch_anthropic("sk-ant-testkeytestkey1234")

    assert models == ["claude-opus-4-8", "claude-sonnet-4-5"]
    assert "key OK" in label

def test_openai_drops_the_non_chat_models(monkeypatch):
    """The OpenAI catalog excludes non-chat model families."""
    monkeypatch.setattr(mc, "http_get_json", lambda *a, **k: {
        "data": [
            {"id": "gpt-4o"}, {"id": "text-embedding-3-small"},
            {"id": "whisper-1"}, {"id": "o1-mini"}, {"id": "dall-e-3"},
        ]
    })

    models, _ = fetch_openai("sk-testkeytestkey1234")

    assert models == ["gpt-4o", "o1-mini"]

def test_openrouter_ids_get_the_prefix_litellm_needs(monkeypatch):
    """OpenRouter catalog ids include the LiteLLM routing prefix."""
    monkeypatch.setattr(mc, "http_get_json", lambda *a, **k: {
        "data": [{"id": "anthropic/claude-3.5-sonnet"}]
    })

    models, _ = fetch_openrouter("")

    assert models == ["openrouter/anthropic/claude-3.5-sonnet"]

def test_a_rejected_key_is_reported_as_such(monkeypatch):
    """An HTTP 401 is reported as credential rejection."""
    def unauthorized(*a, **k):
        raise RuntimeError("HTTP 401: invalid x-api-key")

    monkeypatch.setattr(mc, "http_get_json", unauthorized)

    models, label = fetch_anthropic("sk-ant-wrongkeywrongkey123")

    assert models == mc.ANTHROPIC_FALLBACK
    assert "invalid API key" in label

def test_a_model_adapter_survives_an_unreachable_backend(monkeypatch):
    """Keep fallback models available while the backend is unreachable."""
    def down(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(mc, "http_get_json", down)
    model = LiteLLMModel("claude-sonnet-4-5", api_key="sk-ant-testkeytestkey1234")

    assert model.available_models() == mc.ANTHROPIC_FALLBACK

# the protocol

def test_the_model_contract_is_small_but_litellm_offers_a_catalog():
    """Catalog discovery is a LiteLLM extension, not a custom-model requirement."""
    from lamssi_agents.providers.protocol import Model

    assert hasattr(Model, "stream")
    assert hasattr(LiteLLMModel("gpt-4o", api_key=""), "available_models")
