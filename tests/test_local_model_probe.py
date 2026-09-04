"""Context-window discovery from LM Studio and Ollama responses."""

from __future__ import annotations

import pytest

from lamssi_agents.providers import local_models

# the responses, as the servers actually send them

LM_STUDIO = {
    "data": [
        {
            "id": "google/gemma-4-12b",
            "type": "vlm",
            "arch": "gemma4",
            "state": "loaded",
            "max_context_length": 262144,
            "loaded_context_length": 262144,
            "capabilities": ["tool_use"],
        },
        {
            "id": "qwen/qwen3.5-9b",
            "state": "not-loaded",
            "max_context_length": 262144,
            "capabilities": ["tool_use"],
        },
        {"id": "meta/muse-glimmer", "state": "not-loaded", "max_context_length": 131072},
    ]
}

#: Ollama metadata without a declared ``num_ctx`` value.
OLLAMA_SHOW = {
    "capabilities": ["completion", "tools", "thinking"],
    "parameters": 'stop                           "<turn|>"',
    "model_info": {"general.architecture": "gemma4", "gemma4.context_length": 262144},
}

#: Resident model metadata with a 4,096-token window.
OLLAMA_PS = {
    "models": [{
        "name": "gemma-4-12b:latest",
        "model": "gemma-4-12b:latest",
        "size_vram": 8001610055,
        "context_length": 4096,
    }]
}

@pytest.fixture
def server(monkeypatch):
    """Route the probe's HTTP at a dict of ``url -> body``; anything else 'refuses'."""
    routes: dict = {}

    def fake_get(url, payload=None):
        if url not in routes:
            raise OSError("connection refused")
        return routes[url]

    monkeypatch.setattr(local_models, "_get_json", fake_get)
    return routes

# LM Studio

def test_the_loaded_window_is_read_from_the_server(server):
    server["http://127.0.0.1:1234/api/v0/models"] = LM_STUDIO
    info = local_models.probe("http://127.0.0.1:1234/v1", "openai/google/gemma-4-12b")
    assert info.context_window == 262_144, (
        "LM Studio published the window and it was not used; the caller falls back "
        "to 32,000 and compacts a 262k model at 27k"
    )
    assert info.supports_tools is True
    assert info.source == "LM Studio"

def test_the_loaded_length_wins_over_the_architectural_maximum(server):
    """A 262k model loaded at 32k rejects the 33rd thousand token regardless."""
    server["http://127.0.0.1:1234/api/v0/models"] = {
        "data": [{
            "id": "m", "state": "loaded",
            "max_context_length": 262144,
            "loaded_context_length": 32768,
        }]
    }
    info = local_models.probe("http://127.0.0.1:1234/v1", "m")
    assert info.context_window == 32_768, (
        "took the architecture's maximum over what this instance was loaded with: "
        "the request will be refused long before the reported window is reached"
    )

def test_a_model_not_yet_loaded_reports_its_maximum(server):
    server["http://127.0.0.1:1234/api/v0/models"] = LM_STUDIO
    info = local_models.probe("http://127.0.0.1:1234/v1", "qwen/qwen3.5-9b")
    assert info.context_window == 262_144

def test_a_server_that_omits_capabilities_says_nothing_about_tools(server):
    """``None`` is 'did not say', which is not the same claim as ``False``."""
    server["http://127.0.0.1:1234/api/v0/models"] = LM_STUDIO
    info = local_models.probe("http://127.0.0.1:1234/v1", "meta/muse-glimmer")
    assert info.supports_tools is None

# Ollama

def test_the_resident_window_is_read_not_the_architectures_ceiling(server):
    """Ollama probing reports the resident window, not the model ceiling."""
    server["http://127.0.0.1:11434/api/show"] = OLLAMA_SHOW
    server["http://127.0.0.1:11434/api/ps"] = OLLAMA_PS
    info = local_models.probe("http://127.0.0.1:11434/v1", "ollama/gemma-4-12b:latest")
    assert info.context_window == 4_096, (
        "read gemma4.context_length (262144) rather than the window the instance "
        "was loaded with (4096)"
    )
    assert info.supports_tools is True
    assert info.source == "Ollama"

def test_the_tag_is_optional_in_the_name_asked_for(server):
    server["http://127.0.0.1:11434/api/show"] = OLLAMA_SHOW
    server["http://127.0.0.1:11434/api/ps"] = OLLAMA_PS
    info = local_models.probe("http://127.0.0.1:11434/v1", "ollama/gemma-4-12b")
    assert info.context_window == 4_096

def test_the_resident_window_beats_a_modelfile_parameter(server):
    """A Modelfile num_ctx is what it asked for; /api/ps is what it got."""
    server["http://127.0.0.1:11434/api/show"] = {
        **OLLAMA_SHOW, "parameters": "num_ctx                        32768",
    }
    server["http://127.0.0.1:11434/api/ps"] = OLLAMA_PS
    assert local_models.probe(
        "http://127.0.0.1:11434/v1", "gemma-4-12b:latest"
    ).context_window == 4_096

def test_a_modelfile_parameter_is_used_when_nothing_is_resident(server):
    """Use a declared ``num_ctx`` when no resident measurement exists."""
    server["http://127.0.0.1:11434/api/show"] = {
        **OLLAMA_SHOW, "parameters": "num_ctx                        32768",
    }
    server["http://127.0.0.1:11434/api/ps"] = {"models": []}
    assert local_models.probe(
        "http://127.0.0.1:11434/v1", "gemma-4-12b:latest"
    ).context_window == 32_768

def test_a_model_that_is_not_resident_reports_no_window_at_all(server):
    """A non-resident Ollama model reports an unknown context window."""
    server["http://127.0.0.1:11434/api/show"] = OLLAMA_SHOW
    server["http://127.0.0.1:11434/api/ps"] = {"models": []}
    info = local_models.probe("http://127.0.0.1:11434/v1", "gemma-4-12b:latest")
    assert info.context_window == 0
    assert info.supports_tools is True, "the window is unknown; tool support is not"

def test_a_default_sized_window_is_called_out(server, caplog):
    import logging

    server["http://127.0.0.1:11434/api/show"] = OLLAMA_SHOW
    server["http://127.0.0.1:11434/api/ps"] = OLLAMA_PS
    with caplog.at_level(logging.WARNING):
        local_models.probe("http://127.0.0.1:11434/v1", "gemma-4-12b:latest")
    assert any("OLLAMA_CONTEXT_LENGTH" in r.getMessage() for r in caplog.records)

def test_a_deliberately_large_window_is_not_nagged_about(server, caplog):
    import logging

    server["http://127.0.0.1:11434/api/show"] = OLLAMA_SHOW
    server["http://127.0.0.1:11434/api/ps"] = {
        "models": [{"model": "gemma-4-12b:latest", "context_length": 65536}]
    }
    with caplog.at_level(logging.WARNING):
        local_models.probe("http://127.0.0.1:11434/v1", "gemma-4-12b:latest")
    assert not caplog.records, "65,536 of a possible 262,144 is a choice, not a default"

def test_ps_not_answering_does_not_cost_us_the_rest(server):
    """Older servers, or one that drops the call: still report tool support."""
    server["http://127.0.0.1:11434/api/show"] = OLLAMA_SHOW
    info = local_models.probe("http://127.0.0.1:11434/v1", "gemma-4-12b:latest")
    assert info.context_window == 0
    assert info.supports_tools is True

# refusing to answer

def test_a_remote_base_is_never_probed(server):
    """Reaching into a private API on someone else's host, unasked, is not ours to do."""
    server["https://api.example.com/api/v0/models"] = LM_STUDIO
    assert not local_models.probe("https://api.example.com/v1", "google/gemma-4-12b")
    assert local_models.is_local_base("https://api.example.com/v1") is False


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost.evil.example/v1",
        "https://127.0.0.1.evil.example/v1",
        "https://example.com/localhost/v1",
    ],
)
def test_local_detection_matches_the_hostname_not_a_substring(url):
    assert local_models.is_local_base(url) is False

def test_no_server_answering_is_not_an_error(server):
    assert not local_models.probe("http://127.0.0.1:1234/v1", "anything")

def test_a_shape_we_do_not_recognise_is_not_an_error(server):
    for body in ({"data": "not a list"}, {"data": [{"id": "m"}]}, None, [1, 2, 3]):
        server["http://127.0.0.1:1234/api/v0/models"] = body
        assert local_models.probe("http://127.0.0.1:1234/v1", "m").context_window == 0

def test_a_nonsense_window_is_ignored(server):
    for value in (0, -1, "big", None, 3.7e400):
        server["http://127.0.0.1:1234/api/v0/models"] = {
            "data": [{"id": "m", "state": "loaded", "loaded_context_length": value}]
        }
        assert local_models.probe("http://127.0.0.1:1234/v1", "m").context_window == 0

# name matching

@pytest.mark.parametrize("asked", [
    "google/gemma-4-12b",
    "openai/google/gemma-4-12b",   # the routing prefix the factory adds
    "local/google/gemma-4-12b",
])
def test_the_routing_prefix_does_not_hide_the_model(server, asked):
    server["http://127.0.0.1:1234/api/v0/models"] = LM_STUDIO
    assert local_models.probe("http://127.0.0.1:1234/v1", asked).context_window == 262_144

def test_an_exact_id_is_preferred_over_a_matching_tail(server):
    server["http://127.0.0.1:1234/api/v0/models"] = {
        "data": [
            {"id": "other/gemma-4-12b", "state": "loaded", "loaded_context_length": 4096},
            {"id": "google/gemma-4-12b", "state": "loaded", "loaded_context_length": 262144},
        ]
    }
    info = local_models.probe("http://127.0.0.1:1234/v1", "google/gemma-4-12b")
    assert info.context_window == 262_144, (
        "matched a different publisher's model of the same name; the window would "
        "describe a model that is not the one being called"
    )

def test_a_base_without_the_v1_suffix_still_finds_the_native_api(server):
    server["http://127.0.0.1:1234/api/v0/models"] = LM_STUDIO
    assert local_models.probe(
        "http://127.0.0.1:1234", "google/gemma-4-12b"
    ).context_window == 262_144


# Gateway-prefixed model lookup


class _FakeLiteLLM:
    """Stands in for LiteLLM's model DB: knows vendor-qualified names only."""

    DB = {
        "openai/gpt-5.6-luna": {"max_input_tokens": 1_050_000},
        "anthropic/claude-sonnet-4-5": {"max_input_tokens": 200_000},
        "llama3": {"max_tokens": 8192},
    }

    def get_model_info(self, name):
        if name not in self.DB:
            raise Exception(f"model {name!r} not in the DB")
        return self.DB[name]


def _detect(model):
    from lamssi_agents.providers.litellm_provider import LiteLLMModel

    return LiteLLMModel._detect_context_window(model, _FakeLiteLLM())


def test_a_gateway_prefix_is_peeled_until_the_db_answers():
    assert _detect("openrouter/openai/gpt-5.6-luna") == (1_050_000, "openai/gpt-5.6-luna")


def test_peeling_is_not_limited_to_a_known_list_of_gateways():
    """Recognize provider families behind arbitrary gateway prefixes."""
    assert _detect("some-future-gateway/anthropic/claude-sonnet-4-5") == (
        200_000, "anthropic/claude-sonnet-4-5",
    )


def test_a_name_the_db_knows_outright_is_not_peeled():
    assert _detect("openai/gpt-5.6-luna") == (1_050_000, "openai/gpt-5.6-luna")


def test_peeling_reaches_a_bare_name():
    assert _detect("ollama/llama3") == (8192, "llama3")


def test_a_genuinely_unknown_model_still_reports_nothing():
    """Return no inferred context window when the server provides none."""
    assert _detect("local/some-quantised-thing") == (0, "")


# the window believes the provider over its own guess


class _Widening:
    """Minimal provider surface for the ratchet: a window and a usage recorder."""

    def __init__(self, window):
        from lamssi_agents.providers.litellm_provider import LiteLLMModel

        self._context_window = window
        self._window_widened = False
        self.model = "gateway/vendor/unknown-model"
        self._widen = LiteLLMModel._widen_window_if_exceeded.__get__(self)


def test_an_accepted_request_above_the_window_widens_it():
    """Accepted provider usage widens an underestimated context window."""
    p = _Widening(32_000)
    p._widen(32_668)
    assert p._context_window == 32_668


def test_widening_only_ever_grows():
    p = _Widening(200_000)
    p._widen(9_000)
    assert p._context_window == 200_000, "a small request says nothing about the limit"


def test_widening_announces_itself_once(caplog):
    import logging

    p = _Widening(32_000)
    with caplog.at_level(logging.WARNING):
        p._widen(40_000)
        p._widen(50_000)
    assert sum("widening" in r.message for r in caplog.records) == 1
