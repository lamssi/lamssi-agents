"""Regression tests for defects found during release review."""

from __future__ import annotations

from pathlib import Path

import pytest

from lamssi_agents import (
    Agent,
    ApprovalPolicy,
    Files,
    Guidance,
    LiteLLMModel,
    SystemTools,
)

# configuration precedence


def _agent(tmp_path: Path, *, model=None, max_turns: int = 200) -> Agent:
    return Agent(
        model=model,
        max_turns=max_turns,
        features=[SystemTools(), Guidance(), Files(tmp_path)],
    )


def test_a_tuning_keyword_beats_the_config_object(tmp_path: Path):
    """An explicit constructor value takes precedence over config defaults."""
    agent = _agent(tmp_path, max_turns=30)
    assert agent.max_turns == 30


def test_model_tuning_belongs_to_the_model_adapter(tmp_path: Path):
    model = LiteLLMModel("openai/gpt-5-mini", temperature=0.25, max_tokens=4_000)
    agent = _agent(tmp_path, model=model)
    assert agent.model is model
    assert model.temperature == 0.25
    assert model.max_tokens == 4_000


def test_registry_resolution_normalizes_or_rejects_arguments_once():
    """Preparation is the sole argument-normalization boundary."""
    from lamssi_tools import (
        ToolDefinition,
        ToolExecutionError,
        ToolParameter,
        ToolRegistry,
    )

    definition = ToolDefinition(
        name="t",
        description="",
        parameters=[
            ToolParameter(name="n", type="integer", required=False, description=""),
            ToolParameter(name="f", type="number", required=False, description=""),
            ToolParameter(
                name="b",
                type="boolean",
                required=False,
                default=False,
                description="",
            ),
        ],
    )
    registry = ToolRegistry()
    registry.add_one(definition, lambda **kwargs: kwargs)

    binding = registry.resolve("t", {"n": "3", "f": "1.5", "b": "yes"})
    assert binding.arguments == {"n": 3, "f": 1.5, "b": True}

    with pytest.raises(ToolExecutionError, match="Unknown tool"):
        registry.resolve("other", {"a": 1})

    with pytest.raises(ToolExecutionError, match="Invalid arguments"):
        registry.resolve("t", {"n": "nope"})


# packaging

#: Import name -> the distribution that provides it, for the few that differ.
_DISTRIBUTION = {
    "PIL": "pillow",
    "nptdms": "nptdms",
    "yaml": "pyyaml",
}

#: Declared but never imported by name, with the reason. Listed rather than
#: waved through, so an extra cannot grow a dependency nobody can account for.
_INDIRECT = {
    # pandas selects it as the .xlsx/.xlsm engine; `read_excel` fails without it
    # and no line in this project says its name.
    "openpyxl": "pandas' Excel engine",
}

#: Dependencies used only by runnable examples. Examples are intentionally not
#: package code and therefore are not included in ``_third_party_imports``.
_EXAMPLE_ONLY = {
    "pyside6": "examples/18_pyside6_app.py",
}


def _package_dirs() -> list:
    """Return package roots and fail if the source layout changed."""
    src = Path(__file__).resolve().parent.parent / "src"
    dirs = [src / name for name in ("lamssi_agents", "lamssi_tools")]
    missing = [str(d) for d in dirs if not d.is_dir()]
    assert not missing, f"package tree moved; update _package_dirs(): {missing}"
    return dirs


def _third_party_imports() -> dict:
    """``{module: [where it is imported]}`` for every non-stdlib, non-local import."""
    import ast
    import sys

    root = Path(__file__).resolve().parent.parent
    local = {"lamssi_agents", "lamssi_tools"}
    found: dict = {}
    for package in _package_dirs():
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] if node.level == 0 else []
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    if not top or top in local or top in sys.stdlib_module_names:
                        continue
                    where = f"{path.relative_to(root)}:{node.lineno}"
                    found.setdefault(top, []).append(where)
    return found


def _declared() -> tuple:
    import tomllib

    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    base = {
        r.split(">")[0].split("=")[0].split("[")[0].strip().lower()
        for r in project["dependencies"]
    }
    extras = {
        name: {
            r.split(">")[0].split("=")[0].split("[")[0].strip().lower() for r in reqs
        }
        for name, reqs in project.get("optional-dependencies", {}).items()
    }
    return base, extras


def test_every_third_party_import_is_declared_somewhere():
    """Every third-party import has a base or extra dependency declaration."""
    base, extras = _declared()
    everything = set(base)
    for names in extras.values():
        everything |= names

    missing = {
        module: where
        for module, where in _third_party_imports().items()
        if _DISTRIBUTION.get(module, module).lower() not in everything
    }
    assert not missing, "imported but declared in no dependency group: " + "; ".join(
        f"{m} ({w[0]})" for m, w in sorted(missing.items())
    )


def test_no_extra_installs_something_nothing_imports():
    """Optional extras contain only dependencies used by the package."""
    _, extras = _declared()
    accounted = (
        {_DISTRIBUTION.get(m, m).lower() for m in _third_party_imports()}
        | set(_INDIRECT)
        | set(_EXAMPLE_ONLY)
    )
    unused = {
        name: sorted(names - accounted)
        for name, names in extras.items()
        if names - accounted
    }
    assert not unused, (
        f"declared in an extra but imported nowhere: {unused}. If it is pulled in "
        f"indirectly, say so in _INDIRECT."
    )


def test_pillow_is_optional():
    """Pillow belongs only to the vision extra."""
    base, extras = _declared()
    assert "pillow" not in base
    assert "pillow" in extras["vision"]


# the null host


def test_a_bad_workspace_is_not_reported_as_an_unknown_host(tmp_path: Path):
    """The host resolved fine. The path is wrong, and the exception should say so."""
    from lamssi_cli.hosts import NullHost, UnknownHost

    host = NullHost()
    with pytest.raises(NotADirectoryError):
        host.resolve_root(str(tmp_path / "nowhere"))

    # And specifically not the exception a caller uses to try a different host.
    try:
        host.resolve_root(str(tmp_path / "nowhere"))
    except Exception as exc:
        assert not isinstance(exc, UnknownHost)


def test_the_null_host_still_resolves_a_real_directory(tmp_path: Path):
    from lamssi_cli.hosts import NullHost

    assert NullHost().resolve_root(str(tmp_path)) == tmp_path.resolve()


def test_the_null_host_applies_compaction_config_before_returning(tmp_path: Path):
    """The config and installed strategy must describe the same runtime."""
    from lamssi_agents.history import compress_history
    from lamssi_agents.runtime import AgentConfig
    from lamssi_cli.hosts import NullHost

    config = AgentConfig(
        compaction="ladder",
        history_budget_tokens=12_000,
        keep_recent=8,
        max_tool_result_chars=3_000,
        autocompact_fraction=0.75,
        reserve_tokens=2_000,
    )
    agent = NullHost().create_agent(root=tmp_path, config=config)

    assert agent.compactor is compress_history
    assert agent._config.history_budget_tokens == 12_000
    assert agent._config.keep_recent == 8
    assert agent._config.max_tool_result_chars == 3_000
    assert agent._config.autocompact_fraction == 0.75
    assert agent._config.reserve_tokens == 2_000


def test_the_public_front_door_is_exact_and_resolves():
    """Keep one executable contract for the beginner vocabulary."""
    import lamssi_agents

    expected = {
        "__version__",
        "Agent",
        "Code",
        "ContextBlock",
        "Feature",
        "Files",
        "Guidance",
        "Memory",
        "PromptPosition",
        "RunResult",
        "Shell",
        "Skills",
        "SystemTools",
        "tool",
        "LiteLLMModel",
        "InteractionRequest",
        "InteractionResponse",
        "ApprovalPolicy",
        "ApprovalRequest",
        "ToolApproval",
        "ToolApprovalResult",
    }
    assert set(lamssi_agents.__all__) == expected
    assert not [
        name for name in lamssi_agents.__all__ if not hasattr(lamssi_agents, name)
    ]

    namespace: dict = {}
    exec("from lamssi_agents import *", namespace)
    assert expected <= namespace.keys()


def test_explicit_features_build_a_working_agent(tmp_path):
    agent = Agent(
        features=[SystemTools(), Guidance(), Files(tmp_path)],
        approval=ApprovalPolicy.allow_all(),
    )

    assert {"read_file", "write_file", "fs"} <= {
        t.name for t in agent._tools.list_tools()
    }


def test_the_package_stays_synchronous():
    """The package remains synchronous as documented."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    offenders = []
    for package in _package_dirs():
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)
                ):
                    offenders.append(f"{path.relative_to(root)}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    if any(a.name.split(".")[0] == "asyncio" for a in node.names):
                        offenders.append(f"{path.relative_to(root)}:{node.lineno}")
                elif isinstance(node, ast.ImportFrom):
                    if (node.module or "").split(".")[0] == "asyncio":
                        offenders.append(f"{path.relative_to(root)}:{node.lineno}")

    assert not offenders, (
        "the README says this package contains no asyncio; found: "
        + ", ".join(sorted(set(offenders))[:10])
    )
