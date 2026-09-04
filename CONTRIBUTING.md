# Contributing

Thanks for contributing. The checks below cover the project-specific contracts.

## Running the checks

```bash
uv sync --group dev
uv run pytest
uv run ruff check --select F,E9,B src tests examples
uv run lint-imports
uv run pytest -m slow
```

CI runs the source checks on Linux, Windows and macOS, and the wheel test on Linux.

## Layout

The packages live under `src/`. `uv sync` creates the editable install used by
local development and tests.

## The invariants

**Host independence.** The kernel contains no UI framework, hardware, plugin
layout, or application directory convention. Host contracts live beside their
feature, and tool capabilities live in `lamssi_tools`. Import contracts enforce
this dependency direction.

**Prompt data is an import leaf.** Prompt contracts cannot depend on the runtime
that consumes them.

**Synchronous runtime.** `agent.run()` and `agent.chat()` block. Applications use
threads and a dispatcher when they need concurrency or thread-affine tools.

**A base install works.** Optional functionality is lazy-imported behind extras.
Dependency tests check that imports and declared extras agree.

**Examples run in CI.** Each example must exit successfully, print portable ASCII,
carry an SPDX line, appear in `examples/README.md`, and leave no files behind.

## Tests

Tests should protect public behavior or a concrete runtime invariant. For a bug
fix, confirm that the new test fails before the implementation changes.

## Reporting something

Include the traceback, command, operating system, and Python version. Model issues
also need the model identifier, adapter construction, and endpoint. LiteLLM
transport settings belong to `LiteLLMModel`.

## Licence

MIT. By contributing you agree your contribution is licensed under it.
