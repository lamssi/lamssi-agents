# Capability seams

A bare `Agent()` installs no model, tools, prompt guidance, files, memory, or host
integration. The public design has four extension mechanisms.

| Need | Public seam | Ownership |
| --- | --- | --- |
| Select or replace inference | `Agent(model=...)`, `agent.use_model(...)` | model adapter |
| Add a related behavior bundle | `Feature.install(agent)` | feature |
| Supply a live application object to tools | `capabilities={Type: value}`, `agent.provide(...)` | host application |
| Add dynamic system context | `ContextBlock`, `agent.add_context(...)` | host or feature |

## Models

`model` has one shape: a LiteLLM identifier string or a custom `Model` adapter.
A string becomes `LiteLLMModel`; an object is used directly. The minimal custom
adapter supplies `model` and `stream(...)`. Diagnostic metadata, connectivity,
usage, context-window size, and reasoning controls are optional capabilities.

## Features

`Feature` is the only behavior bundle. `install(agent)` may call narrow registration
methods:

- `add_tools(...)`
- `provide(Type, value)`
- `add_context(block)`
- `set_tool_dispatcher(dispatcher)` for GUI/thread-affine tools

The optional `before_turn`, `before_tool`, `after_tool`, and `on_event` methods are
registered automatically. The registry, hook lists, dispatcher executor, prompt
registry, and run authority remain private.

Each built-in feature owns its integration state:

| Feature | Owns |
| --- | --- |
| `Files(root, read_only, on_write, protected_paths)` | workspace and file policy |
| `Memory(path)` | persistent notes location |
| `Shell()` | shell tool; uses `FileSpace` when installed |
| `Code(executor)` | optional code executor capability |
| `Skills(*roots, allow_model_loading=False)` | skill catalogue and optional loading tool |
| `Budget(every_tokens)` | typed budget checkpoints |
| `SystemTools()` / `Guidance()` | system tools / optional operating advice |

## Typed capabilities

Capabilities are live application objects resolved by type through
`lamssi_tools.CapabilityContext`. Examples include `FileSpace`, `CodeExecutor`,
`MemoryStore`, `SkillRuntime`, and application-defined protocols such as a device
catalog.

Framework contracts live with the feature that consumes them:

- `lamssi_agents.features.code`: `CodeExecutor`, `CodeResult`
- `lamssi_agents.features.system`: `AbortSink`
- `lamssi_agents.features.files`: `WriteEvent`, `WriteHook`, `WriteKind`
- `lamssi_agents.features.skills`: `Skills`, `SkillRuntime`, `Skill`
- `lamssi_agents.prompt`: `PromptContext`, `PromptSection`, prompt ordering
- `lamssi_agents.tooling`: `ToolInvocation`

## Run authority

Approval, interaction, cancellation, events, and disabled-tool state belong to private
run authority. Public operations are narrow: `agent.approval`, `agent.interaction`,
`agent.abort()`, event listeners, and tool enable/disable methods.

## Deliberately fixed behavior

Tool scope is checked before policy hooks. Every assistant tool call receives a
matching tool result, including blocked calls. `run()` is synchronous and returns a
`RunResult`. These invariants are not replaceable plugins.
