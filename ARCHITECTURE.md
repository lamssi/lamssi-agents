# lamssi-agents architecture

lamssi-agents is a synchronous, in-process runtime embedded inside an
application. The application owns UI, storage choices, credentials, device safety,
and thread infrastructure. Lamssi owns the model/tool loop and its invariants.

## Public composition

`Agent` is the only composition object:

```python
agent = Agent(
    model="openai/gpt-5-mini",
    instructions="The complete base system prompt.",
    tools=[...],
    capabilities={DeviceBus: bus},
    context=[...],
    features=[SystemTools(), Guidance(), Files(".")],
    approval=ApprovalPolicy.ask_when_required(approve),
    interaction=interact,
)
```

The constructor must stay explainable:

- `model` is one string-or-`Model` seam. A string creates `LiteLLMModel`; a custom
  adapter is used directly.
- `instructions` is the entire base prompt. Nothing is found by filename and no
  implicit prompt is appended.
- `tools`, `capabilities`, `context`, and `features` are explicit contributions.
- approval and typed interaction are separate decisions.
- common loop policy uses direct arguments such as `max_turns`, or a bundled `config=AgentConfig`.
- `tool_sources=` mounts live external registries a host owns. It is advanced and
  keyword-only; beginners never name it, and the default path stays `Agent("model")`.

Do not add a `provider=`, `host=`, raw registry, dispatcher, or arbitrary new keyword path
to this constructor. The advanced keyword-only seams already present (`tool_sources=`,
`safe_when=`, `config=`, `only=`) are the ceiling.

## Public versus advanced

The beginner surface lives in `lamssi_agents.__all__` and stays small.
Advanced types live in focused modules:

- `lamssi_agents.model`: `Model`, `ModelInput`, model factories.
- `lamssi_agents.interaction`: interaction kinds, decisions, and handler type.
- `lamssi_agents.events`: events and event kinds.
- `lamssi_agents.prompt`: prompt contracts and composition.
- `lamssi_agents.features.<name>`: per-feature framework contracts (`CodeExecutor`,
  `AbortSink`, `WriteEvent`, `SkillRuntime`), catalogued in
  [docs/capability-seams.md](docs/capability-seams.md).
- `lamssi_agents.tooling`: tool invocation and policy types.
- `lamssi_agents.history`: compaction/truncation strategies.

Raw run control, conversation, tool runtime, registry, prompt sections, capability
context, and hook lists are private Agent state. Add a narrow operation when a real
integration need appears; do not expose the owning object. Existing narrow methods
include `add_tools`, `available_tool_names`, `provide`, `get`, `add_context`,
`set_tool_dispatcher`, `history`, `conversation_state`, and event/tool controls.

## Ownership

An Agent has three lifetimes:

1. Composition: tool registry, capabilities, prompt sections, policies, strategies,
   and installed feature instances.
2. Run authority: cancellation, events, approval, typed interaction, and disabled
   tools, owned by the agent's `RunControl`.
3. Conversation: messages, token calibration, and feature state. Tool-safety state
   (dedupe cache, loop guard) belongs to ToolRuntime on the same lifetime.

## Model seam

The agent loop requires a custom model adapter with a model identifier and
`stream(messages, tools, *, abort_event=...)`. Optional metadata such as adapter
name, context window, usage, locality, connectivity, and reasoning controls
degrades safely when absent.

`LiteLLMModel` owns credentials, endpoint, temperature, output limit, retry policy,
reasoning effort, usage, and context-window information. Agent call sites do not
carry those settings separately.

Cancellation is call-scoped: every stream call receives the active run's
event. Never store a newly attached Agent's abort event globally on a shared model
adapter; two agents may intentionally share one adapter.

## Turn and tool invariants

One turn makes exactly one `model.stream(...)` call. Tool calls in one model message
run synchronously and in order.

Every announced tool call receives exactly one matching tool result, including an
unknown, disabled, blocked, rejected, failed, or aborted call. A model endpoint
rejects orphan call/result histories, making this a structural requirement.

The gate order is:

```text
scope -> dedupe -> feature gates -> loop guard -> approval -> tool body
```

Feature `before_tool` gates run before the guard and approval, so a host can refuse a
call the guard or the user would otherwise have been asked about. Scope is fixed before
policy hooks. `ApprovalPolicy.allow_all()` skips generic user consent; it does not bypass
scope, loop guards, feature policy, or application/device safety.

Secrets are redacted before entering history. Shell tools build child environments
through `safe_environ`; do not move credential filtering into prompt instructions.

## Context control

Oversize tool results are capped when created, and ordinary compaction fires at about
0.85 times the model's input window. Before every provider call, request fitting includes
the system prompt, dynamic blocks, tool schemas, and history.

The default `"summarise"` strategy replaces one old prefix and retains as much recent
history as fits, cutting only at a safe turn boundary. It does not stub tool results. The
advanced `"ladder"` strategy may progressively demote tool results before summarising,
including its latest-result emergency step.

Request fitting invokes exactly the configured strategy once. If the result remains over
a known window, it is rejected locally. A provider-reported overflow gets one forced pass
through that same strategy and one retry, only when the request became smaller. Custom
compactors follow the same rule.

## Features

`Feature.install(agent)` is the only behavior-bundle seam. It may call narrow Agent
methods to add tools, capabilities, dynamic context, or a tool dispatcher. Optional
`before_turn`, `before_tool`, `after_tool`, and `on_event` methods are registered by
`Agent.use` in feature-list order.

Per-conversation state belongs in `agent.conversation_state(Key, factory)`, never on the
feature object.

Each built-in feature owns its own integration state; the full ownership table is in
[docs/capability-seams.md](docs/capability-seams.md). Architecturally, `Skills` keeps its
discovery, pins, prompt sections, and per-agent `SkillRuntime` entirely private: the Agent
and the generic prompt sections know none of it.

## Prompts

The fixed base is `instructions`. File-backed instructions are loaded by normal
application code and passed as text.

Use `ContextBlock` only for named dynamic state; its callback signature and closure
capture are covered in [docs/hosting.md](docs/hosting.md) ("Dynamic prompt context").
`explain_prompt()` and assembled prompt parts preserve provenance.

Internal skill assembly state must not widen public `PromptContext`.

## Host and GUI integration

The host supplies ordinary values and focused features. The host and GUI wiring (dispatch
tags, one synchronous dispatcher via `agent.set_tool_dispatcher(...)`, and an
`ApplicationAssistant` facade so widgets never receive a raw Agent) is described in
[docs/hosting.md](docs/hosting.md) ("GUI thread dispatch"). One invariant holds regardless:
the registry binds the active run context before the dispatcher receives the callable, and
the dispatcher must not return until the target call finishes.

Domain safety is separate from Lamssi tool approval. A device governor remains in the
application feature/tool path and cannot be bypassed by a permissive approval policy.

## Source map

```text
src/
  lamssi_tools/            declarations, schemas, registry, context, dispatch
  lamssi_agents/
    agent/                 loop and private runtime ownership
    features/              explicit optional behavior
    prompt/                prompt assembly internals
    history/               compaction, demotion, truncation, token calibration
    providers/             transport internals and LiteLLM adapter
    runtime/               run configuration and capability scope
    tooling/               tool invocation, dedupe, and loop guard
  lamssi_cli/              terminal host and host discovery
  lamssi_packages/         optional packages built on the kernel
```

Prompt data contracts stay an import leaf. Built-in tool handlers are imported
lazily so a bare `import lamssi_agents` remains inert.

## Verification

Before handing off an architectural change:

```bash
python -m pytest tests -q
python -m ruff check src tests examples --select F,E9,B
python -m compileall -q src examples tests
```

The slow wheel test proves a clean installed artifact can import and execute a real
tool turn. The environment-scoping tests execute real shells and may require normal
OS process permissions unavailable in a restricted sandbox.
