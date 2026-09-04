# Hosting lamssi-agents

A host constructs one `Agent` from explicit application values. Settings, prompts,
and directories are supplied at the construction site.

## A small host

```python
from pathlib import Path

from lamssi_agents import Agent, Files, Guidance, Memory, SystemTools


def create_agent(project: Path, settings: dict) -> Agent:
    return Agent(
        model=settings["model"],
        instructions=(project / "prompts" / "assistant.md").read_text(),
        max_turns=int(settings.get("max_turns", 200)),
        features=[
            SystemTools(),
            Guidance(),
            Files(
                project,
                protected_paths=(".git", ".myapp"),
                on_write=(record_write,),
            ),
            Memory(project / ".myapp" / "memory"),
        ],
        log_dir=project / ".myapp" / "conversations",
    )
```

The application owns configuration loading. `Files` owns workspace policy,
`Memory` owns its storage location, and the model adapter owns model-call settings.

## Application capabilities

Tools request live application objects by protocol type:

```python
agent = Agent(
    model,
    capabilities={InstrumentCatalog: instruments},
    tools=[list_instruments, move_instrument],
)
```

The host can also use `agent.provide(Protocol, implementation)` after construction.
`lamssi_tools.CapabilityContext` resolves capabilities inside tool bodies.

## Live application tools

Mount an application-owned `ToolRegistry` to expose its tools live:

```python
agent.mount_tools(coordinator.tool_registry)

# App loading changes coordinator.tool_registry here.
# The Agent sees the new surface on its next model turn.

agent.unmount_tools(coordinator.tool_registry)
```

The external registry remains application-owned. Its dispatcher, validation, owner-based
reload, and listeners continue to work. Lamssi reads its definitions live and applies
Agent scope, guards, dedupe, and approval before handing execution back to it.
Unmounting is by object identity and does not remove any external tools. A duplicate name
across registries is a configuration error.

Capabilities do not cross a mount. A local tool resolves against the agent's
`CapabilityContext`; a mounted tool resolves against its own registry's, attached at the
source (`ModuleSource(module, context=...)`). An external pack sees only what its own
registry provides, never the agent's capabilities.

## Typed interaction and approval

Questions and consent are separate channels:

```python
agent = Agent(
    model,
    interaction=handle_interaction,
    approval=ApprovalPolicy.ask_when_required(handle_approval),
)
```

`handle_interaction` receives an immutable `InteractionRequest` for a question,
guard override, or budget checkpoint and returns `InteractionResponse`. Approval
receives `ApprovalRequest` and returns `ToolApproval`. Qt bridges can correlate
interaction events by `request_id`.

## Dynamic prompt context

Most hosts need only `instructions`. Live application state is an explicit,
advanced block:

```python
from lamssi_agents import ContextBlock

agent.add_context(
    ContextBlock("connected-devices", lambda: render_devices(device_bus))
)
```

If the callable accepts an argument, it receives only `PromptContext(model_id,
tools)`. Capture application objects in the closure; prompt callbacks cannot reach
the run authority or a host bag.

## GUI thread dispatch

A feature can install application tools and one synchronous dispatcher without
exposing the tool registry:

```python
class Instruments(Feature):
    def install(self, agent):
        agent.provide(InstrumentCatalog, self.catalog)
        agent.add_tools(list_instruments, move_instrument)
        agent.set_tool_dispatcher(self.dispatcher)
```

The dispatcher returns each result after the target thread completes. Preserve
Lamssi's active run context when crossing a raw thread boundary.

For a large application, expose an application-owned facade such as
`ApplicationAssistant`; keep the raw `Agent`, event translation, and Lamssi value types
inside that adapter.
