# Embedding lamssi-agents

`Agent` is the application object. It owns the model/tool loop, private run state,
conversation history, and the optional features installed on it.

```python
from lamssi_agents import Agent

agent = Agent("openai/gpt-5-mini")
print(agent.chat("Hello"))
```

A bare agent has no tools and no access to files, memory, or skills.

## Add only the features you want

```python
from lamssi_agents import Agent, ApprovalPolicy
from lamssi_agents import Files, Guidance, Memory, Skills, SystemTools

agent = Agent(
    "openai/gpt-5-mini",
    features=[
        SystemTools(),
        Guidance(),
        Files(".", read_only=[("../reference", "Project reference")]),
        Memory(),
        Skills("./skills"),
    ],
    approval=ApprovalPolicy.reject_when_required(),
)
```

- `SystemTools` provides `ask_user`; `Guidance` adds Lamssi's optional operating advice.
- `Files` provides sandboxed file tools and owns workspace policy. `Shell` and
  `Code` are separate features, so filesystem access does not imply process execution.
- `Memory` provides persistent notes at its explicit path (relative to `Files` by default).
- `Skills` loads skill descriptions and provides a per-agent
  `lamssi_agents.features.skills.SkillRuntime`. Hosts can always pin through the
  runtime; `Skills(..., allow_model_loading=True)` additionally exposes
  `load_skill` to the model.
## Add application tools

`lamssi_tools` remains the tool layer. Decorators, schemas, dispatch tags,
registries, and typed capabilities stay in that lower-level package.

```python
from lamssi_agents import Agent, tool
from lamssi_tools import CapabilityContext, Expose, Str


class Documents:
    def title(self, document_id: str) -> str: ...


@tool(
    expose=Expose.AGENT,
    inject_context=True,
    approval="never",
    parameters={"document_id": Str("Document identifier.")},
)
def document_title(
    ctx: CapabilityContext,
    document_id: str,
) -> dict:
    """Use this to read a document's current title."""
    return {"title": ctx.require(Documents).title(document_id)}


agent = Agent(
    tools=[document_title],
    capabilities={Documents: documents},
)
```

The capability object is stored in the agent's `CapabilityContext`; tools resolve it
by type. A host can also call `agent.provide(Type, value)` and
`agent.add_tools(...)` after construction.

## Conversation ownership

Run authority is private. Its public controls are `approval=`, `interaction=`,
`agent.abort()`, event listeners, and the tool enable/disable methods.

## Prompts and events

```python
from lamssi_agents import ContextBlock

agent.add_context(ContextBlock("units", lambda ctx: "Use metric units.", stable=True))
unsubscribe = agent.add_event_listener(print)
```

Most applications put fixed text in `instructions`. `ContextBlock` is the advanced
layer for named state rendered from `ctx`; features can contribute it without changing
the core.

See `examples/` for runnable, offline-tested programs.
