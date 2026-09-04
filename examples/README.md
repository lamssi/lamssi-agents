# Examples

Nineteen small, self-contained programs. Each one is safe to copy and carries its own
license header. Read them in order for a tour, or jump to the one you need.

## Running

From this directory:

```bash
cd examples
uv run python 01_hello.py
```

Everything runs offline except the two marked **needs a model**, which call a live
provider: set that provider's credentials (for example `OPENAI_API_KEY`) and
`LAMSSI_RUN_MODEL_EXAMPLES=1`. The desktop examples (17, 18) open a window; set
`LAMSSI_EXAMPLE_HEADLESS=1` to run them without one.

## Index

| Example | What it shows |
| --- | --- |
| [01_hello.py](01_hello.py) | The smallest thing that works. **Needs a model.** |
| [02_your_own_tools.py](02_your_own_tools.py) | Ordinary Python functions as tools. |
| [03_typed_tools.py](03_typed_tools.py) | Describing and validating tools with `@tool`. |
| [04_approval.py](04_approval.py) | Deciding what needs your say-so with an explicit application policy. |
| [05_events.py](05_events.py) | Watching a run through the event stream. |
| [06_cost_control.py](06_cost_control.py) | What a turn costs, and how to spend less. |
| [07_capabilities.py](07_capabilities.py) | Providing an application capability to a tool. |
| [08_dynamic_context.py](08_dynamic_context.py) | Putting your own context in the prompt. |
| [09_skills.py](09_skills.py) | Skills: procedures the agent loads when they match the job. |
| [10_memory.py](10_memory.py) | Notes that outlive one conversation. |
| [11_hosting_an_agent.py](11_hosting_an_agent.py) | A host composing an Agent directly. |
| [12_custom_feature.py](12_custom_feature.py) | Installing tools and observing their lifecycle with a Feature. |
| [13_write_hooks_and_roots.py](13_write_hooks_and_roots.py) | Checking what the agent writes, and where it may write. |
| [14_logging_and_history.py](14_logging_and_history.py) | The record of a run, and keeping the context window under control. |
| [15_models.py](15_models.py) | One model argument, from simple to custom. |
| [16_ask_model_and_vision.py](16_ask_model_and_vision.py) | One model call returning a typed value, and sending an image. **Needs a model.** |
| [17_desktop_app.py](17_desktop_app.py) | Running an agent inside a Tkinter application. |
| [18_pyside6_app.py](18_pyside6_app.py) | Running an agent inside a PySide6 application. |
| [19_compaction_strategy.py](19_compaction_strategy.py) | A compaction strategy, packaged as a feature. |
