# SPDX-License-Identifier: MIT
"""15 - one model argument, from simple to custom.

Nothing here contacts a model endpoint.

    python examples/15_models.py
"""

from lamssi_agents import Agent, LiteLLMModel

from _support import ScriptedModel, heading, says


heading("Simple: a model identifier")
simple = Agent(model="openai/gpt-5-mini")
print("  model id:    ", simple.model_id)
print("  adapter:     ", simple.model_adapter_name)


heading("Configured: settings belong to the adapter")
configured_model = LiteLLMModel(
    "openai/gpt-5-mini",
    temperature=0.2,
    max_tokens=4_000,
)
configured = Agent(model=configured_model)
print("  same object: ", configured.model is configured_model)
print("  temperature:", configured_model.temperature)
print("  max tokens: ", configured_model.max_tokens)


heading("Custom: the same model seam")
custom_model = ScriptedModel(says("Hello from a custom adapter."))
custom = Agent(model=custom_model)
print("  same object: ", custom.model is custom_model)
print("  response:    ", custom.chat("Hello"))


heading("Replacement uses the same accepted values")
custom.use_model(ScriptedModel(says("Replacement active.")))
print("  response:    ", custom.chat("Hello again"))
