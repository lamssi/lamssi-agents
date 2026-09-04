# SPDX-License-Identifier: MIT
"""16 - one model call returning a typed value, and sending an image.

``ask_model`` makes one model call with no tools or agent loop. It converts the
answer to the requested Python type.

Both halves need a real model. Set LAMSSI_MODEL, and for the image half point it
at something with vision.

    python examples/16_ask_model_and_vision.py
"""

import os

from lamssi_agents import Agent, ApprovalPolicy
from lamssi_agents import Guidance, SystemTools
from lamssi_agents.ask_model import build_ask_model

from _support import heading, real_model

heading("A typed answer, mid-script")

ask_model = build_ask_model(real_model())

print("""  ask_model("...", type=float)   -> a float, parsed out of prose
  ask_model("...", type=dict)    -> a dict
  ask_model("...", default=0.0)  -> 0.0 when conversion fails
  ask_model("...", callback=fn)  -> hands the value on; your path still validates
  ask_model.history              -> every call, for logging and inspection
""")

if os.environ.get("LAMSSI_RUN_MODEL_EXAMPLES"):
    reading = 4.7
    suggested = ask_model(
        f"A sensor reads {reading}. Suggest a gain between 0 and 10.",
        type=float, default=1.0,
    )
    print(f"  suggested gain: {suggested!r} ({type(suggested).__name__})")
else:
    print("  (set LAMSSI_RUN_MODEL_EXAMPLES=1 to actually call the model)")

heading("An image, to an agent")

print("""  agent.chat("what do you see?", image=path_or_array)

  Accepts a file path, raw bytes, a data: URL, a PIL image, a numpy array, or a
  list of any of those. Needs a vision-capable model; run
  `uv sync --locked --extra vision` for the encoding dependencies.
""")

if os.environ.get("LAMSSI_RUN_MODEL_EXAMPLES") and os.environ.get("LAMSSI_IMAGE"):
    agent = Agent(real_model(), features=[SystemTools(), Guidance()], approval=ApprovalPolicy.allow_all())
    print(agent.chat("Describe this image in one sentence.",
                     image=os.environ["LAMSSI_IMAGE"]))
