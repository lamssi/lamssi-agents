# SPDX-License-Identifier: MIT
"""17 - running an agent inside a Tkinter application.

Tkinter ships with Python and has the same thread constraint as most desktop UI
toolkits: widgets belong to the main thread. The example prints the thread used
for each operation:

  * `agent.chat()` blocks, so it runs on a worker thread and the UI stays live.
  * A tool that touches widgets declares `dispatch="gui"` and is marshalled to
    the main thread by the host's dispatcher.
  * A tool that talks to an instrument declares `dispatch="worker"` and is kept
    off the main thread, so a slow device never freezes the window.
  * Approval is a real dialog. The agent thread waits while a person answers.

    python 17_desktop_app.py              # a real window
    python 17_desktop_app.py --headless   # the thread report, then exit

The default scripted model needs no server or API key. Set ``LAMSSI_MODEL`` to
use a live model and enter your own prompts.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time

from lamssi_agents import Agent, ApprovalPolicy, ToolApproval, tool
from lamssi_agents.events import AgentEventType
from lamssi_tools import Expose, Float

from _support import ScriptedModel, calls, heading, real_model, says

try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext
except Exception as exc:                                   # pragma: no cover
    print(f"tkinter is not available here ({exc}); nothing to demonstrate.")
    sys.exit(0)

# Headless by env var so the example suite can drive it without a blocking mainloop.
HEADLESS = "--headless" in sys.argv or os.environ.get("LAMSSI_EXAMPLE_HEADLESS") == "1"
SHOW = not HEADLESS

# Record the execution thread for the report printed at the end.
THREADS: dict = {}

# Keep tool errors for the final headless report.
FAILURES: list = []


class Panel:
    """A tiny instrument panel.

    State lives in a plain dict behind a lock and is readable from any thread;
    Widgets mirror this value and are updated only on the GUI thread. Reading a
    ``tk.StringVar`` from a worker raises ``main thread is not in main loop``.
    """

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("lamssi-agents in a desktop app")
        self.gui_thread = threading.current_thread().name

        # Safe from any thread.
        self._lock = threading.Lock()
        self._state = {"setpoint": 20.0, "reading": 20.0, "status": "idle"}

        # The mirror. Only ever touched on the GUI thread.
        self.setpoint = tk.StringVar(value="20.0")
        self.reading = tk.StringVar(value="20.0")
        self.status = tk.StringVar(value="idle")

        for label, var in (("Setpoint (C)", self.setpoint),
                           ("Reading  (C)", self.reading),
                           ("Status", self.status)):
            row = tk.Frame(self.root)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=label, width=14, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, anchor="w").pack(side="left")

        self.log = scrolledtext.ScrolledText(self.root, height=14, width=76)
        self.log.pack(padx=8, pady=6)

        if not SHOW:
            # Headless: Tk still needs to exist since widgets have an owning thread the dispatcher must reach.
            self.root.withdraw()

    def read(self, key: str):
        with self._lock:
            return self._state[key]

    def update(self, **values) -> None:
        with self._lock:
            self._state.update(values)

    def refresh(self) -> None:
        """Push state into the widgets. The caller marshals; this does not."""
        with self._lock:
            snapshot = dict(self._state)
        self.setpoint.set(f"{snapshot['setpoint']:.1f}")
        self.reading.set(f"{snapshot['reading']:.1f}")
        self.status.set(str(snapshot["status"]))

    def write(self, line: str) -> None:
        """Widget access. Only ever called on the GUI thread."""
        self.log.insert("end", line + "\n")
        self.log.see("end")


class GuiBridge:
    """Runs a callable on the GUI thread and blocks the caller for the result.

    Tk's `after` is only safe to schedule from the main thread, so work is
    queued and a repeating pump drains it there.
    """

    def __init__(self, panel: Panel) -> None:
        self._root = panel.root
        self._main = threading.current_thread()
        self._jobs: queue.Queue = queue.Queue()
        self._root.after(10, self._pump)

    def _pump(self) -> None:
        while True:
            try:
                fn, kwargs, box, done = self._jobs.get_nowait()
            except queue.Empty:
                break
            try:
                box["value"] = fn(**kwargs)
            except BaseException as exc:                    # noqa: BLE001
                box["error"] = exc
            done.set()
        self._root.after(10, self._pump)

    def __call__(self, fn, *, timeout: float = 60.0, **kwargs):
        if threading.current_thread() is self._main:
            # Already on the GUI thread; queueing would deadlock waiting on a pump that can't run until we return.
            return fn(**kwargs)

        box: dict = {}
        done = threading.Event()
        self._jobs.put((fn, kwargs, box, done))
        if not done.wait(timeout):
            raise TimeoutError(f"the GUI thread did not answer in {timeout:g}s")
        if "error" in box:
            raise box["error"]
        return box["value"]


class Instrument:
    """A serial worker queue, standing in for hardware that must not be shared."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.thread_name = "instrument"

    def run(self, fn, kwargs: dict):
        box: dict = {}

        def body() -> None:
            try:
                box["value"] = fn(**kwargs)
            except BaseException as exc:                    # noqa: BLE001
                box["error"] = exc

        with self._lock:
            worker = threading.Thread(target=body, name=self.thread_name)
            worker.start()
            worker.join(60.0)
        if "error" in box:
            raise box["error"]
        return box.get("value")


def thread_dispatcher(gui: GuiBridge, instrument: Instrument):
    """`dispatch="gui"` to the main thread, `"worker"` to the instrument queue."""

    def dispatch(definition, fn, kwargs):
        tag = getattr(definition, "dispatch", None)
        if tag == "gui":
            return gui(fn, **kwargs)
        if tag == "worker":
            return instrument.run(fn, dict(kwargs))
        return fn(**kwargs)

    return dispatch


def build_tools(panel: Panel, gui: "GuiBridge"):
    @tool(expose=Expose.AGENT, approval="never")
    def read_setpoint() -> dict:
        """Read the configured setpoint. No tag: plain locked state needs no hand-off."""
        THREADS["read_setpoint"] = threading.current_thread().name
        return {"setpoint_c": panel.read("setpoint")}

    @tool(
        dispatch="gui",
        expose=Expose.AGENT,
        approval="always",
        parameters={
            "celsius": Float("Target temperature.", ge=0, le=300)
        },
    )
    def set_setpoint(celsius: float = 20.0) -> dict:
        """Move the setpoint and redraw. Touching widgets needs the GUI thread."""
        THREADS["set_setpoint"] = threading.current_thread().name
        panel.update(setpoint=celsius, status=f"setpoint {celsius:.1f} C")
        panel.refresh()
        return {"setpoint_c": celsius}

    @tool(dispatch="worker", expose=Expose.AGENT, approval="always")
    def ramp_to_setpoint() -> dict:
        """Drive the heater to the setpoint; slow, so it must never block the UI."""
        THREADS["ramp_to_setpoint"] = threading.current_thread().name
        target = panel.read("setpoint")
        time.sleep(0.4)                       # a real instrument would take longer
        panel.update(reading=target, status="settled")
        gui(panel.refresh)
        return {"reached_c": target, "settled": True}

    return [read_setpoint, set_setpoint, ramp_to_setpoint]


def make_agent(panel: Panel, gui: GuiBridge, instrument: Instrument):
    agent = Agent(
        real_model() if os.environ.get("LAMSSI_MODEL") else None,
        tools=build_tools(panel, gui),
        only=["read_setpoint", "set_setpoint", "ramp_to_setpoint"],
        approval=ApprovalPolicy.ask_when_required(
            lambda request: (
                ToolApproval.APPROVE
                if gui(
                    lambda: _ask(panel, request.tool, request.arguments), timeout=300.0
                )
                else ToolApproval.REJECT
            )
        ),
    )
    agent.set_tool_dispatcher(thread_dispatcher(gui, instrument))

    def on_event(event) -> None:
        # Called from the agent's thread. Every widget touch is marshalled.
        if event.type is AgentEventType.TOOL_START:
            gui(lambda: panel.write(f"-> {event.data}"))
        elif event.type is AgentEventType.TOOL_RESULT:
            body = str(event.data)
            # A raising tool doesn't stop the run, so failures are collected and reported separately.
            if "error" in body.lower():
                FAILURES.append(body[:120])
            gui(lambda: panel.write(f"<- {body[:60]}"))
        elif event.type is AgentEventType.TEXT_DONE:
            gui(lambda: panel.write(f" = {event.data}"))

    agent.add_event_listener(on_event)
    return agent


def _report(panel: Panel) -> None:
    """Write the thread report into the log, where the window can show it."""
    panel.write("")
    panel.write("threads this run:")
    for label in ("agent_loop", "read_setpoint", "set_setpoint",
                  "ramp_to_setpoint", "approval"):
        panel.write(f"  {label:<18} {THREADS.get(label, '(not reached)')}")
    panel.write(f"  {'GUI thread':<18} {panel.gui_thread}")
    panel.write("")


def _ask(panel: Panel, name: str, args: dict) -> bool:
    """The approval dialog. Runs on the GUI thread; the agent thread waits."""
    THREADS["approval"] = threading.current_thread().name
    if not SHOW:
        panel.write(f" ? approve {name} -> yes (headless)")
        return True
    shown = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())
    return bool(messagebox.askyesno("Approve?", f"Run {name}({shown})?"))


def main() -> None:
    try:
        panel = Panel()
    except tk.TclError as exc:
        # A real Tk interpreter is required to exercise widget thread ownership.
        print(f"  no display available ({exc}).")
        print("  This example needs a desktop session; on Linux CI, run it under `xvfb-run`.")
        return

    gui = GuiBridge(panel)
    instrument = Instrument()
    agent = make_agent(panel, gui, instrument)

    if SHOW:
        _run_interactive(panel, gui, agent)
        return

    # Headless: one scripted conversation, driven from a worker thread as a real one would be.
    agent.use_model(ScriptedModel(
        calls("read_setpoint"),
        calls("set_setpoint", celsius=42.0),
        calls("ramp_to_setpoint"),
        says("Setpoint moved to 42 C and the heater has settled there."),
    ))

    heading("The agent runs on a worker thread; the UI thread stays free")

    finished = threading.Event()

    def run() -> None:
        THREADS["agent_loop"] = threading.current_thread().name
        try:
            agent.chat("Set the heater to 42 degrees and ramp to it.")
        finally:
            finished.set()

    threading.Thread(target=run, name="agent", daemon=True).start()

    # Pump events manually during the headless run.
    pumps = 0
    while not finished.wait(0.01):
        panel.root.update()
        pumps += 1

    panel.root.update()

    print(f"  GUI thread            : {panel.gui_thread}")
    print(f"  pumped {pumps} times while the agent worked (a frozen UI would be 0)")
    print()
    for label, expected in (("agent_loop", "agent"),
                            ("read_setpoint", "agent"),
                            ("set_setpoint", panel.gui_thread),
                            ("ramp_to_setpoint", "instrument"),
                            ("approval", panel.gui_thread)):
        actual = THREADS.get(label, "(not reached)")
        mark = "ok " if actual == expected else "!! "
        print(f"  {mark}{label:<20} ran on {actual!r} (expected {expected!r})")

    if FAILURES:
        print()
        print("  TOOLS FAILED - the thread report above is meaningless:")
        for failure in FAILURES:
            print(f"    {failure}")
        panel.root.destroy()
        raise SystemExit(1)

    print("""
  That is the whole embedded story:
    - chat() blocks, so it lives on a worker and the window keeps painting
    - a widget-touching tool declared "gui" and arrived on the GUI thread
    - a slow instrument tool declared "worker" and stayed off it
    - approval reached a real dialog while the agent thread waited

  None of it needed an event loop, and none of it needed the framework to know
  what Tkinter is. The dispatcher is twelve lines and it is the only place a
  tag becomes a thread.
""")
    panel.root.destroy()


def _run_interactive(panel: Panel, gui: GuiBridge, agent) -> None:  # pragma: no cover
    """A real window: type a prompt, watch it work, approve what it asks."""
    entry = tk.Entry(panel.root, width=76)
    entry.pack(padx=8, pady=(0, 8))
    entry.insert(0, "Set the heater to 42 degrees and ramp to it.")

    def send(_event=None) -> None:
        prompt = entry.get().strip()
        if not prompt:
            return
        entry.delete(0, "end")
        panel.write(f"you: {prompt}")
        THREADS.clear()

        if agent.model is None or not os.environ.get("LAMSSI_MODEL"):
            # Replay a deterministic conversation when no model is configured.
            agent.use_model(ScriptedModel(
                calls("read_setpoint"),
                calls("set_setpoint", celsius=42.0),
                calls("ramp_to_setpoint"),
                says("Setpoint moved to 42 C and the heater has settled there."),
            ))

        def run() -> None:
            THREADS["agent_loop"] = threading.current_thread().name
            try:
                agent.chat(prompt)
            except Exception as exc:                        # noqa: BLE001
                gui(lambda error=str(exc): panel.write(f"error: {error}"))
            finally:
                gui(lambda: _report(panel))

        threading.Thread(target=run, name="agent", daemon=True).start()

    entry.bind("<Return>", send)
    tk.Button(panel.root, text="Send", command=send).pack(pady=(0, 8))

    if os.environ.get("LAMSSI_MODEL"):
        panel.write(f"model: {real_model()}")
    else:
        panel.write("model: scripted (set LAMSSI_MODEL for a real one)")
    panel.write("Press Send. The window stays responsive while the agent works.")
    panel.write("")
    panel.root.deiconify()
    panel.root.mainloop()


if __name__ == "__main__":
    main()
