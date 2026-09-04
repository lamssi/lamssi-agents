# SPDX-License-Identifier: MIT
"""18 - running an agent inside a PySide6 application.

The host maps Lamssi's dispatch tags onto Qt and instrument threads:

* ``agent.chat()`` is blocking, so the host runs it on a worker thread.
* ``dispatch="gui"`` is interpreted by the host as "run on Qt's main thread".
* ``dispatch="worker"`` is interpreted by the host as "run on the instrument
  thread".
* approval is a real ``QMessageBox`` on the GUI thread while the agent thread
  waits synchronously for the answer.

Run it with::

    python 18_pyside6_app.py
    python 18_pyside6_app.py --headless

The example uses a scripted model unless ``LAMSSI_MODEL`` is set, so no API key
or model server is required to exercise the threading behavior.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Any, Callable

# Set the offscreen Qt platform before importing/constructing QApplication.
HEADLESS = (
    "--headless" in sys.argv
    or os.environ.get("LAMSSI_EXAMPLE_HEADLESS") == "1"
)
SHOW = not HEADLESS
if HEADLESS:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QObject, Qt, Signal, Slot
    from PySide6.QtWidgets import (
        QApplication,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover
    print(f"PySide6 is not available here ({exc}); nothing to demonstrate.")
    print('Install the Qt example extra, e.g. `uv sync --locked --extra qt`.')
    raise SystemExit(0) from None

from lamssi_agents import Agent, ApprovalPolicy, ToolApproval, tool
from lamssi_agents.events import AgentEventType
from lamssi_tools import Expose, Float

from _support import ScriptedModel, calls, heading, real_model, says


# Every tool records where it actually ran so the example proves its claims.
THREADS: dict[str, str] = {}

# Tool errors are collected because a raising tool is converted into a tool result, so the run can continue successfully anyway.
FAILURES: list[str] = []


class Panel(QMainWindow):
    """A tiny instrument panel.

    State is protected by a lock and readable/updatable from worker threads;
    Qt widgets, which merely present it, are touched only on the GUI thread.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("lamssi-agents in a PySide6 application")
        self.gui_thread = threading.current_thread().name

        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "setpoint": 20.0,
            "reading": 20.0,
            "status": "idle",
        }

        self.setpoint_value = QLabel("20.0")
        self.reading_value = QLabel("20.0")
        self.status_value = QLabel("idle")

        form = QFormLayout()
        form.addRow("Setpoint (C)", self.setpoint_value)
        form.addRow("Reading (C)", self.reading_value)
        form.addRow("Status", self.status_value)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(260)

        self.entry = QLineEdit()
        self.entry.setPlaceholderText("Ask the embedded agent…")
        self.entry.setText("Set the heater to 42 degrees and ramp to it.")

        self.send_button = QPushButton("Send")

        input_row = QHBoxLayout()
        input_row.addWidget(self.entry, 1)
        input_row.addWidget(self.send_button)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.log, 1)
        layout.addLayout(input_row)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(720, 520)

    # Thread-safe state

    def read(self, key: str) -> Any:
        with self._lock:
            return self._state[key]

    def update_state(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    # GUI-thread widgets

    def refresh(self) -> None:
        """Push application state into the Qt widgets."""
        with self._lock:
            snapshot = dict(self._state)

        self.setpoint_value.setText(f"{snapshot['setpoint']:.1f}")
        self.reading_value.setText(f"{snapshot['reading']:.1f}")
        self.status_value.setText(str(snapshot["status"]))

    def write(self, line: str) -> None:
        """Append to the GUI log.  The caller is responsible for marshalling."""
        self.log.append(line)


class GuiBridge(QObject):
    """Synchronously execute a Python callable on Qt's GUI thread.

    A queued Qt signal hands work to the GUI thread; the calling worker blocks
    on a ``threading.Event`` until the slot stores a value or an exception.
    The host application owns this bridge.
    """

    _requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._main = threading.current_thread()
        self._requested.connect(
            self._execute,
            Qt.ConnectionType.QueuedConnection,
        )

    @Slot(object)
    def _execute(self, job: object) -> None:
        fn, kwargs, box, done = job
        try:
            box["value"] = fn(**kwargs)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            done.set()

    def __call__(
        self,
        fn: Callable[..., Any],
        *,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> Any:
        if threading.current_thread() is self._main:
            # Already on the GUI thread; queueing would deadlock since the event loop can't process it until we return.
            return fn(**kwargs)

        box: dict[str, Any] = {}
        done = threading.Event()
        self._requested.emit((fn, kwargs, box, done))

        if not done.wait(timeout):
            raise TimeoutError(f"the Qt GUI thread did not answer in {timeout:g}s")

        if "error" in box:
            raise box["error"]

        return box.get("value")


class Instrument:
    """A dedicated serial worker thread standing in for real hardware.

    The host owns this thread and targets it from the dispatcher.
    """

    def __init__(self) -> None:
        self.thread_name = "instrument"
        self._jobs: queue.Queue[object] = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop,
            name=self.thread_name,
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return

            fn, kwargs, box, done = job
            try:
                box["value"] = fn(**kwargs)
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc
            finally:
                done.set()

    def run(
        self,
        fn: Callable[..., Any],
        kwargs: dict[str, Any],
        *,
        timeout: float = 60.0,
    ) -> Any:
        box: dict[str, Any] = {}
        done = threading.Event()
        self._jobs.put((fn, kwargs, box, done))

        if not done.wait(timeout):
            raise TimeoutError(
                f"instrument thread did not answer in {timeout:g}s"
            )

        if "error" in box:
            raise box["error"]

        return box.get("value")

    def close(self) -> None:
        self._jobs.put(None)

def thread_dispatcher(gui: GuiBridge, instrument: Instrument):
    """Map Lamssi's opaque dispatch tags onto this application's execution model."""

    def dispatch(definition, fn, kwargs):
        tag = getattr(definition, "dispatch", None)

        if tag == "gui":
            return gui(fn, **kwargs)

        if tag == "worker":
            return instrument.run(fn, dict(kwargs))

        return fn(**kwargs)

    return dispatch


def build_tools(panel: Panel, gui: GuiBridge):
    @tool(expose=Expose.AGENT, approval="never")
    def read_setpoint() -> dict:
        """Read the configured temperature setpoint."""
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
    def set_setpoint(
        celsius: float = 20.0,
    ) -> dict:
        """Change the setpoint and redraw the panel."""
        THREADS["set_setpoint"] = threading.current_thread().name
        panel.update_state(
            setpoint=celsius,
            status=f"setpoint {celsius:.1f} C",
        )
        panel.refresh()
        return {"setpoint_c": celsius}

    @tool(
        dispatch="worker",
        expose=Expose.AGENT,
        approval="always",
    )
    def ramp_to_setpoint() -> dict:
        """Drive the simulated instrument to the setpoint, off the GUI thread."""
        THREADS["ramp_to_setpoint"] = threading.current_thread().name

        target = panel.read("setpoint")
        time.sleep(0.4)  # a real instrument would take much longer
        panel.update_state(reading=target, status="settled")

        # On the instrument thread here; the host marshals the widget update back to Qt's main thread.
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
                    lambda: _ask(
                        panel,
                        request.tool,
                        request.arguments,
                    ),
                    timeout=300.0,
                )
                else ToolApproval.REJECT
            )
        ),
    )

    agent.set_tool_dispatcher(thread_dispatcher(gui, instrument))

    def on_event(event) -> None:
        # Agent events arrive on the agent's worker thread, so QWidget access is marshalled through the Qt bridge.
        if event.type is AgentEventType.TOOL_START:
            gui(lambda: panel.write(f"-> {event.data}"))

        elif event.type is AgentEventType.TOOL_RESULT:
            body = str(event.data)
            if "error" in body.lower():
                FAILURES.append(body[:120])
            gui(lambda: panel.write(f"<- {body[:60]}"))

        elif event.type is AgentEventType.TEXT_DONE:
            gui(lambda: panel.write(f" = {event.data}"))

    agent.add_event_listener(on_event)
    return agent


def _ask(panel: Panel, name: str, args: dict) -> bool:
    """Show approval on Qt's GUI thread while the agent worker waits."""
    THREADS["approval"] = threading.current_thread().name

    shown = ", ".join(
        f"{key}={value!r}" for key, value in (args or {}).items()
    )

    if not SHOW:
        panel.write(f" ? approve {name}({shown}) -> yes (headless)")
        return True

    answer = QMessageBox.question(
        panel,
        "Approve tool call?",
        f"Run {name}({shown})?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


def _report(panel: Panel) -> None:
    panel.write("")
    panel.write("threads this run:")

    for label in (
        "agent_loop",
        "read_setpoint",
        "set_setpoint",
        "ramp_to_setpoint",
        "approval",
    ):
        panel.write(f"  {label:<18} {THREADS.get(label, '(not reached)')}")

    panel.write(f"  {'GUI thread':<18} {panel.gui_thread}")
    panel.write("")


def scripted_heater_run(agent) -> None:
    """Install the deterministic model used by the no-key demonstration."""
    agent.use_model(
        ScriptedModel(
            calls("read_setpoint"),
            calls("set_setpoint", celsius=42.0),
            calls("ramp_to_setpoint"),
            says("Setpoint moved to 42 C and the heater has settled there."),
        )
    )

def _run_interactive(
    app: QApplication,
    panel: Panel,
    gui: GuiBridge,
    instrument: Instrument,
    agent,
) -> None:  # pragma: no cover
    """A real Qt window: type a prompt, approve calls, and watch thread hops."""

    def send() -> None:
        prompt = panel.entry.text().strip()
        if not prompt:
            return

        panel.entry.clear()
        panel.write(f"you: {prompt}")
        THREADS.clear()
        FAILURES.clear()
        panel.send_button.setEnabled(False)

        if agent.model is None or not os.environ.get("LAMSSI_MODEL"):
            scripted_heater_run(agent)

        def run() -> None:
            THREADS["agent_loop"] = threading.current_thread().name
            try:
                agent.chat(prompt)
            except Exception as exc:  # noqa: BLE001
                gui(lambda error=str(exc): panel.write(f"error: {error}"))
            finally:
                gui(lambda: _report(panel))
                gui(lambda: panel.send_button.setEnabled(True))

        threading.Thread(
            target=run,
            name="agent",
            daemon=True,
        ).start()

    panel.entry.returnPressed.connect(send)
    panel.send_button.clicked.connect(send)

    if os.environ.get("LAMSSI_MODEL"):
        panel.write(f"model: {real_model()}")
    else:
        panel.write("model: scripted (set LAMSSI_MODEL for a real one)")

    panel.write(
        "Press Send. The Qt window stays responsive while the agent works."
    )
    panel.write("")

    panel.show()
    app.aboutToQuit.connect(instrument.close)
    app.exec()


def _run_headless(
    app: QApplication,
    panel: Panel,
    instrument: Instrument,
    agent,
) -> None:
    """Exercise exactly the same Qt marshalling without showing a window."""
    scripted_heater_run(agent)

    heading("The agent runs on a worker thread; the Qt GUI thread stays free")

    finished = threading.Event()

    def run() -> None:
        THREADS["agent_loop"] = threading.current_thread().name
        try:
            agent.chat("Set the heater to 42 degrees and ramp to it.")
        finally:
            finished.set()

    threading.Thread(
        target=run,
        name="agent",
        daemon=True,
    ).start()

    # Drives the same Qt event processing manually so the example can print its assertions and exit.
    pumps = 0
    while not finished.wait(0.01):
        app.processEvents()
        pumps += 1

    app.processEvents()

    print(f"  Qt GUI thread         : {panel.gui_thread}")
    print(
        f"  processed Qt events {pumps} times while the agent worked "
        "(a frozen UI would be 0)"
    )
    print()

    checks = (
        ("agent_loop", "agent"),
        ("read_setpoint", "agent"),
        ("set_setpoint", panel.gui_thread),
        ("ramp_to_setpoint", "instrument"),
        ("approval", panel.gui_thread),
    )

    for label, expected in checks:
        actual = THREADS.get(label, "(not reached)")
        mark = "ok " if actual == expected else "!! "
        print(
            f"  {mark}{label:<20} ran on {actual!r} "
            f"(expected {expected!r})"
        )

    if FAILURES:
        print()
        print("  TOOLS FAILED - the thread report above is meaningless:")
        for failure in FAILURES:
            print(f"    {failure}")
        instrument.close()
        raise SystemExit(1)

    print(
        """
  That is the whole embedded Qt story:
    - chat() blocks, so the host puts it on an agent worker
    - a widget-touching tool declared "gui" and arrived on Qt's main thread
    - a slow instrument tool declared "worker" and ran on the instrument thread
    - approval reached Qt's GUI thread while the agent worker waited

  The application created Qt's event loop and mapped the opaque dispatch tags
  to concrete execution threads.
"""
    )

    instrument.close()


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)

    panel = Panel()
    if HEADLESS:
        panel.hide()

    # QObject affinity is established by construction: created on the Qt main thread, so GuiBridge receives queued invocations there.
    gui = GuiBridge()
    instrument = Instrument()
    agent = make_agent(panel, gui, instrument)

    if SHOW:
        _run_interactive(app, panel, gui, instrument, agent)
    else:
        _run_headless(app, panel, instrument, agent)


if __name__ == "__main__":
    main()
