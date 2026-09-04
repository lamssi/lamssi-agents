"""Background activity spinner for the REPL."""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional


class _Activity:
    """Manage the CLI spinner and elapsed-time line."""

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, *, enabled: bool = True) -> None:
        self._stop_evt = threading.Event()
        self._text = ""
        self._start_ts = 0.0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        # Only enable in real interactive sessions; callers can force-disable for headless setups.
        self._enabled = bool(enabled) and sys.stdout.isatty()
        self._paused: bool = False
        # True once the spinner has painted the current line; pause() clears it only then, else it wipes streamed model text (no trailing \n).
        self._dirty: bool = False

    def start(self, text: str) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._text = text
            self._start_ts = time.monotonic()
            self._paused = False
        if self._thread is None or not self._thread.is_alive():
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def update(self, text: str, reset_clock: bool = True) -> None:
        """Set new label AND un-pause the paint loop."""
        if not self._enabled:
            return
        with self._lock:
            self._text = text
            if reset_clock:
                self._start_ts = time.monotonic()
            self._paused = False

    def note(self, text: str) -> None:
        """Set the label without taking the lock; never blocks, never raises.

        A console write under the lock isn't guaranteed to return (e.g. Windows
        QuickEdit with a selection, or an undrained pipe), which would stall the
        thread reading the model's stream. A bare assignment is GIL-atomic and
        the paint loop re-reads it each tick, so the label still lands, at most
        one frame late. Use this on hot paths; use :meth:`update` at turn
        boundaries, where un-pausing matters.
        """
        self._text = text

    def pause(self) -> None:
        """Suspend painting; clear the line only if the spinner drew on it.

        An unconditional clear would erase caller text already on the
        line (e.g. mid TEXT_DELTA stream).
        """
        if not self._enabled:
            return
        with self._lock:
            self._paused = True
            if self._dirty:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                self._dirty = False

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._enabled and self._dirty:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            self._dirty = False

    def _loop(self) -> None:
        i = 0
        while not self._stop_evt.wait(0.1):
            with self._lock:
                if self._paused:
                    i += 1
                    continue
                # Write + flush under the lock so pause() can't interleave a streamed chunk before our \033[0m closer.
                sys.stdout.write(
                    f"\r\033[K  \033[2m{self._FRAMES[i % len(self._FRAMES)]} "
                    f"{self._text}  {time.monotonic() - self._start_ts:.1f}s\033[0m"
                )
                sys.stdout.flush()
                self._dirty = True
            i += 1
