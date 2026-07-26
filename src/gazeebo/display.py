"""Generic live Wayland output-topology monitoring."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Protocol

from gazeebo.native import NativeRendererError, load_native_renderer

DISPLAY_ERROR_SIZE = 256
DISPLAY_SNAPSHOT_SIZE = 4096
GEOMETRY_FIELD_COUNT = 4

type OutputGeometry = tuple[int, int, int, int]


class DisplayMonitorError(RuntimeError):
    """Live output geometry could not be read safely."""


class DisplayMonitor(Protocol):
    """Refresh the current compositor-neutral output geometry."""

    def snapshot(self) -> tuple[OutputGeometry, ...]:
        """Return every current logical output rectangle."""

    def close(self) -> None:
        """Release the monitor idempotently."""


def parse_output_snapshot(value: str) -> tuple[OutputGeometry, ...]:
    """Parse and canonicalize the bounded native geometry snapshot."""
    if not value:
        msg = "display monitor returned no active outputs"
        raise DisplayMonitorError(msg)
    outputs: list[OutputGeometry] = []
    for item in value.split(";"):
        fields = item.split(":")
        if len(fields) != GEOMETRY_FIELD_COUNT:
            msg = "display monitor returned malformed output geometry"
            raise DisplayMonitorError(msg)
        try:
            x, y, width, height = (int(field) for field in fields)
        except ValueError as error:
            msg = "display monitor returned non-integer output geometry"
            raise DisplayMonitorError(msg) from error
        if width <= 0 or height <= 0:
            msg = "display monitor returned invalid output dimensions"
            raise DisplayMonitorError(msg)
        outputs.append((x, y, width, height))
    if len(outputs) != len(set(outputs)):
        msg = "display monitor returned ambiguous duplicate geometry"
        raise DisplayMonitorError(msg)
    return tuple(sorted(outputs))


@dataclass(slots=True)
class NativeDisplayMonitor:
    """Read live logical output changes from standard Wayland protocols."""

    _library: ctypes.CDLL
    _handle: ctypes.c_void_p
    _closed: bool = False

    @classmethod
    def create(cls) -> NativeDisplayMonitor:
        """Open one process-scoped native output monitor."""
        try:
            library = load_native_renderer()
        except NativeRendererError as error:
            raise DisplayMonitorError(str(error)) from error
        library.gazeebo_display_monitor_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_display_monitor_create.restype = ctypes.c_void_p
        library.gazeebo_display_monitor_snapshot.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_display_monitor_snapshot.restype = ctypes.c_int
        library.gazeebo_display_monitor_destroy.argtypes = [ctypes.c_void_p]
        library.gazeebo_display_monitor_destroy.restype = None
        error_buffer = ctypes.create_string_buffer(DISPLAY_ERROR_SIZE)
        handle = library.gazeebo_display_monitor_create(
            error_buffer,
            DISPLAY_ERROR_SIZE,
        )
        if not handle:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"display monitor failed to start: {detail}"
            raise DisplayMonitorError(msg)
        return cls(library, ctypes.c_void_p(handle))

    def snapshot(self) -> tuple[OutputGeometry, ...]:
        """Refresh and return canonical current logical output geometry."""
        if self._closed:
            msg = "display monitor is closed"
            raise DisplayMonitorError(msg)
        snapshot_buffer = ctypes.create_string_buffer(DISPLAY_SNAPSHOT_SIZE)
        error_buffer = ctypes.create_string_buffer(DISPLAY_ERROR_SIZE)
        result = self._library.gazeebo_display_monitor_snapshot(
            self._handle,
            snapshot_buffer,
            DISPLAY_SNAPSHOT_SIZE,
            error_buffer,
            DISPLAY_ERROR_SIZE,
        )
        if result != 0:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"display refresh failed: {detail}"
            raise DisplayMonitorError(msg)
        return parse_output_snapshot(snapshot_buffer.value.decode(errors="strict"))

    def close(self) -> None:
        """Release the native monitor once."""
        if self._closed:
            return
        self._closed = True
        self._library.gazeebo_display_monitor_destroy(self._handle)
