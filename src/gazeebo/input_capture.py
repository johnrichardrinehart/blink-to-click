"""Optional XDG InputCapture pointer refinement with a libei receiver."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

from dbus_next.aio.message_bus import MessageBus
from dbus_next.constants import MessageType
from dbus_next.message import Message
from dbus_next.signature import Variant

from gazeebo.portal import (
    PORTAL_DESTINATION,
    PORTAL_PATH,
    SESSION,
    PortalError,
    _Bus,
    _RequestBroker,
    _token,
    _unwrap,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

INPUT_CAPTURE = "org.freedesktop.portal.InputCapture"
PROPERTIES = "org.freedesktop.DBus.Properties"
POINTER_CAPABILITY = 2
INPUT_ERROR_SIZE = 512
_ZONE_FIELDS = 4


class InputCaptureError(RuntimeError):
    """Input capture is unavailable or disconnected without affecting navigation."""


_MotionCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_void_p,
)


class _NativeInputReceiver:
    """Dispatch pointer-only libei events on the asyncio loop."""

    def __init__(
        self,
        descriptor: int,
        motion: Callable[[int, float, float], None],
        disconnected: Callable[[str], None],
    ) -> None:
        library = ctypes.CDLL(str(_native_library_path()))
        library.gazeebo_input_create.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_input_create.restype = ctypes.c_void_p
        library.gazeebo_input_get_fd.argtypes = [ctypes.c_void_p]
        library.gazeebo_input_get_fd.restype = ctypes.c_int
        library.gazeebo_input_dispatch.argtypes = [
            ctypes.c_void_p,
            _MotionCallback,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_input_dispatch.restype = ctypes.c_int
        library.gazeebo_input_destroy.argtypes = [ctypes.c_void_p]
        library.gazeebo_input_destroy.restype = None
        error = ctypes.create_string_buffer(INPUT_ERROR_SIZE)
        handle = library.gazeebo_input_create(descriptor, error, INPUT_ERROR_SIZE)
        if not handle:
            detail = error.value.decode(errors="replace") or "libei setup failed"
            raise InputCaptureError(detail)
        self._library = library
        self._handle = handle
        self._motion = motion
        self._disconnected = disconnected
        self._callback = _MotionCallback(self._receive)
        self._descriptor = int(library.gazeebo_input_get_fd(handle))
        self._loop = asyncio.get_running_loop()
        self._closed = False
        self._loop.add_reader(self._descriptor, self._dispatch)

    def close(self) -> None:
        """Remove dispatch and release the libei context idempotently."""
        if self._closed:
            return
        self._closed = True
        self._loop.remove_reader(self._descriptor)
        self._library.gazeebo_input_destroy(self._handle)

    def _receive(
        self,
        absolute: int,
        x: float,
        y: float,
        _user_data: ctypes.c_void_p,
    ) -> None:
        self._motion(absolute, x, y)

    def _dispatch(self) -> None:
        error = ctypes.create_string_buffer(INPUT_ERROR_SIZE)
        result = self._library.gazeebo_input_dispatch(
            self._handle,
            self._callback,
            None,
            error,
            INPUT_ERROR_SIZE,
        )
        if result != 0 and not self._closed:
            detail = error.value.decode(errors="replace") or "libei disconnected"
            self.close()
            self._disconnected(detail)


class PortalInputCapture:
    """Own one optional barrier-activated InputCapture/EIS session."""

    def __init__(
        self,
        bus: _Bus,
        session_path: str,
        receiver: _NativeInputReceiver,
        barrier_count: int,
    ) -> None:
        """Retain one enabled portal session and its native receiver."""
        self._bus = bus
        self._session_path = session_path
        self._receiver = receiver
        self.barrier_count = barrier_count
        self._closed = False

    @classmethod
    async def authorize(
        cls,
        motion: Callable[[int, float, float], None],
        disconnected: Callable[[str], None],
        *,
        request_timeout: float = 300.0,
    ) -> PortalInputCapture:
        """Authorize pointer capture, install boundary barriers, and enable it."""
        connected = await MessageBus().connect()
        bus = cast("_Bus", connected)
        receiver: _NativeInputReceiver | None = None
        session_path: str | None = None
        try:
            capabilities = await _supported_capabilities(bus)
            _require_pointer_capability(capabilities)
            broker = _RequestBroker(bus, request_timeout)
            create = await broker.request(
                INPUT_CAPTURE,
                "CreateSession",
                "sa{sv}",
                [
                    "",
                    {
                        "handle_token": Variant("s", _token("request")),
                        "session_handle_token": Variant("s", _token("input")),
                        "capabilities": Variant("u", POINTER_CAPABILITY),
                    },
                ],
            )
            session_path = _granted_session_path(create)
            zones_result = await broker.request(
                INPUT_CAPTURE,
                "GetZones",
                "oa{sv}",
                [session_path, {"handle_token": Variant("s", _token("request"))}],
            )
            zones = _parse_zones(zones_result.get("zones"))
            zone_set = _integer(zones_result.get("zone_set", 0), "zone set")
            barriers = _zone_barriers(zones)
            barrier_result = await broker.request(
                INPUT_CAPTURE,
                "SetPointerBarriers",
                "oa{sv}aa{sv}u",
                [
                    session_path,
                    {"handle_token": Variant("s", _token("request"))},
                    barriers,
                    zone_set,
                ],
            )
            failed_values = cast("list[object]", barrier_result.get("failed_barriers", []))
            failed = {_integer(item, "failed barrier") for item in failed_values}
            accepted_count = _accepted_barrier_count(len(barriers), failed)
            descriptor = await _connect_to_eis(bus, session_path)
            receiver = _NativeInputReceiver(descriptor, motion, disconnected)
            await _call(bus, INPUT_CAPTURE, "Enable", "oa{sv}", [session_path, {}])
            return cls(bus, session_path, receiver, accepted_count)
        except (InputCaptureError, PortalError, OSError, ValueError) as error:
            if receiver is not None:
                receiver.close()
            if session_path is not None:
                await _close_session(bus, session_path)
            bus.disconnect()
            if isinstance(error, InputCaptureError):
                raise
            raise InputCaptureError(str(error)) from error

    async def close(self) -> None:
        """Disable capture and release EIS, portal session, and bus idempotently."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(InputCaptureError):
            await _call(
                self._bus,
                INPUT_CAPTURE,
                "Disable",
                "oa{sv}",
                [self._session_path, {}],
            )
        self._receiver.close()
        await _close_session(self._bus, self._session_path)
        self._bus.disconnect()


def _require_pointer_capability(capabilities: int) -> None:
    if not capabilities & POINTER_CAPABILITY:
        msg = "XDG InputCapture pointer capability is unsupported"
        raise InputCaptureError(msg)


def _granted_session_path(results: dict[str, object]) -> str:
    raw_path = results.get("session_handle")
    granted = _integer(results.get("capabilities", 0), "granted capabilities")
    if not isinstance(raw_path, str) or not granted & POINTER_CAPABILITY:
        msg = "InputCapture session did not grant pointer capability"
        raise InputCaptureError(msg)
    return raw_path


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"InputCapture portal returned invalid {name}"
        raise InputCaptureError(msg)
    return value


def _accepted_barrier_count(total: int, failed: set[int]) -> int:
    count = total - len(failed)
    if count <= 0:
        msg = "InputCapture accepted no pointer activation barriers"
        raise InputCaptureError(msg)
    return count


async def _supported_capabilities(bus: _Bus) -> int:
    reply = await _call(
        bus,
        PROPERTIES,
        "Get",
        "ss",
        [INPUT_CAPTURE, "SupportedCapabilities"],
    )
    if not reply.body:
        return 0
    return _integer(_unwrap(reply.body[0]), "supported capabilities")


async def _connect_to_eis(bus: _Bus, session_path: str) -> int:
    reply = await _call(
        bus,
        INPUT_CAPTURE,
        "ConnectToEIS",
        "oa{sv}",
        [session_path, {}],
    )
    if not reply.body or not reply.unix_fds:
        msg = "InputCapture portal returned no EIS descriptor"
        raise InputCaptureError(msg)
    index = _integer(reply.body[0], "EIS descriptor index")
    try:
        return os.dup(reply.unix_fds[index])
    except (IndexError, OSError) as error:
        msg = "InputCapture portal returned an invalid EIS descriptor"
        raise InputCaptureError(msg) from error


async def _call(
    bus: _Bus,
    interface: str,
    member: str,
    signature: str,
    body: list[object],
) -> Message:
    reply = await bus.call(
        Message(
            destination=PORTAL_DESTINATION,
            path=PORTAL_PATH,
            interface=interface,
            member=member,
            signature=signature,
            body=body,
        )
    )
    if reply is None or reply.message_type is MessageType.ERROR:
        detail = member if reply is None or not reply.body else str(reply.body[0])
        raise InputCaptureError(detail)
    return reply


def _parse_zones(raw: object) -> tuple[tuple[int, int, int, int], ...]:
    value = _unwrap(raw)
    if not isinstance(value, (list, tuple)) or not value:
        msg = "InputCapture portal returned no zones"
        raise InputCaptureError(msg)
    zones: list[tuple[int, int, int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != _ZONE_FIELDS:
            msg = "InputCapture portal returned malformed zones"
            raise InputCaptureError(msg)
        width, height, x, y = (int(part) for part in item)
        if width <= 0 or height <= 0:
            msg = "InputCapture portal returned an invalid zone"
            raise InputCaptureError(msg)
        zones.append((width, height, x, y))
    return tuple(zones)


def _zone_barriers(
    zones: tuple[tuple[int, int, int, int], ...],
) -> list[dict[str, Variant]]:
    positions: list[tuple[int, int, int, int]] = []
    for width, height, x, y in zones:
        horizontal = (
            (
                y,
                (
                    (other_x, other_x + other_width)
                    for other_width, other_height, other_x, other_y in zones
                    if other_y + other_height == y
                ),
            ),
            (
                y + height,
                (
                    (other_x, other_x + other_width)
                    for other_width, _other_height, other_x, other_y in zones
                    if other_y == y + height
                ),
            ),
        )
        for boundary_y, covered in horizontal:
            positions.extend(
                (start, boundary_y, end - 1, boundary_y)
                for start, end in _uncovered_intervals(x, x + width, covered)
            )
        vertical = (
            (
                x,
                (
                    (other_y, other_y + other_height)
                    for other_width, other_height, other_x, other_y in zones
                    if other_x + other_width == x
                ),
            ),
            (
                x + width,
                (
                    (other_y, other_y + other_height)
                    for _other_width, other_height, other_x, other_y in zones
                    if other_x == x + width
                ),
            ),
        )
        for boundary_x, covered in vertical:
            positions.extend(
                (boundary_x, start, boundary_x, end - 1)
                for start, end in _uncovered_intervals(y, y + height, covered)
            )
    return [
        {
            "barrier_id": Variant("u", barrier_id),
            "position": Variant("(iiii)", list(position)),
        }
        for barrier_id, position in enumerate(positions, start=1)
    ]


def _uncovered_intervals(
    start: int,
    end: int,
    covered: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    remaining = [(start, end)]
    for covered_start, covered_end in sorted(covered):
        updated: list[tuple[int, int]] = []
        for current_start, current_end in remaining:
            if covered_end <= current_start or covered_start >= current_end:
                updated.append((current_start, current_end))
                continue
            if covered_start > current_start:
                updated.append((current_start, min(covered_start, current_end)))
            if covered_end < current_end:
                updated.append((max(covered_end, current_start), current_end))
        remaining = updated
    return tuple(item for item in remaining if item[0] < item[1])


async def _close_session(bus: _Bus, session_path: str) -> None:
    with contextlib.suppress(InputCaptureError):
        await _session_close(bus, session_path)


async def _session_close(bus: _Bus, session_path: str) -> None:
    reply = await bus.call(
        Message(
            destination=PORTAL_DESTINATION,
            path=session_path,
            interface=SESSION,
            member="Close",
        )
    )
    if reply is not None and reply.message_type is MessageType.ERROR:
        raise InputCaptureError(str(reply.body[0]) if reply.body else "session close failed")


def _native_library_path() -> Path:
    override = os.environ.get("GAZEEBO_HUD_LIBRARY")
    path = Path(override) if override else Path("/packaged/libgazeebo-hud.so")
    if not path.is_file():
        msg = f"packaged InputCapture helper is unavailable: {path}"
        raise InputCaptureError(msg)
    return path
