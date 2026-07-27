"""Owner-only runtime control for on-demand training transitions."""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path

_DIRECTORY_MODE = 0o700
_SOCKET_MODE = 0o600
_MAXIMUM_COMMAND_BYTES = 256
_MAXIMUM_PENDING_COMMANDS = 64
_CELL_FIELD_COUNT = 2
_MOTION_FIELD_COUNT = 3
_REFINEMENT_COMMANDS = {"refine", "accept", "cancel", "capture"}


class ControlError(RuntimeError):
    """The local training-control channel is unsafe or unavailable."""


@dataclass(frozen=True, slots=True)
class ControlCommand:
    """One validated process-local cursor-refinement request."""

    kind: str
    values: tuple[float, ...] = ()
    label: str = ""


class TrainingControl:
    """Own one process-scoped Unix socket for training and refinement."""

    def __init__(
        self,
        training_requested: asyncio.Event,
        path: Path | None = None,
        command_queue: asyncio.Queue[ControlCommand] | None = None,
    ) -> None:
        """Bind training and refinement requests to one testable socket."""
        self.training_requested = training_requested
        self.command_queue = command_queue or asyncio.Queue(_MAXIMUM_PENDING_COMMANDS)
        self.path = path or control_path()
        self._server: asyncio.Server | None = None
        self._owns_path = False

    async def start(self) -> None:
        """Create one owner-only socket for the foreground process."""
        directory = self.path.parent
        if not directory.exists():
            directory.mkdir(mode=_DIRECTORY_MODE, parents=True)
            directory.chmod(_DIRECTORY_MODE)
        _validate_directory(directory)
        if self.path.exists() or self.path.is_symlink():
            msg = "another Gazeebo control socket already exists"
            raise ControlError(msg)
        try:
            self._server = await asyncio.start_unix_server(
                self._handle,
                path=self.path,
            )
            self._owns_path = True
            self.path.chmod(_SOCKET_MODE)
            _validate_socket(self.path)
        except OSError as error:
            msg = "could not create the training control socket"
            raise ControlError(msg) from error

    async def close(self) -> None:
        """Close the server and remove its socket idempotently."""
        server, self._server = self._server, None
        owns_path, self._owns_path = self._owns_path, False
        if server is not None:
            server.close()
            await server.wait_closed()
        if owns_path and (self.path.exists() or self.path.is_symlink()):
            try:
                _validate_socket(self.path)
                self.path.unlink()
            except FileNotFoundError:
                pass
        with contextlib.suppress(OSError):
            self.path.parent.rmdir()

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw = await asyncio.wait_for(reader.readline(), 2.0)
            if len(raw) > _MAXIMUM_COMMAND_BYTES:
                writer.write(b"rejected\n")
            elif raw == b"train\n":
                self.training_requested.set()
                writer.write(b"accepted\n")
            else:
                command = _parse_command(raw)
                if command is None or self.command_queue.full():
                    writer.write(b"rejected\n")
                else:
                    self.command_queue.put_nowait(command)
                    writer.write(b"accepted\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


async def request_training(path: Path | None = None) -> bool:
    """Request training from an existing process, returning false when absent."""
    return await request_command("train", path)


async def request_command(command: str, path: Path | None = None) -> bool:
    """Send one bounded command to an existing foreground owner."""
    encoded = f"{command}\n".encode()
    if len(encoded) > _MAXIMUM_COMMAND_BYTES or _parse_client_command(command) is None:
        msg = "Gazeebo control command is invalid"
        raise ControlError(msg)
    target = path or control_path()
    if not target.exists() and not target.is_symlink():
        return False
    _validate_socket(target)
    try:
        reader, writer = await asyncio.open_unix_connection(target)
    except (ConnectionRefusedError, FileNotFoundError):
        _validate_socket(target)
        target.unlink()
        return False
    try:
        writer.write(encoded)
        await writer.drain()
        response = await asyncio.wait_for(reader.readline(), 2.0)
        return response == b"accepted\n"
    finally:
        writer.close()
        await writer.wait_closed()


def _parse_client_command(command: str) -> ControlCommand | str | None:
    if command == "train":
        return command
    return _parse_command(f"{command}\n".encode())


def _parse_command(raw: bytes) -> ControlCommand | None:  # noqa: PLR0911
    try:
        fields = raw.decode("ascii").strip().split()
    except UnicodeDecodeError:
        return None
    if not fields:
        return None
    kind = fields[0]
    if kind in _REFINEMENT_COMMANDS and len(fields) == 1:
        return ControlCommand(kind)
    if kind == "cell" and len(fields) == _CELL_FIELD_COUNT:
        label = fields[1]
        if len(label) == 1 and label.isascii() and label.isprintable() and not label.isspace():
            return ControlCommand(kind, label=label)
        return None
    if kind in {"move", "position"} and len(fields) == _MOTION_FIELD_COUNT:
        try:
            values = tuple(float(value) for value in fields[1:])
        except ValueError:
            return None
        if all(math.isfinite(value) for value in values):
            return ControlCommand(kind, values)
    return None


def control_path() -> Path:
    """Return the standard process-scoped runtime socket path."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        msg = "XDG_RUNTIME_DIR is required for on-demand training control"
        raise ControlError(msg)
    return Path(runtime) / "gazeebo" / "control.sock"


def _validate_directory(path: Path) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        msg = "training control directory must be owner-only"
        raise ControlError(msg)


def _validate_socket(path: Path) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        msg = "training control path must be an owner-only socket"
        raise ControlError(msg)
