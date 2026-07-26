"""Owner-only warning-triggered false-negative frame diagnostics."""

from __future__ import annotations

import json
import math
import os
import secrets
import stat
import struct
import tomllib
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gazeebo.contracts import EyeObservation, HeadTrackingFailure, RuntimeStatus

if TYPE_CHECKING:
    from gazeebo.contracts import Frame, StatusSink, VisionEstimator

ARCHIVE_SCHEMA_VERSION = 1
DEFAULT_MAXIMUM_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_WINDOW_SECONDS = 3.0
MAXIMUM_RING_FRAMES = 1000
MAXIMUM_EVENT_RAW_BYTES = 1024 * 1024 * 1024
MAXIMUM_FRAME_HEADER_BYTES = 65536
_FRAME_MAGIC = b"GAZEEBO-FRAMES\x00"
_FRAME_ENCODING = "xor-gzip-v1"
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600


class DiagnosticArchiveError(RuntimeError):
    """The sensitive diagnostic archive is unsafe or unavailable."""


@dataclass(frozen=True, slots=True)
class DiagnosticArchiveStats:
    """Bounded read-only archive accounting."""

    schema_version: int
    event_count: int
    on_disk_bytes: int
    maximum_bytes: int


@dataclass(frozen=True, slots=True)
class _CapturedFrame:
    timestamp: float
    pixels: np.ndarray
    metadata: dict[str, object]


@dataclass(slots=True)
class _ActiveEvent:
    event_id: str
    trigger: float
    deadline: float
    frames: list[_CapturedFrame]
    warnings: list[dict[str, object]]


def _default_archive_path() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return root / "gazeebo" / "diagnostics-v1"


def _default_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home) if config_home else Path.home() / ".config"
    return root / "gazeebo" / "config.toml"


def diagnostic_capture_enabled(
    *,
    cli_value: bool | None,
    path: Path | None = None,
) -> bool:
    """Resolve CLI-over-config capture policy with a privacy-visible default."""
    if cli_value is not None:
        return cli_value
    config = path or _default_config_path()
    if not config.exists():
        return True
    if config.is_symlink() or not config.is_file():
        msg = "diagnostic configuration must be a regular file"
        raise DiagnosticArchiveError(msg)
    try:
        raw = tomllib.loads(config.read_text())
        diagnostics = raw.get("diagnostics", {})
        value = diagnostics.get("capture", True)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, AttributeError) as error:
        msg = "diagnostic configuration is malformed"
        raise DiagnosticArchiveError(msg) from error
    if not isinstance(value, bool):
        msg = "diagnostic capture configuration must be true or false"
        raise DiagnosticArchiveError(msg)
    return value


def _metadata(result: EyeObservation | HeadTrackingFailure) -> dict[str, object]:
    if isinstance(result, EyeObservation):
        reason: str | None = None
        confidence: float | None = result.confidence
        bounds = result.head_bounds
        pose = result.head_pose
        landmarks = result.landmarks
    else:
        reason = result.reason
        confidence = result.confidence
        bounds = result.head_bounds
        pose = result.head_pose
        landmarks = result.landmarks
    return {
        "reason": reason,
        "confidence": confidence,
        "head_bounds": None if bounds is None else list(bounds),
        "head_pose": None if pose is None else list(pose),
        "landmarks": None if landmarks is None else [list(point) for point in landmarks],
    }


def _encode_frame_sequence(frames: list[_CapturedFrame]) -> bytes:
    """Compress one exact first frame plus inter-frame XOR deltas."""
    if not frames:
        msg = "diagnostic event must contain at least one frame"
        raise DiagnosticArchiveError(msg)
    first = frames[0].pixels
    if first.dtype != np.uint8 or first.ndim not in {2, 3}:
        msg = "diagnostic frames must be uint8 images"
        raise DiagnosticArchiveError(msg)
    if any(
        frame.pixels.shape != first.shape or frame.pixels.dtype != first.dtype for frame in frames
    ):
        msg = "diagnostic event frames must have one shape and type"
        raise DiagnosticArchiveError(msg)
    raw_size = first.nbytes * len(frames)
    if raw_size > MAXIMUM_EVENT_RAW_BYTES:
        msg = "diagnostic event exceeds its raw frame limit"
        raise DiagnosticArchiveError(msg)
    header = json.dumps(
        {
            "encoding": _FRAME_ENCODING,
            "shape": list(first.shape),
            "dtype": str(first.dtype),
            "count": len(frames),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    compressor = zlib.compressobj(level=9, wbits=31)
    compressed = bytearray(compressor.compress(first.tobytes(order="C")))
    previous = first
    for frame in frames[1:]:
        delta = np.bitwise_xor(frame.pixels, previous)
        compressed.extend(compressor.compress(delta.tobytes(order="C")))
        previous = frame.pixels
    compressed.extend(compressor.flush())
    return _FRAME_MAGIC + struct.pack(">I", len(header)) + header + bytes(compressed)


def _decode_frame_sequence(payload: bytes) -> tuple[np.ndarray, ...]:
    """Validate and reconstruct one exact diagnostic frame sequence."""
    prefix = len(_FRAME_MAGIC) + 4
    if len(payload) < prefix or not payload.startswith(_FRAME_MAGIC):
        msg = "diagnostic frame sequence header is invalid"
        raise DiagnosticArchiveError(msg)
    header_size = struct.unpack(">I", payload[len(_FRAME_MAGIC) : prefix])[0]
    if (
        header_size <= 0
        or header_size > MAXIMUM_FRAME_HEADER_BYTES
        or prefix + header_size > len(payload)
    ):
        msg = "diagnostic frame sequence header is truncated"
        raise DiagnosticArchiveError(msg)
    try:
        header = json.loads(payload[prefix : prefix + header_size])
        shape = tuple(int(value) for value in header["shape"])
        count = int(header["count"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        msg = "diagnostic frame sequence header is malformed"
        raise DiagnosticArchiveError(msg) from error
    if (
        header.get("encoding") != _FRAME_ENCODING
        or header.get("dtype") != "uint8"
        or len(shape) not in {2, 3}
        or any(value <= 0 for value in shape)
        or count <= 0
        or count > MAXIMUM_RING_FRAMES * 2
    ):
        msg = "diagnostic frame sequence dimensions are invalid"
        raise DiagnosticArchiveError(msg)
    frame_size = math.prod(shape)
    expected = frame_size * count
    if expected > MAXIMUM_EVENT_RAW_BYTES:
        msg = "diagnostic frame sequence exceeds its raw limit"
        raise DiagnosticArchiveError(msg)
    decompressor = zlib.decompressobj(wbits=31)
    raw = decompressor.decompress(payload[prefix + header_size :], expected + 1)
    if len(raw) > expected or decompressor.unconsumed_tail:
        msg = "diagnostic frame sequence expands beyond its declared size"
        raise DiagnosticArchiveError(msg)
    raw += decompressor.flush(max(1, expected + 1 - len(raw)))
    if len(raw) != expected or not decompressor.eof or decompressor.unused_data:
        msg = "diagnostic frame sequence payload is invalid"
        raise DiagnosticArchiveError(msg)
    encoded = np.frombuffer(raw, dtype=np.uint8).reshape((count, *shape))
    result = [encoded[0].copy()]
    for delta in encoded[1:]:
        result.append(np.bitwise_xor(delta, result[-1]))
    return tuple(result)


def read_diagnostic_frames(event: Path) -> tuple[np.ndarray, ...]:
    """Read one validated local event sequence for explicit inspection."""
    _validate_directory(event, "diagnostic event")
    sequence = event / "frames.gzd"
    _validate_file(sequence)
    if sequence.stat().st_size > DEFAULT_MAXIMUM_BYTES:
        msg = "diagnostic frame sequence exceeds its encoded limit"
        raise DiagnosticArchiveError(msg)
    return _decode_frame_sequence(sequence.read_bytes())


def _write_private(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(path, flags, _FILE_MODE)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


class DiagnosticArchive:
    """Buffer a bounded frame window and atomically retain warning events."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        status: StatusSink,
        maximum_bytes: int = DEFAULT_MAXIMUM_BYTES,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        """Validate bounds and open one safe archive ownership scope."""
        if maximum_bytes <= 0 or not math.isfinite(window_seconds) or window_seconds <= 0.0:
            msg = "diagnostic archive bounds must be finite and positive"
            raise ValueError(msg)
        self.path = path or _default_archive_path()
        self._status = status
        self.maximum_bytes = maximum_bytes
        self.window_seconds = window_seconds
        self._ring: deque[_CapturedFrame] = deque(maxlen=MAXIMUM_RING_FRAMES)
        self._active: _ActiveEvent | None = None
        self._session_id = secrets.token_hex(8)
        self._next_event = 0
        self._closed = False
        self.quota_exhausted = False
        self._ensure_directory()
        self._used_bytes, _event_count = self._inspect()
        if self._used_bytes >= self.maximum_bytes:
            self._set_quota_exhausted()

    @property
    def buffered_frame_count(self) -> int:
        """Expose bounded transient ownership for cleanup verification."""
        frames = {id(frame) for frame in self._ring}
        if self._active is not None:
            frames.update(id(frame) for frame in self._active.frames)
        return len(frames)

    def record(
        self,
        frame: Frame,
        timestamp: float,
        result: EyeObservation | HeadTrackingFailure,
    ) -> None:
        """Retain one detector input in RAM and any active warning event."""
        if self._closed or self.quota_exhausted:
            return
        if not math.isfinite(timestamp) or not isinstance(frame, np.ndarray):
            msg = "diagnostic frame and timestamp are invalid"
            raise DiagnosticArchiveError(msg)
        captured = _CapturedFrame(timestamp, frame.copy(), _metadata(result))
        if self._active is not None and timestamp > self._active.deadline:
            self._finalize(complete=True)
        if self._active is not None:
            self._active.frames.append(captured)
        self._ring.append(captured)
        cutoff = timestamp - self.window_seconds
        while self._ring and self._ring[0].timestamp < cutoff:
            self._ring.popleft()

    def warning(self, failure: HeadTrackingFailure, timestamp: float) -> None:
        """Start or coalesce one sustained-warning capture window."""
        if self._closed or self.quota_exhausted:
            return
        if not math.isfinite(timestamp):
            msg = "diagnostic warning timestamp is invalid"
            raise DiagnosticArchiveError(msg)
        warning = {"relative_seconds": 0.0, **_metadata(failure)}
        if self._active is not None and timestamp <= self._active.deadline:
            warning["relative_seconds"] = timestamp - self._active.trigger
            self._active.warnings.append(warning)
            return
        if self._active is not None:
            self._finalize(complete=True)
        self._next_event += 1
        event_id = f"{self._session_id}-{self._next_event:04d}"
        frames = [
            item
            for item in self._ring
            if timestamp - self.window_seconds <= item.timestamp <= timestamp
        ]
        self._active = _ActiveEvent(
            event_id,
            timestamp,
            timestamp + self.window_seconds,
            frames,
            [warning],
        )

    def close(self) -> None:
        """Atomically finish available evidence and release every frame reference."""
        if self._closed:
            return
        try:
            if self._active is not None:
                complete = bool(
                    self._active.frames
                    and self._active.frames[-1].timestamp >= self._active.deadline
                )
                self._finalize(complete=complete)
        finally:
            self._active = None
            self._ring.clear()
            self._closed = True

    def reset(self) -> None:
        """Remove only a validated archive tree and remain idempotent."""
        self.close()
        reset_diagnostic_archive(self.path)

    def _ensure_directory(self) -> None:
        try:
            if self.path.is_symlink():
                msg = "diagnostic archive must not be a symlink"
                raise DiagnosticArchiveError(msg)
            self.path.parent.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
            if self.path.parent.is_symlink():
                msg = "diagnostic archive parent must not be a symlink"
                raise DiagnosticArchiveError(msg)
            if not self.path.exists():
                self.path.mkdir(mode=_DIRECTORY_MODE)
        except OSError as error:
            msg = "could not create diagnostic archive"
            raise DiagnosticArchiveError(msg) from error
        _validate_directory(self.path, "diagnostic archive")

    def _inspect(self) -> tuple[int, int]:
        used = 0
        events = 0
        for event in self.path.iterdir():
            if event.is_symlink() or not event.name.startswith("event-"):
                msg = "diagnostic archive contains an unsafe entry"
                raise DiagnosticArchiveError(msg)
            _validate_directory(event, "diagnostic event")
            events += 1
            names = {item.name for item in event.iterdir()}
            if names != {"metadata.json", "frames.gzd"}:
                msg = "diagnostic event files are incomplete or unexpected"
                raise DiagnosticArchiveError(msg)
            for item in event.iterdir():
                _validate_file(item)
                used += item.stat().st_size
            _validate_metadata(event / "metadata.json")
        return used, events

    def _finalize(self, *, complete: bool) -> None:
        event = self._active
        if event is None:
            return
        frame_metadata: list[dict[str, object]] = []
        try:
            sequence = _encode_frame_sequence(event.frames)
            for index, captured in enumerate(event.frames):
                frame_metadata.append(
                    {
                        "sequence_index": index,
                        "relative_seconds": captured.timestamp - event.trigger,
                        **captured.metadata,
                    }
                )
            metadata = json.dumps(
                {
                    "schema_version": ARCHIVE_SCHEMA_VERSION,
                    "event_id": event.event_id,
                    "complete": complete,
                    "window_seconds": self.window_seconds,
                    "frame_encoding": _FRAME_ENCODING,
                    "warnings": event.warnings,
                    "frames": frame_metadata,
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        except (TypeError, ValueError) as error:
            msg = "diagnostic event metadata is invalid"
            raise DiagnosticArchiveError(msg) from error
        event_size = len(metadata) + len(sequence)
        if self._used_bytes + event_size > self.maximum_bytes:
            self._active = None
            self._ring.clear()
            self._set_quota_exhausted()
            return
        final = self.path / f"event-{event.event_id}"
        temporary = self.path / f".event-{event.event_id}"
        try:
            temporary.mkdir(mode=_DIRECTORY_MODE)
            _write_private(temporary / "frames.gzd", sequence)
            _write_private(temporary / "metadata.json", metadata)
            directory = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            temporary.replace(final)
            parent = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        except OSError as error:
            _remove_temporary(temporary)
            msg = "could not atomically write diagnostic event"
            raise DiagnosticArchiveError(msg) from error
        self._used_bytes += event_size
        self._active = None
        self._status.report(
            RuntimeStatus.DIAGNOSTIC_CAPTURE,
            f"retained {len(event.frames)} warning-adjacent frames",
        )

    def _set_quota_exhausted(self) -> None:
        if self.quota_exhausted:
            return
        self.quota_exhausted = True
        self._active = None
        self._ring.clear()
        self._status.report(
            RuntimeStatus.DIAGNOSTIC_CAPTURE,
            "2 GiB quota reached; new warning captures are disabled until reset",
        )


class CapturingVisionEstimator:
    """Decorate one estimator without changing inference or target data."""

    def __init__(self, estimator: VisionEstimator, archive: DiagnosticArchive) -> None:
        """Bind capture lifecycle to the wrapped estimator lifecycle."""
        self._estimator = estimator
        self._archive = archive
        self._closed = False

    def observe(
        self,
        frame: Frame,
        timestamp: float,
    ) -> EyeObservation | HeadTrackingFailure:
        """Run inference unchanged and retain only bounded diagnostic evidence."""
        result = self._estimator.observe(frame, timestamp)
        self._archive.record(frame, timestamp, result)
        return result

    def capture_warning(self, failure: HeadTrackingFailure, timestamp: float) -> None:
        """Mark the warning after recovery confirms it is sustained."""
        self._archive.warning(failure, timestamp)

    def close(self) -> None:
        """Finalize capture and release wrapped inference exactly once."""
        if self._closed:
            return
        try:
            self._archive.close()
        finally:
            self._estimator.close()
            self._closed = True


def diagnostic_archive_stats(path: Path | None = None) -> DiagnosticArchiveStats:
    """Inspect safe completed events without creating or mutating the archive."""
    archive = path or _default_archive_path()
    if not archive.exists():
        return DiagnosticArchiveStats(
            ARCHIVE_SCHEMA_VERSION,
            0,
            0,
            DEFAULT_MAXIMUM_BYTES,
        )
    _validate_directory(archive, "diagnostic archive")
    reader = DiagnosticArchive.__new__(DiagnosticArchive)
    reader.path = archive
    used, events = reader._inspect()  # noqa: SLF001
    return DiagnosticArchiveStats(
        ARCHIVE_SCHEMA_VERSION,
        events,
        used,
        DEFAULT_MAXIMUM_BYTES,
    )


def reset_diagnostic_archive(path: Path | None = None) -> None:
    """Remove a safe archive tree without following any link."""
    archive = path or _default_archive_path()
    if not archive.exists() and not archive.is_symlink():
        return
    _validate_directory(archive, "diagnostic archive")
    for event in archive.iterdir():
        if event.is_symlink() or not event.name.startswith("event-"):
            msg = "diagnostic archive contains an unsafe entry"
            raise DiagnosticArchiveError(msg)
        _validate_directory(event, "diagnostic event")
        for item in event.iterdir():
            _validate_file(item)
            item.unlink()
        event.rmdir()
    archive.rmdir()


def _validate_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        msg = f"{label} is unavailable"
        raise DiagnosticArchiveError(msg) from error
    if stat.S_ISLNK(metadata.st_mode):
        msg = f"{label} must not be a symlink"
        raise DiagnosticArchiveError(msg)
    if not stat.S_ISDIR(metadata.st_mode):
        msg = f"{label} must be a directory"
        raise DiagnosticArchiveError(msg)
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != _DIRECTORY_MODE:
        msg = f"{label} must be owner-only"
        raise DiagnosticArchiveError(msg)


def _validate_file(path: Path) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        msg = "diagnostic event files must be regular and not symlinks"
        raise DiagnosticArchiveError(msg)
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != _FILE_MODE:
        msg = "diagnostic event files must be owner-only"
        raise DiagnosticArchiveError(msg)


def _validate_metadata(path: Path) -> None:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        msg = "diagnostic event metadata is malformed"
        raise DiagnosticArchiveError(msg) from error
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != ARCHIVE_SCHEMA_VERSION
        or raw.get("frame_encoding") != _FRAME_ENCODING
        or not isinstance(raw.get("frames"), list)
        or not isinstance(raw.get("warnings"), list)
    ):
        msg = "diagnostic event metadata is invalid"
        raise DiagnosticArchiveError(msg)


def _remove_temporary(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        item.unlink()
    path.rmdir()
