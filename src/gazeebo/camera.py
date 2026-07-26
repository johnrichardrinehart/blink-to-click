"""Local webcam capture through OpenCV and V4L2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast

import cv2  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gazeebo.contracts import Frame


FRAME_DIMENSIONS = 2
FOURCC_LENGTH = 4


class CameraError(RuntimeError):
    """A local capture device could not provide a usable frame."""


class _Capture(Protocol):
    def isOpened(self) -> bool:  # noqa: N802
        """Return whether the capture backend opened."""

    def read(self) -> tuple[bool, Frame]:
        """Read one frame."""

    def release(self) -> None:
        """Release the device."""

    def set(self, property_id: int, value: float) -> bool:
        """Set one optional capture property."""


class _CV2(Protocol):
    CAP_V4L2: int
    CAP_PROP_FRAME_WIDTH: int
    CAP_PROP_FRAME_HEIGHT: int
    CAP_PROP_FPS: int
    CAP_PROP_FOURCC: int

    def VideoCapture(self, source: str | int, backend: int) -> _Capture:  # noqa: N802
        """Open a capture source."""

    def VideoWriter_fourcc(self, *codec: str) -> int:  # noqa: N802
        """Encode a four-character capture format."""


def local_camera_candidates() -> tuple[str | int, ...]:
    """List stable V4L2 capture paths before volatile numeric devices."""
    stable = sorted(str(path) for path in Path("/dev/v4l/by-path").glob("*-video-index0"))
    numeric = sorted(str(path) for path in Path("/dev").glob("video*"))
    seen: set[str] = set()
    candidates: list[str | int] = []
    for path in (*stable, *numeric):
        if path not in seen:
            candidates.append(path)
            seen.add(path)
    if not candidates:
        candidates.append(0)
    return tuple(candidates)


class OpenCVCamera:
    """Own one V4L2 camera and retain no frames after the caller releases them."""

    def __init__(  # noqa: PLR0913
        self,
        device: str | int | None = None,
        *,
        width: int = 640,
        height: int = 480,
        frames_per_second: int = 30,
        codec: str | None = None,
        candidates: Iterable[str | int] | None = None,
        cv2_module: _CV2 | None = None,
    ) -> None:
        """Open an explicit device or the first local source that returns a frame."""
        if width <= 0 or height <= 0 or frames_per_second <= 0:
            msg = "camera dimensions and frame rate must be positive"
            raise ValueError(msg)
        if codec is not None and (len(codec) != FOURCC_LENGTH or not codec.isascii()):
            msg = "camera codec must be a four-character ASCII code"
            raise ValueError(msg)
        if cv2_module is None:
            cv2_module = cast("_CV2", cv2)
        self._cv2 = cv2_module
        self._capture: _Capture | None = None
        self._pending_frame: Frame | None = None
        self._dimensions = (width, height)
        self._camera_id = ""
        sources = (
            (device,) if device is not None else tuple(candidates or local_camera_candidates())
        )
        for source in sources:
            capture = cv2_module.VideoCapture(source, cv2_module.CAP_V4L2)
            if codec is not None:
                capture.set(
                    cv2_module.CAP_PROP_FOURCC,
                    float(cv2_module.VideoWriter_fourcc(*codec)),
                )
            capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, float(width))
            capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, float(height))
            capture.set(cv2_module.CAP_PROP_FPS, float(frames_per_second))
            opened = capture.isOpened()
            success, frame = capture.read() if opened else (False, None)
            if opened and success and frame is not None:
                self._capture = capture
                self._pending_frame = frame
                shape = getattr(frame, "shape", ())
                if isinstance(shape, tuple) and len(shape) >= FRAME_DIMENSIONS:
                    self._dimensions = (int(shape[1]), int(shape[0]))
                identity = f"{source!s}:{self._dimensions[0]}x{self._dimensions[1]}"
                if codec is not None:
                    identity = f"{identity}:{codec}"
                self._camera_id = hashlib.sha256(identity.encode()).hexdigest()
                break
            capture.release()
        if self._capture is None:
            msg = "no local camera returned a frame"
            raise CameraError(msg)

    @property
    def camera_id(self) -> str:
        """Return an opaque fingerprint for automatic context compatibility."""
        return self._camera_id

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return the actual first-frame width and height when available."""
        return self._dimensions

    def read(self) -> Frame:
        """Return the next in-memory frame or fail without reopening the device."""
        if self._capture is None:
            msg = "camera is closed"
            raise CameraError(msg)
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return frame
        success, frame = self._capture.read()
        if not success or frame is None:
            msg = "camera stopped returning frames"
            raise CameraError(msg)
        return frame

    def close(self) -> None:
        """Release the camera idempotently and discard the pending frame."""
        capture, self._capture = self._capture, None
        self._pending_frame = None
        if capture is not None:
            capture.release()

    def __enter__(self) -> Self:
        """Return this owned camera."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release the camera on context exit."""
        self.close()
