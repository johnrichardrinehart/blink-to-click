"""Camera-free tests for local capture behavior."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from gazeebo.camera import CameraError, OpenCVCamera


@dataclass(slots=True)
class StubCapture:
    """Model one OpenCV capture without a device."""

    opened: bool
    frames: list[object]
    released: bool = False
    settings: list[tuple[int, float]] = field(default_factory=list)

    def isOpened(self) -> bool:  # noqa: N802
        """Return configured open state."""
        return self.opened

    def read(self) -> tuple[bool, object]:
        """Return one frame or an explicit read failure."""
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        """Record device release."""
        self.released = True

    def set(self, property_id: int, value: float) -> bool:
        """Record optional camera settings."""
        self.settings.append((property_id, value))
        return True


@dataclass(slots=True)
class StubCV2:
    """Return configured captures in source order."""

    captures: list[StubCapture]
    CAP_V4L2: int = 1
    CAP_PROP_FRAME_WIDTH: int = 2
    CAP_PROP_FRAME_HEIGHT: int = 3
    CAP_PROP_FPS: int = 4
    CAP_PROP_FOURCC: int = 5
    opened_sources: list[str | int] = field(default_factory=list)

    def VideoCapture(self, source: str | int, backend: int) -> StubCapture:  # noqa: N802
        """Return the next capture and record the attempted source."""
        assert backend == self.CAP_V4L2
        self.opened_sources.append(source)
        return self.captures.pop(0)

    def VideoWriter_fourcc(self, *codec: str) -> int:  # noqa: N802
        """Return one deterministic test codec identifier."""
        assert codec == ("M", "J", "P", "G")
        return 99


class CameraTests(unittest.TestCase):
    """Lock source probing and cleanup without webcam access."""

    def test_probe_releases_failure_and_keeps_first_frame(self) -> None:
        """Automatic probing owns only the first source that returns a frame."""
        failed = StubCapture(False, [])
        first_frame = object()
        second_frame = object()
        working = StubCapture(True, [first_frame, second_frame])
        cv2 = StubCV2([failed, working])

        camera = OpenCVCamera(candidates=("/dev/a", "/dev/b"), cv2_module=cv2)
        assert failed.released
        assert cv2.opened_sources == ["/dev/a", "/dev/b"]
        assert camera.read() is first_frame
        assert camera.read() is second_frame
        camera.close()
        camera.close()
        assert working.released

    def test_explicit_codec_precedes_geometry_and_isolates_camera_identity(self) -> None:
        """Transport changes cannot silently reuse another calibration corpus."""
        compressed = StubCapture(True, [object()])
        camera = OpenCVCamera(
            device=0,
            width=848,
            height=480,
            frames_per_second=30,
            codec="MJPG",
            cv2_module=StubCV2([compressed]),
        )
        assert compressed.settings == [(5, 99.0), (2, 848.0), (3, 480.0), (4, 30.0)]
        compressed_id = camera.camera_id
        camera.close()
        ordinary = StubCapture(True, [object()])
        camera = OpenCVCamera(device=0, cv2_module=StubCV2([ordinary]))
        assert camera.camera_id != compressed_id
        camera.close()
        with self.assertRaisesRegex(ValueError, "four-character"):
            OpenCVCamera(device=0, codec="JPG", cv2_module=StubCV2([]))

    def test_no_frame_fails_after_releasing_every_candidate(self) -> None:
        """No-camera startup fails safely without retaining a capture."""
        first = StubCapture(True, [])
        second = StubCapture(False, [])
        cv2 = StubCV2([first, second])
        with self.assertRaisesRegex(CameraError, "no local camera"):
            OpenCVCamera(candidates=(0, 1), cv2_module=cv2)
        assert first.released
        assert second.released

    def test_read_failure_does_not_reopen_or_hide_error(self) -> None:
        """A camera that disappears produces a clear terminal error."""
        capture = StubCapture(True, [object()])
        camera = OpenCVCamera(device=0, cv2_module=StubCV2([capture]))
        camera.read()
        with self.assertRaisesRegex(CameraError, "stopped returning"):
            camera.read()
        camera.close()
        assert capture.released


if __name__ == "__main__":
    unittest.main()
