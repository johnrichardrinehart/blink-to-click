"""Tests for owner-only warning-triggered false-negative captures."""

from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gazeebo.contracts import EyeObservation, HeadTrackingFailure, RuntimeStatus
from gazeebo.diagnostics import (
    CapturingVisionEstimator,
    DiagnosticArchive,
    DiagnosticArchiveError,
    diagnostic_archive_stats,
    diagnostic_capture_enabled,
    read_diagnostic_frames,
)


class _Status:
    def __init__(self) -> None:
        self.events: list[tuple[RuntimeStatus, str]] = []

    def report(self, status: RuntimeStatus, detail: str = "") -> None:
        self.events.append((status, detail))


class _Vision:
    def __init__(self, results: list[EyeObservation | HeadTrackingFailure]) -> None:
        self.results = results
        self.closed = False

    def observe(
        self,
        _frame: object,
        _timestamp: float,
    ) -> EyeObservation | HeadTrackingFailure:
        return self.results.pop(0)

    def close(self) -> None:
        self.closed = True


def _observation(timestamp: float) -> EyeObservation:
    return EyeObservation(
        timestamp,
        1.0,
        1.0,
        (0.1,),
        0.9,
        (0.2,),
        head_bounds=(0.1, 0.2, 0.3, 0.4),
        head_pose=(1.0, 2.0, 3.0),
        landmarks=((10.0, 20.0), (30.0, 40.0)),
    )


def _frame(value: int) -> np.ndarray:
    return np.full((4, 5, 3), value, dtype=np.uint8)


class DiagnosticArchiveTests(unittest.TestCase):
    """Lock capture boundaries, sensitive storage, and cleanup."""

    def test_warning_writes_exact_lossless_three_second_window_and_metadata(self) -> None:
        """Every detector frame from three seconds before through after is retained once."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostics"
            status = _Status()
            archive = DiagnosticArchive(root, status=status)
            failure = HeadTrackingFailure("no face")
            for timestamp in range(4):
                archive.record(_frame(timestamp), float(timestamp), _observation(float(timestamp)))
            archive.warning(failure, 3.0)
            archive.warning(failure, 4.0)
            for timestamp in range(4, 8):
                archive.record(_frame(timestamp), float(timestamp), _observation(float(timestamp)))
            archive.close()

            events = [path for path in root.iterdir() if path.name.startswith("event-")]
            assert len(events) == 1
            event = events[0]
            metadata = json.loads((event / "metadata.json").read_text())
            assert metadata["schema_version"] == 1
            assert [item["relative_seconds"] for item in metadata["frames"]] == [
                -3.0,
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                3.0,
            ]
            assert len(metadata["warnings"]) == 2
            assert metadata["frames"][0]["confidence"] == 0.9
            assert metadata["frames"][0]["head_bounds"] == [0.1, 0.2, 0.3, 0.4]
            assert metadata["frames"][0]["head_pose"] == [1.0, 2.0, 3.0]
            assert metadata["frames"][0]["landmarks"] == [[10.0, 20.0], [30.0, 40.0]]
            restored = read_diagnostic_frames(event)
            assert len(restored) == 7
            for image, expected in zip(restored, range(7), strict=True):
                assert np.array_equal(image, _frame(expected))
            assert {path.name for path in event.iterdir()} == {
                "frames.gzd",
                "metadata.json",
            }
            assert stat.S_IMODE(root.stat().st_mode) == 0o700
            assert stat.S_IMODE(event.stat().st_mode) == 0o700
            assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in event.iterdir())
            assert not any(path.name.startswith(".event-") for path in root.iterdir())
            assert any(item[0] is RuntimeStatus.DIAGNOSTIC_CAPTURE for item in status.events)

    def test_sequence_encoding_uses_inter_frame_similarity(self) -> None:
        """One lossless delta stream is smaller than repeated raw stationary frames."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostics"
            archive = DiagnosticArchive(root, status=_Status())
            for index in range(30):
                frame = np.full((100, 100, 3), 40 + index % 2, dtype=np.uint8)
                archive.record(frame, index / 10.0, _observation(index / 10.0))
            archive.warning(HeadTrackingFailure("warning"), 2.9)
            archive.close()
            event = next(root.glob("event-*"))
            assert (event / "frames.gzd").stat().st_size < 30 * 100 * 100 * 3 // 10
            restored = read_diagnostic_frames(event)
            assert len(restored) == 30
            assert np.array_equal(restored[-1], np.full((100, 100, 3), 41, dtype=np.uint8))

    def test_quota_preserves_prior_events_and_disables_new_capture(self) -> None:
        """Quota exhaustion never evicts old evidence or leaves a partial event."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostics"
            first = DiagnosticArchive(root, status=_Status())
            first.record(_frame(1), 0.0, _observation(0.0))
            first.warning(HeadTrackingFailure("first"), 0.0)
            first.close()
            before = sorted(path.name for path in root.iterdir())
            used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

            status = _Status()
            second = DiagnosticArchive(root, status=status, maximum_bytes=used + 1)
            second.record(_frame(2), 0.0, _observation(0.0))
            second.warning(HeadTrackingFailure("second"), 0.0)
            second.close()
            assert sorted(path.name for path in root.iterdir()) == before
            assert second.quota_exhausted
            assert any("quota" in detail for _state, detail in status.events)

    def test_reset_is_confined_idempotent_and_rejects_unsafe_paths(self) -> None:
        """Reset removes only validated owner-only archive contents."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostics"
            archive = DiagnosticArchive(root, status=_Status())
            archive.record(_frame(1), 0.0, _observation(0.0))
            archive.warning(HeadTrackingFailure("warning"), 0.0)
            archive.close()
            archive.reset()
            archive.reset()
            assert not root.exists()

            outside = Path(directory) / "outside"
            outside.mkdir()
            root.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(DiagnosticArchiveError, "symlink"):
                DiagnosticArchive(root, status=_Status())
            assert outside.exists()

    def test_inspection_rejects_permissions_metadata_and_sequence_corruption(self) -> None:
        """Read-only inspection fails closed instead of parsing unsafe evidence."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostics"
            archive = DiagnosticArchive(root, status=_Status())
            archive.record(_frame(1), 0.0, _observation(0.0))
            archive.warning(HeadTrackingFailure("warning"), 0.0)
            archive.close()
            event = next(root.glob("event-*"))
            sequence = event / "frames.gzd"
            payload = sequence.read_bytes()
            sequence.write_bytes(payload[:-1])
            sequence.chmod(0o600)
            with self.assertRaisesRegex(DiagnosticArchiveError, "payload"):
                read_diagnostic_frames(event)
            sequence.write_bytes(payload)
            sequence.chmod(0o600)
            metadata = event / "metadata.json"
            metadata.write_text("not-json")
            metadata.chmod(0o600)
            with self.assertRaisesRegex(DiagnosticArchiveError, "malformed"):
                diagnostic_archive_stats(root)
            metadata.chmod(0o644)
            with self.assertRaisesRegex(DiagnosticArchiveError, "owner-only"):
                diagnostic_archive_stats(root)

    def test_wrapper_releases_ring_and_finalizes_interrupted_event(self) -> None:
        """Estimator cleanup drops transient frames and leaves one atomic event."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostics"
            archive = DiagnosticArchive(root, status=_Status())
            vision = _Vision([_observation(0.0), HeadTrackingFailure("warning")])
            wrapped = CapturingVisionEstimator(vision, archive)
            wrapped.observe(_frame(1), 0.0)
            failure = wrapped.observe(_frame(2), 1.0)
            assert isinstance(failure, HeadTrackingFailure)
            wrapped.capture_warning(failure, 1.0)
            assert archive.buffered_frame_count == 2
            wrapped.close()
            assert vision.closed
            assert archive.buffered_frame_count == 0
            assert len(tuple(root.glob("event-*"))) == 1
            assert not tuple(root.glob(".event-*"))

    def test_config_defaults_on_and_cli_overrides_xdg_toml(self) -> None:
        """Default capture is explicit, while CLI has final precedence."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            assert diagnostic_capture_enabled(cli_value=None, path=path)
            path.write_text("[diagnostics]\ncapture = false\n")
            assert not diagnostic_capture_enabled(cli_value=None, path=path)
            assert diagnostic_capture_enabled(cli_value=True, path=path)
            assert not diagnostic_capture_enabled(cli_value=False, path=path)
            path.write_text("[diagnostics]\ncapture = 'invalid'\n")
            with self.assertRaisesRegex(DiagnosticArchiveError, "capture"):
                diagnostic_capture_enabled(cli_value=None, path=path)


if __name__ == "__main__":
    unittest.main()
