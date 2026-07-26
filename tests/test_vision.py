"""Model-free tests for vision feature extraction."""

from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np

from gazeebo.contracts import EyeObservation, HeadTrackingFailure
from gazeebo.vision import (
    OpenSeeFaceEstimator,
    VisionError,
    _angle_feature,
    _default_tracker_factory,
    _normalized_pupil,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(slots=True)
class StubFace:
    """Expose the OpenSeeFace fields consumed by the adapter."""

    conf: float
    eye_state: Sequence[Sequence[float]] | None
    euler: Sequence[float] | None
    lms: Sequence[Sequence[float]] | None


@dataclass(slots=True)
class StubTracker:
    """Return a fixed face list."""

    faces: Sequence[StubFace]

    def predict(self, frame: object) -> Sequence[StubFace]:
        """Return configured observations."""
        del frame
        return self.faces


def landmarks(
    *,
    right_eye_height: float = 3.0,
    left_eye_height: float = 3.0,
) -> list[list[float]]:
    """Create non-degenerate landmarks with independently shaped eyes."""
    # OpenSeeFace's landmark arrays use (row/y, column/x, confidence).
    points = [[20.0 + 60.0 * index / 65.0, 10.0 + 100.0 * index / 65.0, 1.0] for index in range(66)]

    def set_eye(start: int, center_x: float, height: float) -> None:
        points[start : start + 6] = [
            [50.0, center_x - 10.0, 1.0],
            [50.0 - height, center_x - 5.0, 1.0],
            [50.0 - height, center_x + 5.0, 1.0],
            [50.0, center_x + 10.0, 1.0],
            [50.0 + height, center_x + 5.0, 1.0],
            [50.0 + height, center_x - 5.0, 1.0],
        ]

    set_eye(36, 40.0, right_eye_height)
    set_eye(42, 80.0, left_eye_height)
    return points


def tracker_factory(faces: Sequence[StubFace]) -> Callable[[int, int], StubTracker]:
    """Bind a typed tracker result to the factory contract."""

    def create(_width: int, _height: int) -> StubTracker:
        return StubTracker(faces)

    return create


class VisionTests(unittest.TestCase):
    """Lock eye anatomy, confidence, and feature normalization."""

    def test_default_tracker_uses_robust_profile_face_detection(self) -> None:
        """The packaged tracker tries high-quality full-frame recovery at steep yaw."""
        captured: dict[str, object] = {}

        def tracker(width: int, height: int, **options: object) -> StubTracker:
            captured.update({"width": width, "height": height, **options})
            return StubTracker(())

        module = types.ModuleType("tracker")
        module.Tracker = tracker  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"tracker": module}):
            _default_tracker_factory(640, 480)
        assert captured["model_type"] == 3
        assert captured["detection_threshold"] == 0.2
        assert captured["threshold"] == 0.6
        assert captured["discard_after"] == 5
        assert captured["use_retinaface"] is False
        assert captured["try_hard"] is True

    def test_profile_confidence_remains_usable_with_valid_head_geometry(self) -> None:
        """A clear side profile is not rejected only for lower landmark confidence."""
        face = StubFace(
            conf=0.30,
            eye_state=None,
            euler=(0.0, 65.0, 0.0),
            lms=landmarks(),
        )
        observation = OpenSeeFaceEstimator(
            200,
            100,
            tracker_factory=tracker_factory((face,)),
        ).observe(np.zeros((100, 200, 3), dtype=np.uint8), 1.0)
        assert isinstance(observation, EyeObservation)
        assert observation.confidence == 0.30
        assert not observation.pupil_available

    def test_euler_wraparound_does_not_create_a_feature_jump(self) -> None:
        """Equivalent near-180-degree poses remain adjacent model inputs."""
        assert abs(_angle_feature(179.0) - _angle_feature(-179.0)) < 0.04

    def test_pupil_coordinates_are_local_to_each_eye_aperture(self) -> None:
        """Head translation and face size do not masquerade as pupil motion."""
        points = np.asarray(landmarks(), dtype=np.float64)[:, :2][:, [1, 0]]
        normalized = _normalized_pupil(points, (36, 37, 38, 39, 40, 41), 45.0, 50.0)
        shifted = points + np.asarray((17.0, 9.0))
        shifted_normalized = _normalized_pupil(
            shifted,
            (36, 37, 38, 39, 40, 41),
            62.0,
            59.0,
        )
        assert normalized is not None
        self.assertAlmostEqual(normalized[0], 0.75)
        self.assertAlmostEqual(normalized[1], 0.5)
        assert shifted_normalized == normalized

    def test_closed_eye_keeps_reliable_head_features_without_pupil_evidence(self) -> None:
        """Closed-eye evidence is optional while head/face geometry remains usable."""
        face = StubFace(
            conf=0.92,
            eye_state=((0.2, 50.0, 40.0, 0.90), (0.8, 50.0, 80.0, 0.95)),
            euler=(0.1, -0.2, 0.0),
            lms=landmarks(right_eye_height=0.8),
        )
        tracker = StubTracker((face,))
        estimator = OpenSeeFaceEstimator(
            200,
            100,
            tracker_factory=lambda _width, _height: tracker,
        )

        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        observation = estimator.observe(frame, 12.5)
        assert isinstance(observation, EyeObservation)
        assert observation.timestamp == 12.5
        assert observation.left_open > 0.8
        assert observation.right_open < 0.2
        assert observation.confidence == 0.92
        assert not observation.pupil_available
        assert observation.pupil_confidence == 0.90
        self.assertAlmostEqual(observation.features[0], 0.5)
        self.assertAlmostEqual(observation.features[1], 0.5)
        self.assertAlmostEqual(observation.features[2], 0.5)
        self.assertAlmostEqual(observation.features[3], 0.5)
        self.assertAlmostEqual(observation.features[6], 0.3)
        self.assertAlmostEqual(observation.features[7], 0.5)
        self.assertAlmostEqual(observation.features[8], 0.5)
        self.assertAlmostEqual(observation.features[9], 0.5)
        self.assertAlmostEqual(observation.features[13], 0.5)
        self.assertAlmostEqual(observation.features[14], 0.6)
        assert observation.head_bounds is not None
        for actual, expected in zip(observation.head_bounds, (0.05, 0.2, 0.5, 0.6), strict=True):
            self.assertAlmostEqual(actual, expected)
        assert observation.features[10] == 0.0
        assert len(observation.features) == 15
        assert len(observation.context) == 9
        self.assertAlmostEqual(observation.context[3], 0.3)
        self.assertAlmostEqual(observation.context[4], 0.5)
        self.assertAlmostEqual(observation.context[7], 128.0 / 255.0)
        self.assertAlmostEqual(observation.context[8], 0.0)

    def test_reliable_pupils_use_eye_local_features(self) -> None:
        """Available pupil evidence records displacement within each eye."""
        face = StubFace(
            conf=0.92,
            eye_state=((1.0, 50.0, 45.0, 0.90), (1.0, 47.0, 75.0, 0.95)),
            euler=(0.0, 0.0, 0.0),
            lms=landmarks(),
        )
        observation = OpenSeeFaceEstimator(
            200,
            100,
            tracker_factory=tracker_factory((face,)),
        ).observe(np.zeros((100, 200, 3), dtype=np.uint8), 1.0)
        assert isinstance(observation, EyeObservation)
        assert observation.pupil_available
        assert observation.landmarks is not None
        assert len(observation.landmarks) == 66
        self.assertAlmostEqual(observation.features[0], 0.25)
        self.assertAlmostEqual(observation.features[1], 0.0)
        self.assertAlmostEqual(observation.features[2], 0.75)
        self.assertAlmostEqual(observation.features[3], 0.5)
        self.assertAlmostEqual(observation.features[8], 0.5)
        self.assertAlmostEqual(observation.features[9], 0.25)

    def test_missing_pupils_use_head_only_while_missing_heads_request_recovery(self) -> None:
        """Pupils are optional, but absent or ambiguous head geometry is not."""
        weak = StubFace(
            conf=0.9,
            eye_state=((1.0, 50.0, 40.0, 0.2), (1.0, 50.0, 80.0, 0.9)),
            euler=(0.0, 0.0, 0.0),
            lms=landmarks(),
        )
        weak_pupils = OpenSeeFaceEstimator(
            200,
            100,
            tracker_factory=tracker_factory((weak,)),
        ).observe(np.zeros((100, 200, 3), dtype=np.uint8), 1.0)
        assert isinstance(weak_pupils, EyeObservation)
        assert not weak_pupils.pupil_available
        assert weak_pupils.features[10] == 0.0

        no_pupils = StubFace(0.9, None, (0.0, 0.0, 0.0), landmarks())
        head_only = OpenSeeFaceEstimator(
            200,
            100,
            tracker_factory=tracker_factory((no_pupils,)),
        ).observe(np.zeros((100, 200, 3), dtype=np.uint8), 1.0)
        assert isinstance(head_only, EyeObservation)
        assert not head_only.pupil_available

        for faces in ((), (weak, weak)):
            estimator = OpenSeeFaceEstimator(
                200,
                100,
                tracker_factory=tracker_factory(faces),
            )
            failure = estimator.observe(object(), 1.0)
            assert isinstance(failure, HeadTrackingFailure)
            assert failure.reason
            if not faces:
                assert "tracking was lost" in failure.reason
                assert "not visible" not in failure.reason
            if faces:
                assert failure.landmarks is not None
                assert len(failure.landmarks) == 66

    def test_close_is_idempotent_and_blocks_future_inference(self) -> None:
        """Released model resources cannot process another frame."""
        estimator = OpenSeeFaceEstimator(
            200,
            100,
            tracker_factory=lambda _width, _height: StubTracker(()),
        )
        estimator.close()
        estimator.close()
        with self.assertRaisesRegex(VisionError, "closed"):
            estimator.observe(object(), 1.0)


if __name__ == "__main__":
    unittest.main()
