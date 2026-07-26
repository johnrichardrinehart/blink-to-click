"""In-process face, pupil, and eye-state estimation."""

from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING, Protocol, cast

import cv2  # type: ignore[import-not-found]
import numpy as np

from gazeebo.contracts import EyeObservation, Frame, HeadTrackingFailure

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


LANDMARK_COUNT = 66
LANDMARK_DIMENSIONS = 2
EYE_STATE_SHAPE = (2, 4)
RIGHT_EYE_INDICES = (36, 37, 38, 39, 40, 41)
LEFT_EYE_INDICES = (42, 43, 44, 45, 46, 47)
CLOSED_EYE_RATIO = 0.08
EYE_RATIO_RANGE = 0.18
GAZE_OPEN_THRESHOLD = 0.35
MINIMUM_PUPIL_CONFIDENCE = 0.55
MINIMUM_EYE_WIDTH = 1e-6
ROLL_INDEX = 2
MINIMUM_IMAGE_DIMENSIONS = 2
COLOR_IMAGE_DIMENSIONS = 3
MINIMUM_FACE_SPAN = 0.04
MAXIMUM_FACE_SPAN = 0.98
FACE_FRAME_MARGIN = 0.10
_LIGHTING_NORMALIZER = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))


class VisionError(RuntimeError):
    """The local vision model could not produce a safe observation."""


class _Face(Protocol):
    conf: float
    eye_state: Sequence[Sequence[float]] | None
    euler: Sequence[float] | None
    lms: Sequence[Sequence[float]] | None


class _Tracker(Protocol):
    def predict(self, frame: Frame) -> Sequence[_Face]:
        """Return tracked faces."""


def _normalize_tracking_lighting(frame: Frame) -> Frame:
    """Reduce local overhead-light contrast without retaining camera pixels."""
    if not isinstance(frame, np.ndarray) or frame.ndim != COLOR_IMAGE_DIMENSIONS:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    luminance, first_color, second_color = cv2.split(lab)
    normalized = _LIGHTING_NORMALIZER.apply(luminance)
    merged = cv2.merge((normalized, first_color, second_color))
    return cast("Frame", cv2.cvtColor(merged, cv2.COLOR_LAB2BGR))


def _default_tracker_factory(width: int, height: int) -> _Tracker:
    """Load the packaged OpenSeeFace tracker without starting its UDP executable."""
    tracker_directory = os.environ.get("GAZEEBO_TRACKER_DIR")
    if tracker_directory and tracker_directory not in sys.path:
        sys.path.insert(0, tracker_directory)
    try:
        from tracker import Tracker  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as error:
        msg = "packaged face tracker is unavailable"
        raise VisionError(msg) from error
    model_directory = os.environ.get("GAZEEBO_MODEL_DIR")
    return cast(
        "_Tracker",
        Tracker(
            width,
            height,
            model_type=3,
            detection_threshold=0.2,
            threshold=0.6,
            max_faces=1,
            discard_after=5,
            max_threads=4,
            silent=True,
            model_dir=model_directory,
            feature_level=2,
            use_retinaface=False,
            try_hard=True,
        ),
    )


def _angle_feature(degrees: float) -> float:
    """Map equivalent Euler wraparound values to one continuous feature."""
    return math.sin(math.radians(degrees))


def _normalized_pupil(
    points: np.ndarray,
    indices: tuple[int, ...],
    pupil_x: float,
    pupil_y: float,
) -> tuple[float, float] | None:
    """Express one pupil in its rotated eye aperture instead of the face box."""
    eye = points[list(indices)]
    horizontal = eye[3] - eye[0]
    width = float(np.linalg.norm(horizontal))
    if not math.isfinite(width) or width <= MINIMUM_EYE_WIDTH:
        return None
    horizontal /= width
    vertical = np.asarray((-horizontal[1], horizontal[0]), dtype=np.float64)
    horizontal_positions = eye @ horizontal
    vertical_positions = eye @ vertical
    horizontal_span = float(np.ptp(horizontal_positions))
    vertical_span = float(np.ptp(vertical_positions))
    if horizontal_span <= MINIMUM_EYE_WIDTH or vertical_span <= MINIMUM_EYE_WIDTH:
        return None
    pupil = np.asarray((pupil_x, pupil_y), dtype=np.float64)
    normalized_x = (float(pupil @ horizontal) - float(horizontal_positions.min())) / horizontal_span
    normalized_y = (float(pupil @ vertical) - float(vertical_positions.min())) / vertical_span
    return (
        float(np.clip(normalized_x, 0.0, 1.0)),
        float(np.clip(normalized_y, 0.0, 1.0)),
    )


def _diagnostic_geometry(
    face: _Face,
    width: int,
    height: int,
) -> tuple[
    float | None,
    tuple[float, float, float, float] | None,
    tuple[float, float, float] | None,
    tuple[tuple[float, float], ...] | None,
]:
    """Extract every finite detector value available for warning diagnostics."""
    raw_confidence = float(face.conf)
    confidence = (
        None if not math.isfinite(raw_confidence) else float(np.clip(raw_confidence, 0.0, 1.0))
    )
    landmarks: tuple[tuple[float, float], ...] | None = None
    bounds: tuple[float, float, float, float] | None = None
    if face.lms is not None:
        raw = np.asarray(face.lms, dtype=np.float64)
        if (
            raw.ndim == LANDMARK_DIMENSIONS
            and raw.shape[0] >= LANDMARK_COUNT
            and raw.shape[1] >= LANDMARK_DIMENSIONS
        ):
            points = raw[:LANDMARK_COUNT, :LANDMARK_DIMENSIONS][:, [1, 0]]
            if np.isfinite(points).all():
                landmarks = tuple((float(point[0]), float(point[1])) for point in points)
                minimum = points.min(axis=0)
                size = points.max(axis=0) - minimum
                bounds = (
                    float(minimum[0] / width),
                    float(minimum[1] / height),
                    float(size[0] / width),
                    float(size[1] / height),
                )
    pose: tuple[float, float, float] | None = None
    if face.euler is not None:
        euler = np.asarray(face.euler, dtype=np.float64)
        if euler.size > ROLL_INDEX and np.isfinite(euler[: ROLL_INDEX + 1]).all():
            pose = (float(euler[0]), float(euler[1]), float(euler[ROLL_INDEX]))
    return confidence, bounds, pose, landmarks


def _head_failure(
    reason: str,
    face: _Face,
    width: int,
    height: int,
) -> HeadTrackingFailure:
    confidence, bounds, pose, landmarks = _diagnostic_geometry(face, width, height)
    return HeadTrackingFailure(reason, bounds, pose, confidence, landmarks)


def _eye_openness(points: np.ndarray, indices: tuple[int, ...]) -> float:
    """Normalize one six-landmark eye aspect ratio to an open confidence."""
    eye = points[np.asarray(indices)]
    horizontal = float(np.linalg.norm(eye[0] - eye[3]))
    if horizontal <= MINIMUM_EYE_WIDTH:
        return 0.0
    vertical = float(np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4]))
    ratio = vertical / (2.0 * horizontal)
    return float(np.clip((ratio - CLOSED_EYE_RATIO) / EYE_RATIO_RANGE, 0.0, 1.0))


class OpenSeeFaceEstimator:
    """Convert OpenSeeFace output into ephemeral normalized gaze features."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        minimum_confidence: float = 0.2,
        tracker_factory: Callable[[int, int], _Tracker] | None = None,
    ) -> None:
        """Create one in-process tracker for fixed camera dimensions."""
        if width <= 0 or height <= 0:
            msg = "vision dimensions must be positive"
            raise ValueError(msg)
        if not 0.0 <= minimum_confidence <= 1.0:
            msg = "minimum vision confidence must be between zero and one"
            raise ValueError(msg)
        self._width = width
        self._height = height
        self._minimum_confidence = minimum_confidence
        self._tracker: _Tracker | None = (tracker_factory or _default_tracker_factory)(
            width,
            height,
        )

    def observe(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        frame: Frame,
        timestamp: float,
    ) -> EyeObservation | HeadTrackingFailure:
        """Return reliable head/face geometry with optional pupil evidence."""
        if self._tracker is None:
            msg = "vision estimator is closed"
            raise VisionError(msg)
        faces = self._tracker.predict(_normalize_tracking_lighting(frame))
        if not faces:
            return HeadTrackingFailure(
                "Face tracking was lost; keep your full head visible and centered"
            )
        if len(faces) != 1:
            return _head_failure(
                "More than one face is visible; keep one viewer in frame",
                faces[0],
                self._width,
                self._height,
            )
        face = faces[0]
        face_confidence = float(face.conf)
        if not np.isfinite(face_confidence) or face_confidence < self._minimum_confidence:
            return _head_failure(
                "Head tracking is uncertain; improve lighting and camera focus",
                face,
                self._width,
                self._height,
            )
        if face.lms is None:
            return _head_failure(
                "Face landmarks are unavailable; center your full face in view",
                face,
                self._width,
                self._height,
            )
        landmarks = np.asarray(face.lms, dtype=np.float64)
        if (
            landmarks.ndim != LANDMARK_DIMENSIONS
            or landmarks.shape[0] < LANDMARK_COUNT
            or landmarks.shape[1] < LANDMARK_DIMENSIONS
        ):
            return _head_failure(
                "Face geometry is incomplete; move fully into the camera view",
                face,
                self._width,
                self._height,
            )

        # OpenSeeFace exposes image coordinates as (row/y, column/x).
        # Convert once so all Gazeebo geometry uses the conventional (x, y) order.
        face_points = landmarks[:LANDMARK_COUNT, :LANDMARK_DIMENSIONS][:, [1, 0]]
        if not np.isfinite(face_points).all():
            return _head_failure(
                "Face geometry is unstable; hold your head inside the camera view",
                face,
                self._width,
                self._height,
            )
        minimum = face_points.min(axis=0)
        maximum = face_points.max(axis=0)
        size = maximum - minimum
        bounds = (
            float(minimum[0] / self._width),
            float(minimum[1] / self._height),
            float(size[0] / self._width),
            float(size[1] / self._height),
        )
        if np.any(size <= 1.0):
            return _head_failure(
                "Face geometry is too small to track reliably",
                face,
                self._width,
                self._height,
            )
        if (
            bounds[2] < MINIMUM_FACE_SPAN
            or bounds[3] < MINIMUM_FACE_SPAN
            or bounds[2] > MAXIMUM_FACE_SPAN
            or bounds[3] > MAXIMUM_FACE_SPAN
        ):
            return _head_failure(
                "Move closer to or farther from the camera until your full head is visible",
                face,
                self._width,
                self._height,
            )
        if (
            minimum[0] < -self._width * FACE_FRAME_MARGIN
            or minimum[1] < -self._height * FACE_FRAME_MARGIN
            or maximum[0] > self._width * (1.0 + FACE_FRAME_MARGIN)
            or maximum[1] > self._height * (1.0 + FACE_FRAME_MARGIN)
        ):
            return _head_failure(
                "Your head is partly out of frame; recenter it",
                face,
                self._width,
                self._height,
            )
        if face.euler is None:
            return _head_failure(
                "Head angle is unavailable; face the camera and improve lighting",
                face,
                self._width,
                self._height,
            )
        euler = np.asarray(face.euler, dtype=np.float64)
        if euler.size <= ROLL_INDEX or not np.isfinite(euler[: ROLL_INDEX + 1]).all():
            return _head_failure(
                "Head angle is unstable; hold your full head in view",
                face,
                self._width,
                self._height,
            )
        pitch = float(euler[0])
        yaw = float(euler[1])
        roll = float(euler[ROLL_INDEX])
        pose = (pitch, yaw, roll)
        center = (minimum + maximum) / 2.0

        left_open = 0.0
        right_open = 0.0
        pupil_confidence = 0.0
        normalized_left_x = 0.5
        normalized_left_y = 0.5
        normalized_right_x = 0.5
        normalized_right_y = 0.5
        pupil_available = False
        if face.eye_state is not None:
            eyes = np.asarray(face.eye_state, dtype=np.float64)
            if eyes.shape == EYE_STATE_SHAPE and np.isfinite(eyes).all():
                _, right_y, right_x, right_confidence = eyes[0]
                _, left_y, left_x, left_confidence = eyes[1]
                right_open = _eye_openness(face_points, RIGHT_EYE_INDICES)
                left_open = _eye_openness(face_points, LEFT_EYE_INDICES)
                pupil_confidence = float(
                    np.clip(min(float(right_confidence), float(left_confidence)), 0.0, 1.0)
                )
                pupil_available = (
                    pupil_confidence >= MINIMUM_PUPIL_CONFIDENCE
                    and left_open > GAZE_OPEN_THRESHOLD
                    and right_open > GAZE_OPEN_THRESHOLD
                )
                if pupil_available:
                    right_pupil = _normalized_pupil(
                        face_points,
                        RIGHT_EYE_INDICES,
                        float(right_x),
                        float(right_y),
                    )
                    left_pupil = _normalized_pupil(
                        face_points,
                        LEFT_EYE_INDICES,
                        float(left_x),
                        float(left_y),
                    )
                    if right_pupil is None or left_pupil is None:
                        pupil_available = False
                    else:
                        normalized_right_x, normalized_right_y = right_pupil
                        normalized_left_x, normalized_left_y = left_pupil

        features = (
            normalized_left_x,
            normalized_left_y,
            normalized_right_x,
            normalized_right_y,
            _angle_feature(pitch),
            _angle_feature(yaw),
            float(center[0] / self._width),
            float(center[1] / self._height),
            (normalized_left_x + normalized_right_x) / 2.0,
            (normalized_left_y + normalized_right_y) / 2.0,
            1.0 if pupil_available else 0.0,
            pupil_confidence,
            _angle_feature(roll),
            float(size[0] / self._width),
            float(size[1] / self._height),
        )
        luminance_mean, luminance_spread = _illumination(frame)
        context = (
            _angle_feature(pitch),
            _angle_feature(yaw),
            _angle_feature(roll),
            float(center[0] / self._width),
            float(center[1] / self._height),
            float(size[0] / self._width),
            float(size[1] / self._height),
            luminance_mean,
            luminance_spread,
        )
        if not all(np.isfinite(value) for value in (*features, *context)):
            return _head_failure(
                "Head or pupil geometry is not finite; recenter your head",
                face,
                self._width,
                self._height,
            )
        return EyeObservation(
            timestamp=timestamp,
            left_open=float(np.clip(left_open, 0.0, 1.0)),
            right_open=float(np.clip(right_open, 0.0, 1.0)),
            features=features,
            confidence=float(np.clip(face_confidence, 0.0, 1.0)),
            context=context,
            pupil_available=pupil_available,
            pupil_confidence=pupil_confidence,
            head_bounds=bounds,
            head_pose=pose,
            landmarks=tuple((float(point[0]), float(point[1])) for point in face_points),
        )

    def close(self) -> None:
        """Release model references idempotently."""
        self._tracker = None


def _illumination(frame: Frame) -> tuple[float, float]:
    """Reduce a frame to bounded luminance context without retaining pixels."""
    try:
        pixels = np.asarray(frame, dtype=np.float64)
    except (TypeError, ValueError):
        return 0.5, 0.0
    if pixels.ndim < MINIMUM_IMAGE_DIMENSIONS or pixels.size == 0:
        return 0.5, 0.0
    if pixels.ndim >= COLOR_IMAGE_DIMENSIONS:
        pixels = pixels[..., :3].mean(axis=-1)
    mean = float(np.clip(pixels.mean() / 255.0, 0.0, 1.0))
    spread = float(np.clip(pixels.std() / 128.0, 0.0, 1.0))
    return mean, spread
