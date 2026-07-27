"""Hardware-independent runtime contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

type Frame = object
type FeatureVector = tuple[float, ...]
type ContextVector = tuple[float, ...]
LANDMARK_DIMENSIONS = 2


class RuntimeStatus(Enum):
    """Observable process states."""

    STARTING = "starting"
    LOADING = "loading"
    AUTHORIZING = "authorizing"
    SELECTING_MODEL = "selecting-model"
    TOPOLOGY_UNVALIDATED = "topology-unvalidated"
    INITIAL_TRAINING = "initial-training"
    TRAINING_VALIDATING = "validating"
    ADAPTIVE_TRAINING = "adaptive-training"
    ALL_DATA_REFITTING = "all-data-refitting"
    TRAINING_COUNTDOWN = "training-countdown"
    TRAINING_COMPLETED = "training-completed"
    DISPLAY_CHANGE = "display-change"
    NOISE_MODEL_SELECTION = "noise-model-selection"
    TARGET_PREPARATION = "target-preparation"
    TARGET_MEASUREMENT = "target-measurement"
    HEAD_TRACKING_RECOVERY = "head-tracking-recovery"
    DIAGNOSTIC_CAPTURE = "diagnostic-capture"
    TRAINING_RECOMMENDED = "training-recommended"
    TRAINING_ERROR = "training-error"
    MODEL_ERROR = "model-error"
    STORE_ERROR = "store-error"
    ACTIVE = "active"
    ROUGH_IN = "rough-in"
    REFINEMENT = "refinement"
    REFINEMENT_SETTLED = "refinement-settled"
    INPUT_CAPTURE = "input-capture"
    RECALIBRATION_REQUIRED = "recalibration-required"
    CAMERA_ERROR = "camera-error"
    INPUT_ERROR = "input-error"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class TrainingRegionStatus:
    """Bounded output-region evidence safe for status and HUD display."""

    region: str
    cvar90: float
    surprise_lower: float
    surprise_upper: float
    observed_regions: int
    total_regions: int
    mode: str


@dataclass(frozen=True, slots=True)
class DisplayRegion:
    """One pointer region authorized for the current session."""

    region_id: str
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Reject geometry that cannot receive a pointer coordinate."""
        if not self.region_id:
            msg = "display region ID must not be empty"
            raise ValueError(msg)
        if self.width <= 0 or self.height <= 0:
            msg = "display region dimensions must be positive"
            raise ValueError(msg)

    @property
    def right(self) -> int:
        """Return the exclusive right edge."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Return the exclusive bottom edge."""
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class EyeObservation:
    """One reliable head/face result with optional pupil evidence."""

    timestamp: float
    left_open: float
    right_open: float
    features: FeatureVector
    confidence: float
    context: ContextVector
    pupil_available: bool = True
    pupil_confidence: float = 1.0
    head_bounds: tuple[float, float, float, float] | None = None
    head_pose: tuple[float, float, float] | None = None
    landmarks: tuple[tuple[float, float], ...] | None = None

    @property
    def evidence_class(self) -> str:
        """Describe whether optional pupil evidence refines head/face tracking."""
        return "head+face+pupils" if self.pupil_available else "head+face"

    def __post_init__(self) -> None:
        """Validate normalized confidence values and a usable feature vector."""
        values = (
            self.left_open,
            self.right_open,
            self.confidence,
            self.pupil_confidence,
        )
        if any(value < 0.0 or value > 1.0 for value in values):
            msg = "eye and face confidence values must be between zero and one"
            raise ValueError(msg)
        if not self.features or not self.context:
            msg = "gaze feature and routing context vectors must not be empty"
            raise ValueError(msg)
        if self.landmarks is not None and (
            not self.landmarks
            or any(
                len(point) != LANDMARK_DIMENSIONS
                or not all(math.isfinite(value) for value in point)
                for point in self.landmarks
            )
        ):
            msg = "transient diagnostic landmarks must be finite points"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class HeadTrackingFailure:
    """Transient guidance when reliable head/face geometry is unavailable."""

    reason: str
    head_bounds: tuple[float, float, float, float] | None = None
    head_pose: tuple[float, float, float] | None = None
    confidence: float | None = None
    landmarks: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        """Reject diagnostics that cannot explain the recovery action."""
        if not self.reason:
            msg = "head-tracking failure reason must not be empty"
            raise ValueError(msg)
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            msg = "head-tracking failure confidence must be normalized"
            raise ValueError(msg)
        if self.landmarks is not None and (
            not self.landmarks
            or any(
                len(point) != LANDMARK_DIMENSIONS
                or not all(math.isfinite(value) for value in point)
                for point in self.landmarks
            )
        ):
            msg = "transient diagnostic landmarks must be finite points"
            raise ValueError(msg)


class CameraCapture(Protocol):
    """Own a local camera for exactly one process session."""

    @property
    def camera_id(self) -> str:
        """Return an opaque local fingerprint without exposing a device path."""

    def read(self) -> Frame:
        """Return the next in-memory frame or raise a camera error."""

    def close(self) -> None:
        """Release the camera; repeated calls must be safe."""


class VisionEstimator(Protocol):
    """Estimate gaze and independent eye state from an in-memory frame."""

    def observe(
        self,
        frame: Frame,
        timestamp: float,
    ) -> EyeObservation | HeadTrackingFailure:
        """Return reliable head/face evidence or transient corrective guidance."""

    def close(self) -> None:
        """Release inference resources; repeated calls must be safe."""


class PointerController(Protocol):
    """Control the display regions authorized for this session."""

    @property
    def regions(self) -> tuple[DisplayRegion, ...]:
        """Return every authorized display region."""

    @property
    def topology_id(self) -> str:
        """Return an opaque identity that changes with selected geometry."""

    def move(self, region_id: str, x: float, y: float) -> None:
        """Move to a region-local logical coordinate."""

    async def close(self) -> None:
        """Drain motion events and close authorization idempotently."""


class DebugHud(Protocol):
    """Display rate-limited pointer diagnostics when explicitly enabled."""

    def set_model_context(
        self,
        routing: str,
        topology_quality: str,
        model_confidence: str,
    ) -> None:
        """Set safe inferred routing and quality labels without feature values."""

    def set_noise_context(self, confidence: str, alpha: float, dead_zone: float) -> None:
        """Set bounded inferred noise and smoothing diagnostics."""

    def set_evidence_context(self, evidence_class: str) -> None:
        """Report the current head-only or pupil-refined evidence class."""

    def set_training_region(self, context: TrainingRegionStatus) -> None:
        """Report bounded region-aware target selection without feature values."""

    def set_refinement_context(self, context: str) -> None:
        """Report bounded matrix, confidence, and settlement state."""

    async def update(self, region_id: str, x: float, y: float) -> None:
        """Show the current authorized region and global logical coordinate."""

    async def close(self) -> None:
        """Remove the HUD and release its desktop connection idempotently."""


class HeadDiagnosticSurface(Protocol):
    """Render transient local camera guidance while head tracking recovers."""

    def show_head_diagnostic(
        self,
        frame: Frame,
        failure: HeadTrackingFailure,
        seconds_remaining: float,
    ) -> None:
        """Show camera pixels, bounds, pose axes, and corrective guidance."""

    def hide_head_diagnostic(self) -> None:
        """Remove transient camera pixels after tracking recovers."""

    async def close(self) -> None:
        """Remove the diagnostic surface and release it idempotently."""


class InputCaptureSession(Protocol):
    """Own one optional portal-authorized physical pointer capture."""

    barrier_count: int

    async def close(self) -> None:
        """Release the portal and EIS resources idempotently."""


class RefinementSurface(Protocol):
    """Show one transient click-through recursive character matrix."""

    def show_refinement(  # noqa: PLR0913
        self,
        left: float,
        top: float,
        width: float,
        height: float,
        depth: int,
        source: str,
        rows: tuple[str, ...],
    ) -> None:
        """Show labelled matrix geometry intersected with authorized outputs."""

    def hide_refinement(self) -> None:
        """Clear the active refinement grid."""

    async def close(self) -> None:
        """Destroy every refinement surface idempotently."""


class TrainingSurface(HeadDiagnosticSurface, Protocol):
    """Show transient click-through targets across authorized displays."""

    def show_target(
        self,
        region_id: str,
        x: float,
        y: float,
        diameter: float,
        label: str,
    ) -> None:
        """Replace the visible target on one authorized region."""

    def show_message(self, label: str) -> None:
        """Show one training message on every authorized region."""

    def show_preparation(  # noqa: PLR0913
        self,
        region_id: str,
        x: float,
        y: float,
        diameter: float,
        label: str,
        prior_opacity: float,
    ) -> None:
        """Show the squared next dot while fading and cueing from the prior dot."""

    def target_diameter(
        self,
        region_id: str,
        physical_millimetres: float,
        fallback_pixels: float,
    ) -> float:
        """Resolve one physical target size for an authorized region."""

    async def close(self) -> None:
        """Remove the training surface and release its desktop connection."""


class StatusSink(Protocol):
    """Report process state without coupling policy to a user interface."""

    def report(self, status: RuntimeStatus, detail: str = "") -> None:
        """Publish a status transition and optional safe detail."""
