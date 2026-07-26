"""Iterative adaptive calibration training and unseen-batch metrics."""

from __future__ import annotations

import copy
import ctypes
import math
import statistics
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import numpy as np

from gazeebo.calibration import (
    CalibrationModel,
    CalibrationSample,
    IncrementalCalibration,
    aggregate_feature_dispersion,
    aggregate_features,
    target_fit_weight,
)
from gazeebo.contexts import ValidationMetrics, candidate_is_acceptable
from gazeebo.contracts import (
    Frame,
    HeadTrackingFailure,
    RuntimeStatus,
    TrainingRegionStatus,
)
from gazeebo.geometry import DisplayTopology, Point, PointerTarget, rolling_point_median
from gazeebo.native import NativeRendererError, load_native_renderer
from gazeebo.recovery import HeadTrackingError, observe_with_head_recovery
from gazeebo.state import CursorNoiseSummary
from gazeebo.surprise import RegionKey, RegionSelection, RegionSurpriseScheduler

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from gazeebo.contracts import (
        CameraCapture,
        DebugHud,
        DisplayRegion,
        EyeObservation,
        FeatureVector,
        HeadDiagnosticSurface,
        PointerController,
        StatusSink,
        TrainingSurface,
        VisionEstimator,
    )

TRAINING_ERROR_SIZE = 256
MINIMUM_TRAINING_TARGETS = 5
EDGE_BOUNDARY = 0.18
OVERLAP_FADE_OPACITIES = (1.0, 0.5, 0.0)
GRAYSCALE_DIMENSIONS = 2
COLOR_DIMENSIONS = 3
DEFAULT_CVAR_TAIL_FRACTION = 0.10
MAXIMUM_CVAR_TAIL_FRACTION = 0.50
MINIMUM_CVAR_HISTOGRAM_BINS = 32
MAXIMUM_CVAR_HISTOGRAM_BINS = 4096

_TARGET_POSITIONS = (
    (0.00, 0.50),
    (1.00, 0.50),
    (0.50, 0.50),
    (0.00, 0.00),
    (1.00, 0.00),
    (0.00, 1.00),
    (1.00, 1.00),
    (0.50, 0.00),
    (0.50, 1.00),
    (0.25, 0.25),
    (0.75, 0.25),
    (0.25, 0.75),
    (0.75, 0.75),
    (0.12, 0.25),
    (0.88, 0.75),
)


class TrainingError(RuntimeError):
    """Adaptive calibration cannot continue safely."""


class GazePredictor(Protocol):
    """Predict global coordinates with optional context routing."""

    @property
    def kind(self) -> str:
        """Return a concise estimator label."""

    def predict(
        self,
        features: FeatureVector,
        context: tuple[float, ...] | None = None,
    ) -> Point:
        """Return one global logical prediction."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Finite adaptive-training timing, target, and sizing controls."""

    batch_size: int = 5
    maximum_targets: int = 55
    precision_threshold: float = 100.0
    preparation_seconds: float = 2.0
    transition_overlap_seconds: float = 1.0
    measurement_seconds: float = 2.0
    physical_target_diameter_mm: float = 12.0
    fallback_target_diameter: float = 72.0
    countdown_interval_seconds: float = 1.0
    completion_seconds: float = 2.0
    surprise_confidence_z: float = 1.6448536269514722
    surprise_maximum: float = 100.0
    surprise_decay: float = 0.95
    surprise_maximum_outputs: int = 16
    surprise_tail_fraction: float = DEFAULT_CVAR_TAIL_FRACTION
    surprise_histogram_bins: int = 1024

    def __post_init__(self) -> None:  # noqa: C901
        """Reject invalid or non-terminating training settings."""
        if self.batch_size < MINIMUM_TRAINING_TARGETS:
            msg = f"training batch size must be at least {MINIMUM_TRAINING_TARGETS}"
            raise ValueError(msg)
        if self.maximum_targets < self.batch_size:
            msg = "training maximum must cover one complete batch"
            raise ValueError(msg)
        if self.maximum_targets % self.batch_size != 0:
            msg = "training maximum must be a multiple of batch size"
            raise ValueError(msg)
        finite_values = (
            self.precision_threshold,
            self.preparation_seconds,
            self.transition_overlap_seconds,
            self.measurement_seconds,
            self.physical_target_diameter_mm,
            self.fallback_target_diameter,
            self.countdown_interval_seconds,
            self.completion_seconds,
            self.surprise_confidence_z,
            self.surprise_maximum,
            self.surprise_decay,
            self.surprise_tail_fraction,
        )
        if not all(math.isfinite(value) for value in finite_values):
            msg = "training settings must be finite"
            raise ValueError(msg)
        if self.precision_threshold <= 0.0:
            msg = "training precision threshold must be positive"
            raise ValueError(msg)
        if self.preparation_seconds <= 0.0 or self.measurement_seconds <= 0.0:
            msg = "training preparation and measurement windows must be positive"
            raise ValueError(msg)
        if not 0.0 < self.transition_overlap_seconds <= self.preparation_seconds:
            msg = "training transition overlap must be positive and cover no more than preparation"
            raise ValueError(msg)
        if self.physical_target_diameter_mm <= 0.0:
            msg = "training physical target diameter must be positive"
            raise ValueError(msg)
        if self.fallback_target_diameter <= 0.0:
            msg = "training fallback target diameter must be positive"
            raise ValueError(msg)
        if self.countdown_interval_seconds < 0.0 or self.completion_seconds < 0.0:
            msg = "training message timings must be non-negative"
            raise ValueError(msg)
        if (
            self.surprise_confidence_z <= 0.0
            or self.surprise_maximum <= 0.0
            or not 0.0 < self.surprise_decay <= 1.0
            or self.surprise_maximum_outputs <= 0
            or not math.isclose(
                self.surprise_tail_fraction,
                DEFAULT_CVAR_TAIL_FRACTION,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not MINIMUM_CVAR_HISTOGRAM_BINS
            <= self.surprise_histogram_bins
            <= MAXIMUM_CVAR_HISTOGRAM_BINS
        ):
            msg = "training surprise bounds are invalid"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DisplayModeMetrics:
    """Physical dimensions and the selected active pixel mode of one output."""

    mode_width: int
    mode_height: int
    physical_width_mm: int
    physical_height_mm: int

    def __post_init__(self) -> None:
        """Reject missing or impossible selected-mode metadata."""
        if (
            min(
                self.mode_width,
                self.mode_height,
                self.physical_width_mm,
                self.physical_height_mm,
            )
            <= 0
        ):
            msg = "display mode and physical dimensions must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TrainingTarget:
    """One region-local visual target."""

    region_id: str
    x: float
    y: float
    diameter: float
    edge_or_corner: bool


@dataclass(frozen=True, slots=True)
class TargetMeasurement:
    """Accuracy and uncertainty measured before incorporating one target."""

    radial_error: float
    edge_or_corner: bool
    response_seconds: float | None
    noise_spread: float = 0.0
    predictive_uncertainty: float | None = None
    region: str = ""


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Aggregate non-persistent validation measurements."""

    target_count: int
    hit_count: int
    median_error: float
    edge_error: float
    median_response: float | None
    median_noise_spread: float
    maximum_region_error: float = 0.0
    maximum_region_cvar90: float = 0.0
    maximum_region_upper: float = 0.0
    regions_precise: bool = False
    selected_regions: tuple[str, ...] = ()
    regional_surprise_lower: float = 0.0
    regional_surprise_upper: float = 0.0
    observed_regions: int = 0
    total_regions: int = 0
    regions_equalized: bool = False

    def summary(self, label: str) -> str:
        """Format one concise status line without persisting measurements."""
        response = f"{self.median_response:.2f}s" if self.median_response is not None else "n/a"
        base = (
            f"{label}: hits {self.hit_count}/{self.target_count}, "
            f"median error {self.median_error:.0f}px, "
            f"edge error {self.edge_error:.0f}px, response {response}, "
            f"cursor spread {self.median_noise_spread:.0f}px p95, "
            f"worst region {self.maximum_region_error:.0f}px, "
            f"worst CVaR90 {self.maximum_region_cvar90:.0f}px, "
            f"maximum bound {self.maximum_region_upper:.0f}px"
        )
        if not self.selected_regions:
            return base
        regions = ",".join(self.selected_regions)
        state = "equalized" if self.regions_equalized else "unequalized"
        precision = "region-precise" if self.regions_precise else "region-above-threshold"
        return (
            f"{base}, regions {regions}, surprise "
            f"{self.regional_surprise_lower:.2f}..{self.regional_surprise_upper:.2f}, "
            f"coverage {self.observed_regions}/{self.total_regions}, {state}, {precision}"
        )


@dataclass(frozen=True, slots=True)
class CollectedTarget:
    """One reported target aggregate eligible for later training."""

    features: FeatureVector
    context: tuple[float, ...]
    target: PointerTarget
    zone: str
    noise: CursorNoiseSummary | None = None
    feature_dispersion: FeatureVector = ()
    unseen_error: float | None = None
    predictive_uncertainty: float | None = None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Final all-data candidate, completed aggregates, and unseen evidence."""

    model: GazePredictor
    rounds: tuple[TrainingMetrics, ...]
    precision_met: bool
    aggregate_metrics: TrainingMetrics
    completed_targets: tuple[CollectedTarget, ...] = ()
    validation_targets: tuple[CollectedTarget, ...] = ()
    incumbent_metrics: TrainingMetrics | None = None
    model_accepted: bool = True

    @property
    def before(self) -> TrainingMetrics:
        """Return the first unseen-batch metric for comparisons."""
        return self.rounds[0]

    @property
    def after(self) -> TrainingMetrics:
        """Return the final unseen-batch metric for comparisons."""
        return self.rounds[-1]


@dataclass(frozen=True, slots=True)
class _TargetResult:
    features: tuple[FeatureVector, ...]
    contexts: tuple[tuple[float, ...], ...]
    measurement: TargetMeasurement
    incumbent_measurement: TargetMeasurement
    noise: CursorNoiseSummary


@dataclass(frozen=True, slots=True)
class _ValidationResult:
    metrics: TrainingMetrics
    incumbent_metrics: TrainingMetrics
    measurements: tuple[TargetMeasurement, ...]
    incumbent_measurements: tuple[TargetMeasurement, ...]
    samples: tuple[CalibrationSample, ...]
    targets: tuple[CollectedTarget, ...]
    model: GazePredictor


@dataclass(frozen=True, slots=True)
class _ValidationInterruption:
    """Completed target count from an unfinished five-target report."""

    completed_count: int


@dataclass(slots=True)
class _PointerCadence:
    interval: float
    last_update: float | None = None
    visible_point: Point | None = None

    def due(self, timestamp: float) -> bool:
        if self.last_update is None or timestamp - self.last_update >= self.interval:
            self.last_update = timestamp
            return True
        return False


class _NativeTrainingSurface:
    """Bind the adaptive training to the packaged native Wayland renderer."""

    def __init__(self, region: DisplayRegion) -> None:
        try:
            library = load_native_renderer()
        except NativeRendererError as load_error:
            raise TrainingError(str(load_error)) from load_error
        library.gazeebo_training_create.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_training_create.restype = ctypes.c_void_p
        library.gazeebo_training_show_target.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_training_show_target.restype = ctypes.c_int
        library.gazeebo_training_show_message.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_training_show_message.restype = ctypes.c_int
        library.gazeebo_training_show_cue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_training_show_cue.restype = ctypes.c_int
        library.gazeebo_training_show_diagnostic.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_training_show_diagnostic.restype = ctypes.c_int
        library.gazeebo_training_hide.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_training_hide.restype = ctypes.c_int
        library.gazeebo_training_display_metrics.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        library.gazeebo_training_display_metrics.restype = ctypes.c_int
        library.gazeebo_training_destroy.argtypes = [ctypes.c_void_p]
        library.gazeebo_training_destroy.restype = None
        error_buffer = ctypes.create_string_buffer(TRAINING_ERROR_SIZE)
        handle = library.gazeebo_training_create(
            region.x,
            region.y,
            region.width,
            region.height,
            error_buffer,
            TRAINING_ERROR_SIZE,
        )
        if not handle:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"calibration training failed to start: {detail}"
            raise TrainingError(msg)
        self._library = library
        self._handle = ctypes.c_void_p(handle)
        self._closed = False

    def show_target(self, x: float, y: float, diameter: float, label: str) -> None:
        if self._closed:
            return
        error_buffer = ctypes.create_string_buffer(TRAINING_ERROR_SIZE)
        result = self._library.gazeebo_training_show_target(
            self._handle,
            x,
            y,
            diameter,
            label.encode(),
            error_buffer,
            TRAINING_ERROR_SIZE,
        )
        if result != 0:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"calibration training update failed: {detail}"
            raise TrainingError(msg)

    def show_message(self, label: str) -> None:
        if self._closed:
            return
        error_buffer = ctypes.create_string_buffer(TRAINING_ERROR_SIZE)
        result = self._library.gazeebo_training_show_message(
            self._handle,
            label.encode(),
            error_buffer,
            TRAINING_ERROR_SIZE,
        )
        if result != 0:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"calibration training message failed: {detail}"
            raise TrainingError(msg)

    def show_preparation(
        self,
        direction: str,
        label: str,
        next_target: tuple[float, float, float] | None,
        prior: tuple[float, float, float] | None,
        prior_opacity: float,
    ) -> None:
        """Show the squared next dot and direction while fading the prior circle."""
        if self._closed:
            return
        next_x, next_y, next_diameter = next_target or (-1.0, -1.0, 0.0)
        prior_x, prior_y, prior_diameter = prior or (-1.0, -1.0, 0.0)
        error_buffer = ctypes.create_string_buffer(TRAINING_ERROR_SIZE)
        result = self._library.gazeebo_training_show_cue(
            self._handle,
            direction.encode(),
            next_x,
            next_y,
            next_diameter,
            prior_x,
            prior_y,
            prior_diameter,
            prior_opacity,
            label.encode(),
            error_buffer,
            TRAINING_ERROR_SIZE,
        )
        if result != 0:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"calibration training cue failed: {detail}"
            raise TrainingError(msg)

    def show_head_diagnostic(
        self,
        pixels: np.ndarray,
        failure: HeadTrackingFailure,
        label: str,
    ) -> None:
        """Render one transient camera frame with head guidance."""
        if self._closed:
            return
        height, width = pixels.shape[:2]
        bounds = failure.head_bounds or (0.0, 0.0, 0.0, 0.0)
        pose = failure.head_pose or (0.0, 0.0, 0.0)
        pointer = pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        error_buffer = ctypes.create_string_buffer(TRAINING_ERROR_SIZE)
        result = self._library.gazeebo_training_show_diagnostic(
            self._handle,
            pointer,
            width,
            height,
            int(pixels.strides[0]),
            *bounds,
            *pose,
            int(failure.head_bounds is not None),
            int(failure.head_pose is not None),
            label.encode(),
            error_buffer,
            TRAINING_ERROR_SIZE,
        )
        if result != 0:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"head diagnostic update failed: {detail}"
            raise TrainingError(msg)

    def hide(self) -> None:
        if self._closed:
            return
        error_buffer = ctypes.create_string_buffer(TRAINING_ERROR_SIZE)
        result = self._library.gazeebo_training_hide(
            self._handle,
            error_buffer,
            TRAINING_ERROR_SIZE,
        )
        if result != 0:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"calibration training hide failed: {detail}"
            raise TrainingError(msg)

    def display_metrics(self) -> DisplayModeMetrics | None:
        """Return exact current-mode metrics, or none for the safe fallback."""
        if self._closed:
            return None
        mode_width = ctypes.c_int32()
        mode_height = ctypes.c_int32()
        physical_width = ctypes.c_int32()
        physical_height = ctypes.c_int32()
        result = self._library.gazeebo_training_display_metrics(
            self._handle,
            ctypes.byref(mode_width),
            ctypes.byref(mode_height),
            ctypes.byref(physical_width),
            ctypes.byref(physical_height),
        )
        if result != 0:
            return None
        return DisplayModeMetrics(
            mode_width.value,
            mode_height.value,
            physical_width.value,
            physical_height.value,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._library.gazeebo_training_destroy(self._handle)


class LayerShellTraining:
    """Own click-through training surfaces on all authorized displays."""

    def __init__(
        self,
        surfaces: dict[str, _NativeTrainingSurface],
        regions: Sequence[DisplayRegion],
    ) -> None:
        """Own the supplied native surfaces until explicit cleanup."""
        self._surfaces = surfaces
        self._regions = {region.region_id: region for region in regions}
        self._last_target: tuple[str, float, float, float] | None = None
        self._closed = False

    @classmethod
    def create(cls, regions: Sequence[DisplayRegion]) -> LayerShellTraining:
        """Create one native training surface per authorized portal region."""
        surfaces: dict[str, _NativeTrainingSurface] = {}
        try:
            for region in regions:
                surfaces[region.region_id] = _NativeTrainingSurface(region)
        except Exception:
            for surface in surfaces.values():
                surface.close()
            raise
        return cls(surfaces, regions)

    def show_target(
        self,
        region_id: str,
        x: float,
        y: float,
        diameter: float,
        label: str,
    ) -> None:
        """Show one target and clear every other authorized display."""
        if self._closed:
            return
        target_surface = self._surfaces.get(region_id)
        if target_surface is None:
            msg = f"unknown calibration training region: {region_id}"
            raise TrainingError(msg)
        for identifier, surface in self._surfaces.items():
            if identifier == region_id:
                surface.show_target(x, y, diameter, label)
            else:
                surface.hide()
        self._last_target = (region_id, x, y, diameter)

    def show_message(self, label: str) -> None:
        """Show one training message on every authorized display."""
        if self._closed:
            return
        for surface in self._surfaces.values():
            surface.show_message(label)

    def show_preparation(  # noqa: PLR0913
        self,
        region_id: str,
        x: float,
        y: float,
        diameter: float,
        label: str,
        prior_opacity: float,
    ) -> None:
        """Show the squared next dot and cues while fading the prior target."""
        if self._closed:
            return
        target_region = self._regions.get(region_id)
        if target_region is None:
            msg = f"unknown calibration cue region: {region_id}"
            raise TrainingError(msg)
        global_x = target_region.x + x
        global_y = target_region.y + y
        for identifier, surface in self._surfaces.items():
            region = self._regions[identifier]
            direction = _direction_arrow(
                global_x - (region.x + region.width / 2.0),
                global_y - (region.y + region.height / 2.0),
            )
            next_target = (x, y, diameter) if identifier == region_id else None
            prior = None
            if self._last_target is not None and self._last_target[0] == identifier:
                prior = self._last_target[1:]
            surface.show_preparation(direction, label, next_target, prior, prior_opacity)

    def show_head_diagnostic(
        self,
        frame: Frame,
        failure: HeadTrackingFailure,
        seconds_remaining: float,
    ) -> None:
        """Show transient local camera guidance on every authorized display."""
        if self._closed:
            return
        pixels = _diagnostic_pixels(frame)
        outcome = (
            f"Diagnostic interval: {seconds_remaining:.1f}s"
            if seconds_remaining > 0.0
            else "Head tracking did not recover; Gazeebo will exit"
        )
        label = f"{failure.reason}\n{outcome}"
        for surface in self._surfaces.values():
            surface.show_head_diagnostic(pixels, failure, label)

    def hide_head_diagnostic(self) -> None:
        """Remove transient diagnostic pixels after recovery."""
        if self._closed:
            return
        for surface in self._surfaces.values():
            surface.hide()

    def target_diameter(
        self,
        region_id: str,
        physical_millimetres: float,
        fallback_pixels: float,
    ) -> float:
        """Resolve selected-mode physical sizing for one exact output."""
        region = self._regions.get(region_id)
        surface = self._surfaces.get(region_id)
        if region is None or surface is None:
            msg = f"unknown calibration training region: {region_id}"
            raise TrainingError(msg)
        return target_diameter_for_region(
            region,
            surface.display_metrics(),
            physical_millimetres,
            fallback_pixels,
        )

    async def close(self) -> None:
        """Destroy every native surface idempotently."""
        if self._closed:
            return
        self._closed = True
        for surface in self._surfaces.values():
            surface.close()


def _diagnostic_pixels(frame: Frame) -> np.ndarray:
    """Borrow one camera frame as contiguous BGR pixels for transient rendering."""
    pixels = np.asarray(frame)
    if pixels.ndim == GRAYSCALE_DIMENSIONS:
        pixels = np.repeat(pixels[:, :, np.newaxis], COLOR_DIMENSIONS, axis=2)
    if pixels.ndim != COLOR_DIMENSIONS or pixels.shape[2] < COLOR_DIMENSIONS or pixels.size == 0:
        msg = "camera frame cannot be rendered for head-tracking guidance"
        raise TrainingError(msg)
    return np.ascontiguousarray(pixels[:, :, :3], dtype=np.uint8)


def target_diameter_for_region(
    region: DisplayRegion,
    metrics: DisplayModeMetrics | None,
    physical_millimetres: float,
    fallback_pixels: float,
) -> float:
    """Convert physical size through the current mode or use the pixel fallback."""
    if not math.isfinite(physical_millimetres) or physical_millimetres <= 0.0:
        msg = "physical target diameter must be finite and positive"
        raise ValueError(msg)
    if not math.isfinite(fallback_pixels) or fallback_pixels <= 0.0:
        msg = "fallback target diameter must be finite and positive"
        raise ValueError(msg)
    if metrics is None:
        return fallback_pixels
    mode_x_per_mm = metrics.mode_width / metrics.physical_width_mm
    mode_y_per_mm = metrics.mode_height / metrics.physical_height_mm
    logical_per_mode_x = region.width / metrics.mode_width
    logical_per_mode_y = region.height / metrics.mode_height
    logical_per_mm = math.sqrt(
        mode_x_per_mm * logical_per_mode_x * mode_y_per_mm * logical_per_mode_y
    )
    diameter = physical_millimetres * logical_per_mm
    if not math.isfinite(diameter) or diameter <= 0.0:
        return fallback_pixels
    return diameter


def _direction_arrow(delta_x: float, delta_y: float) -> str:
    """Choose one of eight clear arrows toward a global target."""
    if math.hypot(delta_x, delta_y) < 1.0:
        return "●"
    angle = math.atan2(delta_y, delta_x)
    sector = round(angle / (math.pi / 4.0)) % 8
    return ("→", "↘", "↓", "↙", "←", "↖", "↑", "↗")[sector]


def training_targets(
    regions: Sequence[DisplayRegion],
    count: int,
    diameters: Mapping[str, float],
    *,
    start_index: int = 0,
) -> tuple[TrainingTarget, ...]:
    """Generate deterministic unseen targets across authorized displays."""
    if not regions:
        msg = "training requires at least one authorized display"
        raise ValueError(msg)
    if count <= 0 or start_index < 0:
        msg = "training target count and start index must be non-negative"
        raise ValueError(msg)
    if set(diameters) != {region.region_id for region in regions}:
        msg = "training target sizes must cover every authorized display"
        raise ValueError(msg)
    if any(
        not math.isfinite(diameters[region.region_id])
        or diameters[region.region_id] <= 0.0
        or diameters[region.region_id] >= min(region.width, region.height)
        for region in regions
    ):
        msg = "training circles must fit within every authorized display"
        raise TrainingError(msg)
    targets: list[TrainingTarget] = []
    for index in range(start_index, start_index + count):
        region = regions[index % len(regions)]
        position_index = index // len(regions)
        if position_index < len(_TARGET_POSITIONS):
            normalized_x, normalized_y = _TARGET_POSITIONS[position_index]
        else:
            offset = position_index + 11
            normalized_x = 0.08 + 0.84 * _halton(offset, 2)
            normalized_y = 0.08 + 0.84 * _halton(offset, 3)
        diameter = diameters[region.region_id]
        radius = diameter / 2.0
        x = radius + normalized_x * (region.width - diameter)
        y = radius + normalized_y * (region.height - diameter)
        edge = (
            normalized_x <= EDGE_BOUNDARY
            or normalized_x >= 1.0 - EDGE_BOUNDARY
            or normalized_y <= EDGE_BOUNDARY
            or normalized_y >= 1.0 - EDGE_BOUNDARY
        )
        targets.append(TrainingTarget(region.region_id, x, y, diameter, edge))
    return tuple(targets)


def _halton(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0
    while index > 0:
        fraction /= base
        result += fraction * (index % base)
        index //= base
    return result


def _uniform_cvar(
    values: Sequence[float],
    tail_fraction: float = DEFAULT_CVAR_TAIL_FRACTION,
) -> float:
    """Return an interpolated uniform-weight worst-tail mean."""
    if not values or not 0.0 < tail_fraction <= MAXIMUM_CVAR_TAIL_FRACTION:
        msg = "CVaR requires values and a bounded tail fraction"
        raise ValueError(msg)
    descending = sorted(values, reverse=True)
    tail_mass = len(descending) * tail_fraction
    whole = math.floor(tail_mass)
    total = math.fsum(descending[:whole])
    remainder = tail_mass - whole
    if remainder > 0.0:
        total += remainder * descending[whole]
    return total / tail_mass


def training_metrics(measurements: Sequence[TargetMeasurement]) -> TrainingMetrics:
    """Aggregate holdout target measurements deterministically."""
    if not measurements:
        msg = "validation requires at least one target measurement"
        raise ValueError(msg)
    errors = [item.radial_error for item in measurements]
    edge_errors = [item.radial_error for item in measurements if item.edge_or_corner]
    responses = [
        item.response_seconds for item in measurements if item.response_seconds is not None
    ]
    noise_spreads = [item.noise_spread for item in measurements]
    region_errors: dict[str, list[float]] = {}
    for item in measurements:
        if item.region:
            region_errors.setdefault(item.region, []).append(item.radial_error)
    return TrainingMetrics(
        target_count=len(measurements),
        hit_count=len(responses),
        median_error=statistics.median(errors),
        edge_error=statistics.median(edge_errors or errors),
        median_response=statistics.median(responses) if responses else None,
        median_noise_spread=statistics.median(noise_spreads),
        maximum_region_error=max(
            (statistics.median(values) for values in region_errors.values()),
            default=statistics.median(errors),
        ),
        maximum_region_cvar90=max(
            (_uniform_cvar(values) for values in region_errors.values()),
            default=_uniform_cvar(errors),
        ),
    )


def _metrics_with_surprise(
    metrics: TrainingMetrics,
    selections: Sequence[RegionSelection],
    scheduler: RegionSurpriseScheduler,
) -> TrainingMetrics:
    """Attach bounded pre-target region evidence to one five-target report."""
    estimates = tuple(scheduler.estimate(key) for key in scheduler.region_keys)
    selected = tuple(selection.key.label for selection in selections)
    selected_estimates = tuple(selection.estimate for selection in selections)
    return TrainingMetrics(
        target_count=metrics.target_count,
        hit_count=metrics.hit_count,
        median_error=metrics.median_error,
        edge_error=metrics.edge_error,
        median_response=metrics.median_response,
        median_noise_spread=metrics.median_noise_spread,
        maximum_region_error=metrics.maximum_region_error,
        maximum_region_cvar90=(
            max(item.cvar90 for item in estimates) * scheduler.precision_threshold
        ),
        maximum_region_upper=(
            max(item.upper for item in estimates) * scheduler.precision_threshold
        ),
        regions_precise=scheduler.regions_precise,
        selected_regions=selected,
        regional_surprise_lower=(
            max(item.lower for item in selected_estimates) if selected_estimates else 0.0
        ),
        regional_surprise_upper=(
            max(item.upper for item in selected_estimates) if selected_estimates else 0.0
        ),
        observed_regions=scheduler.observed_regions,
        total_regions=scheduler.total_regions,
        regions_equalized=scheduler.equalized,
    )


def _candidate_metrics_are_acceptable(
    incumbent: TrainingMetrics,
    candidate: TrainingMetrics,
) -> bool:
    """Reject regressions in global, regional median, tail, or confidence quality."""
    return (
        candidate_is_acceptable(
            ValidationMetrics(incumbent.median_error, incumbent.edge_error),
            ValidationMetrics(candidate.median_error, candidate.edge_error),
        )
        and candidate.maximum_region_error <= incumbent.maximum_region_error
        and candidate.maximum_region_cvar90 <= incumbent.maximum_region_cvar90
        and candidate.maximum_region_upper <= incumbent.maximum_region_upper
    )


def _predict_with_uncertainty(
    model: GazePredictor,
    features: FeatureVector,
    context: tuple[float, ...],
) -> tuple[Point, float | None]:
    """Use bounded posterior uncertainty when one predictor exposes it."""
    method = getattr(model, "predict_with_uncertainty", None)
    if callable(method):
        point, uncertainty = method(features, context)
        return point, uncertainty
    return model.predict(features, context), None


async def show_target_preparation(  # noqa: PLR0913
    surface: TrainingSurface,
    target: TrainingTarget | PointerTarget,
    diameter: float,
    label: str,
    preparation_seconds: float,
    overlap_seconds: float,
    stop: object,
    sleep: Callable[[float], Awaitable[None]],
) -> bool:
    """Overlap and fade the prior dot while the squared next dot prepares."""
    overlap_interval = overlap_seconds / len(OVERLAP_FADE_OPACITIES)
    for opacity in OVERLAP_FADE_OPACITIES:
        if _stop_is_set(stop):
            return False
        surface.show_preparation(
            target.region_id,
            target.x,
            target.y,
            diameter,
            label,
            opacity,
        )
        await sleep(overlap_interval)
    if _stop_is_set(stop):
        return False
    surface.show_preparation(
        target.region_id,
        target.x,
        target.y,
        diameter,
        label,
        0.0,
    )
    await sleep(preparation_seconds - overlap_seconds)
    return not _stop_is_set(stop)


async def show_training_countdown(
    surface: TrainingSurface,
    status: StatusSink,
    stop: object,
    interval_seconds: float,
    sleep: Callable[[float], Awaitable[None]],
) -> bool:
    """Show one finite invocation-level countdown on every output."""
    for count in (3, 2, 1):
        if _stop_is_set(stop):
            return False
        label = f"Training starts in {count}"
        status.report(RuntimeStatus.TRAINING_COUNTDOWN, label)
        surface.show_message(label)
        await sleep(interval_seconds)
    return not _stop_is_set(stop)


async def show_training_completion(  # noqa: PLR0913
    surface: TrainingSurface,
    status: StatusSink,
    collected: int,
    maximum: int,
    result: str,
    hold_seconds: float,
    stop: object,
    sleep: Callable[[float], Awaitable[None]],
) -> None:
    """Show and report one truthful terminal training outcome."""
    label = f"Training completed!\n{collected}/{maximum} circles\n{result}"
    surface.show_message(label)
    status.report(
        RuntimeStatus.TRAINING_COMPLETED,
        f"{collected}/{maximum} circles; {result.lower()}",
    )
    if not _stop_is_set(stop):
        await sleep(hold_seconds)


async def run_adaptive_training(  # noqa: PLR0913, PLR0915
    camera: CameraCapture,
    vision: VisionEstimator,
    pointer: PointerController,
    topology: DisplayTopology,
    surface: TrainingSurface,
    status: StatusSink,
    stop: object,
    base_samples: Sequence[CalibrationSample],
    initial_model: GazePredictor,
    config: TrainingConfig,
    *,
    incumbent_model: GazePredictor | None = None,
    force_adaptation: bool = False,
    establishing_model: bool = False,
    target_offset: int = 0,
    diagnostic_factory: Callable[[Sequence[DisplayRegion]], HeadDiagnosticSurface],
    head_diagnostic_minimum: float = 0.0,
    head_recovery_timeout: float,
    failure_panel_seconds: float,
    pointer_interval: float,
    frame_interval: float,
    hud: DebugHud | None = None,
    show_countdown: bool = True,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]],
    completed_target_sink: Callable[[CollectedTarget], None] | None = None,
    surprise_scheduler: RegionSurpriseScheduler | None = None,
) -> TrainingResult | None:
    """Evaluate unseen batches, adapt after failures, and stop at precision."""
    if show_countdown and not await show_training_countdown(
        surface,
        status,
        stop,
        config.countdown_interval_seconds,
        sleep,
    ):
        await show_training_completion(
            surface,
            status,
            target_offset,
            config.maximum_targets,
            "Training interrupted",
            0.0,
            stop,
            sleep,
        )
        return None
    diameters = {
        region.region_id: surface.target_diameter(
            region.region_id,
            config.physical_target_diameter_mm,
            config.fallback_target_diameter,
        )
        for region in topology.regions
    }
    cadence = _PointerCadence(pointer_interval)
    scheduler = surprise_scheduler or RegionSurpriseScheduler(
        topology,
        config.precision_threshold,
        confidence_z=config.surprise_confidence_z,
        maximum_surprise=config.surprise_maximum,
        decay=config.surprise_decay,
        maximum_outputs=config.surprise_maximum_outputs,
        tail_fraction=config.surprise_tail_fraction,
        histogram_bins=config.surprise_histogram_bins,
    )
    incumbent_scheduler = copy.deepcopy(scheduler)
    samples = list(base_samples)
    model = initial_model
    incumbent = incumbent_model or initial_model
    provisional = IncrementalCalibration(base_samples, topology=topology)
    completed_targets: list[CollectedTarget] = []
    rounds: list[TrainingMetrics] = []
    all_measurements: list[TargetMeasurement] = []
    all_incumbent_measurements: list[TargetMeasurement] = []
    if target_offset < 0 or target_offset % config.batch_size != 0:
        msg = "training target offset must be a non-negative batch multiple"
        raise ValueError(msg)
    if target_offset >= config.maximum_targets:
        msg = "training target offset must leave room for one unseen batch"
        raise ValueError(msg)
    presented = 0
    while target_offset + presented < config.maximum_targets:
        batch_number = len(rounds) + 1
        status.report(
            RuntimeStatus.TRAINING_VALIDATING,
            f"unseen batch {batch_number}",
        )
        result = await _validate(
            camera,
            vision,
            pointer,
            topology,
            surface,
            status,
            config.batch_size,
            diameters,
            model,
            incumbent,
            provisional,
            scheduler,
            incumbent_scheduler,
            config,
            cadence,
            stop,
            diagnostic_factory,
            head_diagnostic_minimum,
            head_recovery_timeout,
            failure_panel_seconds,
            frame_interval,
            hud,
            clock,
            sleep,
            target_offset + presented,
            config.maximum_targets,
            completed_target_sink,
        )
        if isinstance(result, _ValidationInterruption):
            await show_training_completion(
                surface,
                status,
                target_offset + presented + result.completed_count,
                config.maximum_targets,
                "Training interrupted",
                0.0,
                stop,
                sleep,
            )
            return None
        presented += result.metrics.target_count
        rounds.append(result.metrics)
        status.report(
            RuntimeStatus.TRAINING_VALIDATING,
            (
                f"{result.metrics.summary(f'batch {batch_number}')}, "
                f"estimator {model.kind}, "
                f"total {target_offset + presented}/{config.maximum_targets}"
            ),
        )
        precision_met = (
            result.metrics.median_error <= config.precision_threshold
            and result.metrics.edge_error <= config.precision_threshold
            and scheduler.equalized
            and scheduler.regions_precise
        )
        all_measurements.extend(result.measurements)
        all_incumbent_measurements.extend(result.incumbent_measurements)
        aggregate_metrics = _metrics_with_surprise(
            training_metrics(all_measurements),
            (),
            scheduler,
        )
        aggregate_incumbent_metrics = _metrics_with_surprise(
            training_metrics(all_incumbent_measurements),
            (),
            incumbent_scheduler,
        )
        accepted = establishing_model or _candidate_metrics_are_acceptable(
            aggregate_incumbent_metrics,
            aggregate_metrics,
        )
        must_adapt = force_adaptation and not completed_targets
        samples.extend(result.samples)
        completed_targets.extend(result.targets)
        model = result.model
        if precision_met and accepted and not must_adapt:
            status.report(
                RuntimeStatus.ALL_DATA_REFITTING,
                f"final grouped selection over {len(samples)} compatible targets",
            )
            model = CalibrationModel.fit(
                samples,
                routing_contexts=tuple(target.context for target in completed_targets),
                topology=topology,
            )
            await show_training_completion(
                surface,
                status,
                target_offset + presented,
                config.maximum_targets,
                "Precision target met; all completed circles refitted",
                config.completion_seconds,
                stop,
                sleep,
            )
            return TrainingResult(
                model,
                tuple(rounds),
                precision_met=True,
                aggregate_metrics=aggregate_metrics,
                completed_targets=tuple(completed_targets),
                validation_targets=tuple(completed_targets),
                incumbent_metrics=aggregate_incumbent_metrics,
                model_accepted=True,
            )
        if precision_met and not accepted:
            status.report(
                RuntimeStatus.ADAPTIVE_TRAINING,
                "candidate rejected because unseen incumbent comparison regressed",
            )
        if target_offset + presented >= config.maximum_targets:
            status.report(
                RuntimeStatus.ALL_DATA_REFITTING,
                f"final grouped selection over {len(samples)} compatible targets",
            )
            model = CalibrationModel.fit(
                samples,
                routing_contexts=tuple(target.context for target in completed_targets),
                topology=topology,
            )
            message = (
                f"precision target {config.precision_threshold:.0f}px not met "
                f"after {target_offset + presented} circles"
            )
            status.report(RuntimeStatus.TRAINING_RECOMMENDED, message)
            result_message = (
                "Precision target not met; all-data refit accepted"
                if accepted
                else "Precision target not met; measurements retained; stored model unchanged"
            )
            await show_training_completion(
                surface,
                status,
                target_offset + presented,
                config.maximum_targets,
                result_message,
                config.completion_seconds,
                stop,
                sleep,
            )
            return TrainingResult(
                model,
                tuple(rounds),
                precision_met=False,
                aggregate_metrics=aggregate_metrics,
                completed_targets=tuple(completed_targets),
                validation_targets=tuple(completed_targets),
                incumbent_metrics=aggregate_incumbent_metrics,
                model_accepted=accepted,
            )
        status.report(
            RuntimeStatus.ADAPTIVE_TRAINING,
            f"adapting from reported batch {batch_number} with all completed circles",
        )
    msg = "adaptive calibration ended without a terminal batch"
    raise TrainingError(msg)


async def _validate(  # noqa: PLR0913
    camera: CameraCapture,
    vision: VisionEstimator,
    pointer: PointerController,
    topology: DisplayTopology,
    surface: TrainingSurface,
    status: StatusSink,
    target_count: int,
    diameters: Mapping[str, float],
    model: GazePredictor,
    incumbent: GazePredictor,
    provisional: IncrementalCalibration,
    scheduler: RegionSurpriseScheduler,
    incumbent_scheduler: RegionSurpriseScheduler,
    config: TrainingConfig,
    cadence: _PointerCadence,
    stop: object,
    diagnostic_factory: Callable[[Sequence[DisplayRegion]], HeadDiagnosticSurface],
    head_diagnostic_minimum: float,
    head_recovery_timeout: float,
    failure_panel_seconds: float,
    frame_interval: float,
    hud: DebugHud | None,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
    progress_offset: int,
    maximum_targets: int,
    completed_target_sink: Callable[[CollectedTarget], None] | None,
) -> _ValidationResult | _ValidationInterruption:
    measurements: list[TargetMeasurement] = []
    incumbent_measurements: list[TargetMeasurement] = []
    samples: list[CalibrationSample] = []
    collected: list[CollectedTarget] = []
    selections: list[RegionSelection] = []
    for index in range(1, target_count + 1):
        if _stop_is_set(stop):
            return _ValidationInterruption(len(collected))
        progress = progress_offset + index
        selection = scheduler.select(diameters)
        target = TrainingTarget(
            selection.target.region_id,
            selection.target.x,
            selection.target.y,
            diameters[selection.target.region_id],
            selection.key.edge_or_corner,
        )
        selections.append(selection)
        if hud is not None:
            hud.set_training_region(
                TrainingRegionStatus(
                    selection.key.label,
                    selection.estimate.cvar90,
                    selection.estimate.lower,
                    selection.estimate.upper,
                    selection.observed_regions,
                    selection.total_regions,
                    selection.mode,
                )
            )
        status.report(
            RuntimeStatus.TARGET_PREPARATION,
            f"circle {progress}/{maximum_targets}; {selection.summary}",
        )
        if not await show_target_preparation(
            surface,
            target,
            target.diameter,
            f"Prepare for circle: {progress}/{maximum_targets}",
            config.preparation_seconds,
            config.transition_overlap_seconds,
            stop,
            sleep,
        ):
            return _ValidationInterruption(len(collected))
        status.report(
            RuntimeStatus.TARGET_MEASUREMENT,
            f"circle {progress}/{maximum_targets}",
        )
        surface.show_target(
            target.region_id,
            target.x,
            target.y,
            target.diameter,
            f"Training {progress}/{maximum_targets}",
        )
        try:
            result = await _collect_target(
                camera,
                vision,
                status,
                pointer,
                topology,
                target,
                model,
                incumbent,
                config,
                cadence,
                stop,
                diagnostic_factory,
                head_diagnostic_minimum,
                head_recovery_timeout,
                failure_panel_seconds,
                frame_interval,
                hud,
                clock,
                sleep,
            )
        except HeadTrackingError as error:
            await show_training_completion(
                surface,
                status,
                progress - 1,
                maximum_targets,
                "Training failed: head tracking did not recover",
                config.completion_seconds,
                stop,
                sleep,
            )
            raise TrainingError(str(error)) from error
        if result is None:
            return _ValidationInterruption(len(collected))
        measurements.append(replace(result.measurement, region=selection.key.label))
        incumbent_measurements.append(
            replace(result.incumbent_measurement, region=selection.key.label)
        )
        features = aggregate_features(result.features)
        feature_dispersion = aggregate_feature_dispersion(result.features)
        context = aggregate_features(result.contexts)
        pointer_target = PointerTarget(target.region_id, target.x, target.y)
        sample = CalibrationSample(
            features,
            topology.to_global(pointer_target),
            target_fit_weight(result.noise),
            context,
            feature_dispersion,
        )
        samples.append(sample)
        scheduler.observe(
            pointer_target,
            result.measurement.radial_error,
            result.measurement.predictive_uncertainty,
            result.noise.p95_radial_spread,
        )
        incumbent_scheduler.observe(
            pointer_target,
            result.incumbent_measurement.radial_error,
            result.incumbent_measurement.predictive_uncertainty,
            result.noise.p95_radial_spread,
        )
        completed_target = CollectedTarget(
            features,
            context,
            pointer_target,
            _region_zone(selection.key),
            result.noise,
            feature_dispersion,
            result.measurement.radial_error,
            result.measurement.predictive_uncertainty,
        )
        collected.append(completed_target)
        if completed_target_sink is not None:
            completed_target_sink(completed_target)
        model = provisional.add(sample)
        status.report(
            RuntimeStatus.ALL_DATA_REFITTING,
            f"provisional update includes {model.sample_count} compatible targets",
        )
    metrics = _metrics_with_surprise(
        training_metrics(measurements),
        selections,
        scheduler,
    )
    return _ValidationResult(
        metrics,
        _metrics_with_surprise(
            training_metrics(incumbent_measurements),
            (),
            incumbent_scheduler,
        ),
        tuple(measurements),
        tuple(incumbent_measurements),
        tuple(samples),
        tuple(collected),
        model,
    )


async def _collect_target(  # noqa: PLR0913
    camera: CameraCapture,
    vision: VisionEstimator,
    status: StatusSink,
    pointer: PointerController,
    topology: DisplayTopology,
    target: TrainingTarget,
    model: GazePredictor,
    incumbent: GazePredictor,
    config: TrainingConfig,
    cadence: _PointerCadence,
    stop: object,
    diagnostic_factory: Callable[[Sequence[DisplayRegion]], HeadDiagnosticSurface],
    head_diagnostic_minimum: float,
    head_recovery_timeout: float,
    failure_panel_seconds: float,
    frame_interval: float,
    hud: DebugHud | None,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
) -> _TargetResult | None:
    started = clock()
    paused = 0.0
    first_hit: float | None = None
    window: list[
        tuple[
            float,
            EyeObservation,
            Point,
            float,
            float,
            float | None,
            float | None,
        ]
    ] = []
    rendered_predictions: list[Point] = []
    target_global = topology.to_global(PointerTarget(target.region_id, target.x, target.y))
    while clock() - started - paused < config.measurement_seconds:
        if _stop_is_set(stop):
            return None
        recovered = await observe_with_head_recovery(
            camera,
            vision,
            topology.regions,
            diagnostic_factory,
            status,
            stop,
            head_diagnostic_minimum,
            head_recovery_timeout,
            frame_interval,
            failure_panel_seconds,
            clock,
            sleep,
        )
        if recovered is None:
            return None
        paused += recovered.paused_seconds
        observation = recovered.observation
        if hud is not None:
            hud.set_evidence_context(observation.evidence_class)
        estimated, predictive_uncertainty = _predict_with_uncertainty(
            model,
            observation.features,
            observation.context,
        )
        incumbent_estimated, incumbent_uncertainty = _predict_with_uncertainty(
            incumbent,
            observation.features,
            observation.context,
        )
        clipped_estimate = topology.to_global(topology.locate(estimated))
        incumbent_clipped = topology.to_global(topology.locate(incumbent_estimated))
        rendered = rolling_point_median(rendered_predictions, estimated)
        pointer_target = topology.locate(rendered)
        rendered_clipped = topology.to_global(pointer_target)
        if cadence.due(observation.timestamp):
            pointer.move(pointer_target.region_id, pointer_target.x, pointer_target.y)
            cadence.visible_point = rendered_clipped
            if hud is not None:
                await hud.update(
                    pointer_target.region_id,
                    rendered_clipped.x,
                    rendered_clipped.y,
                )
        visible = cadence.visible_point or rendered_clipped
        error = math.hypot(
            clipped_estimate.x - target_global.x,
            clipped_estimate.y - target_global.y,
        )
        visible_error = math.hypot(visible.x - target_global.x, visible.y - target_global.y)
        incumbent_error = math.hypot(
            incumbent_clipped.x - target_global.x,
            incumbent_clipped.y - target_global.y,
        )
        if first_hit is None and visible_error <= target.diameter / 2.0:
            first_hit = max(0.0, observation.timestamp - started - paused)
        window.append(
            (
                observation.timestamp,
                observation,
                clipped_estimate,
                error,
                incumbent_error,
                predictive_uncertainty,
                incumbent_uncertainty,
            )
        )
        await sleep(frame_interval)
    if not window:
        msg = "head-reliable measurement completed without observations"
        raise TrainingError(msg)
    noise = cursor_noise_summary(tuple(item[2] for item in window))
    return _TargetResult(
        tuple(item[1].features for item in window),
        tuple(item[1].context for item in window),
        TargetMeasurement(
            radial_error=statistics.median(item[3] for item in window),
            edge_or_corner=target.edge_or_corner,
            response_seconds=first_hit,
            noise_spread=noise.p95_radial_spread,
            predictive_uncertainty=(
                statistics.median(item[5] for item in window if item[5] is not None)
                if any(item[5] is not None for item in window)
                else None
            ),
        ),
        TargetMeasurement(
            radial_error=statistics.median(item[4] for item in window),
            edge_or_corner=target.edge_or_corner,
            response_seconds=None,
            predictive_uncertainty=(
                statistics.median(item[6] for item in window if item[6] is not None)
                if any(item[6] is not None for item in window)
                else None
            ),
        ),
        noise,
    )


def cursor_noise_summary(points: Sequence[Point]) -> CursorNoiseSummary:
    """Reduce one stationary active window without retaining its trajectory."""
    if not points:
        msg = "cursor noise summary requires active-window predictions"
        raise ValueError(msg)
    center_x = statistics.fmean(point.x for point in points)
    center_y = statistics.fmean(point.y for point in points)
    horizontal = statistics.pstdev(point.x for point in points)
    vertical = statistics.pstdev(point.y for point in points)
    covariance = statistics.fmean((point.x - center_x) * (point.y - center_y) for point in points)
    radial = sorted(math.hypot(point.x - center_x, point.y - center_y) for point in points)
    median_radial = statistics.median(radial)
    return CursorNoiseSummary(
        sample_count=len(points),
        horizontal_dispersion=horizontal,
        vertical_dispersion=vertical,
        covariance=covariance,
        median_radial_spread=median_radial,
        p95_radial_spread=max(median_radial, _percentile(radial, 0.95)),
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Interpolate one finite sorted percentile deterministically."""
    if not values or not 0.0 <= quantile <= 1.0:
        msg = "percentile requires values and a normalized quantile"
        raise ValueError(msg)
    index = quantile * (len(values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def _region_zone(key: RegionKey) -> str:
    """Classify one normalized scheduler cell for persistent balanced fitting."""
    horizontal_edge = key.column != 1
    vertical_edge = key.row != 1
    if horizontal_edge and vertical_edge:
        return "corner"
    if horizontal_edge or vertical_edge:
        return "edge"
    return "center"


def _stop_is_set(stop: object) -> bool:
    method = getattr(stop, "is_set", None)
    return bool(method()) if callable(method) else False
