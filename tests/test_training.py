"""Tests for adaptive calibration targets, sampling, and validation."""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field, replace
from unittest.mock import patch

from gazeebo.calibration import CalibrationModel, CalibrationSample
from gazeebo.contracts import (
    DisplayRegion,
    EyeObservation,
    HeadTrackingFailure,
    RuntimeStatus,
)
from gazeebo.geometry import DisplayTopology, Point, PointerTarget
from gazeebo.state import CursorNoiseSummary
from gazeebo.surprise import RegionKey, RegionSurpriseScheduler
from gazeebo.training import (
    DisplayModeMetrics,
    TargetMeasurement,
    TrainingConfig,
    TrainingMetrics,
    TrainingTarget,
    _candidate_metrics_are_acceptable,
    _collect_target,
    _PointerCadence,
    _TargetResult,
    run_adaptive_training,
    target_diameter_for_region,
    training_metrics,
    training_targets,
)
from tests.fakes import FakePointer, FakeStatus


@dataclass(slots=True)
class FakeClock:
    """Advance deterministic monotonic time through injected sleeps."""

    value: float = 0.0

    def __call__(self) -> float:
        """Return current fixture time."""
        return self.value

    async def sleep(self, delay: float) -> None:
        """Advance fixture time without blocking."""
        self.value += delay
        await asyncio.sleep(0)


@dataclass(slots=True)
class FakeTrainingSurface:
    """Expose the visible target to a synthetic gaze estimator."""

    targets: list[tuple[str, float, float, float, str]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    preparations: list[tuple[str, float, float, float, str, float]] = field(default_factory=list)
    current: tuple[str, float, float] = ("", 0.0, 0.0)
    diagnostics: list[tuple[object, HeadTrackingFailure, float]] = field(default_factory=list)
    diagnostic_hides: int = 0
    closed: bool = False

    def show_target(
        self,
        region_id: str,
        x: float,
        y: float,
        diameter: float,
        label: str,
    ) -> None:
        """Record and expose one current target."""
        self.current = (region_id, x, y)
        self.targets.append((region_id, x, y, diameter, label))

    def show_message(self, label: str) -> None:
        """Record one all-output training message."""
        self.messages.append(label)

    def show_preparation(  # noqa: PLR0913
        self,
        region_id: str,
        x: float,
        y: float,
        diameter: float,
        label: str,
        prior_opacity: float,
    ) -> None:
        """Record one preparation fade step and expose its next target."""
        self.current = (region_id, x, y)
        self.preparations.append((region_id, x, y, diameter, label, prior_opacity))

    def show_head_diagnostic(
        self,
        frame: object,
        failure: HeadTrackingFailure,
        seconds_remaining: float,
    ) -> None:
        """Record one transient corrective camera view."""
        self.diagnostics.append((frame, failure, seconds_remaining))

    def hide_head_diagnostic(self) -> None:
        """Record diagnostic removal after recovery."""
        self.diagnostic_hides += 1

    def target_diameter(
        self,
        _region_id: str,
        _physical_millimetres: float,
        fallback_pixels: float,
    ) -> float:
        """Use the deterministic fallback in graphical-free tests."""
        return fallback_pixels

    async def close(self) -> None:
        """Record training-surface cleanup."""
        self.closed = True


@dataclass(slots=True)
class SyntheticCamera:
    """Return transient fixture frames indefinitely."""

    camera_id: str = "fixture-camera"

    def read(self) -> object:
        """Return one ephemeral frame marker."""
        return object()

    def close(self) -> None:
        """Satisfy the camera lifecycle contract."""


@dataclass(slots=True)
class InterruptingVision:
    """Interrupt one otherwise stable active measurement window."""

    count: int = 0

    def observe(
        self,
        _frame: object,
        timestamp: float,
    ) -> EyeObservation | HeadTrackingFailure:
        """Return one head-loss observation after a partial valid window."""
        self.count += 1
        if self.count == 4:
            return HeadTrackingFailure("fixture head moved out of frame")
        return EyeObservation(timestamp, 1.0, 1.0, (0.25, 0.25), 0.9, (0.0, 0.5))

    def close(self) -> None:
        """Satisfy the estimator lifecycle contract."""


class SequenceModel:
    """Return and record a repeating sequence of direct cursor estimates."""

    kind = "sequence"

    def __init__(self, points: tuple[Point, ...]) -> None:
        """Retain deterministic predictions for direct-motion assertions."""
        self._points = points
        self.inputs: list[tuple[float, ...]] = []
        self.predictions: list[Point] = []

    def predict(
        self,
        features: tuple[float, ...],
        _context: tuple[float, ...] | None = None,
    ) -> Point:
        """Return the next unsmoothed fixture prediction."""
        self.inputs.append(features)
        point = self._points[len(self.predictions) % len(self._points)]
        self.predictions.append(point)
        return point


@dataclass(slots=True)
class TargetVision:
    """Produce stable features for the target currently shown."""

    surface: FakeTrainingSurface
    width: float
    height: float
    unstable: bool = False
    count: int = 0

    def observe(self, frame: object, timestamp: float) -> EyeObservation:
        """Derive synthetic features from the visible target."""
        del frame
        self.count += 1
        _, x, y = self.surface.current
        offset = 0.4 * (-1.0 if self.count % 2 else 1.0)
        if self.unstable:
            offset = 1000.0 * (-1.0 if self.count % 2 else 1.0)
        return EyeObservation(
            timestamp,
            1.0,
            1.0,
            (x / self.width + offset / self.width, y / self.height),
            0.9,
            (0.0, 0.5),
        )

    def close(self) -> None:
        """Satisfy the estimator lifecycle contract."""


def base_calibration() -> tuple[tuple[CalibrationSample, ...], CalibrationModel]:
    """Create a deliberately under-scaled initial model."""
    points = (Point(100.0, 100.0), Point(500.0, 350.0), Point(900.0, 600.0))
    samples = tuple(
        CalibrationSample((point.x / 2000.0, point.y / 1400.0), point) for point in points
    )
    return samples, CalibrationModel.fit(samples)


class TrainingTests(unittest.TestCase):
    """Lock deterministic training behavior without a graphical session."""

    def test_targets_are_varied_deterministic_and_cover_all_displays(self) -> None:
        """Unseen batches vary across regions without leaving any display."""
        regions = (
            DisplayRegion("left", 0, 0, 1000, 700),
            DisplayRegion("right", 1000, 200, 800, 600),
        )
        diameters = {"left": 40.0, "right": 60.0}
        first = training_targets(regions, 10, diameters)
        repeated = training_targets(regions, 10, diameters)
        second = training_targets(regions, 5, diameters, start_index=10)
        assert first == repeated
        assert {target.region_id for target in first} == {"left", "right"}
        assert {(target.region_id, target.x, target.y) for target in first}.isdisjoint(
            (target.region_id, target.x, target.y) for target in second
        )
        assert {target.diameter for target in first} == {40.0, 60.0}
        assert any(target.edge_or_corner for target in first)
        assert any(not target.edge_or_corner for target in first)
        by_id = {region.region_id: region for region in regions}
        for target in first:
            region = by_id[target.region_id]
            radius = target.diameter / 2.0
            assert radius <= target.x <= region.width - radius
            assert radius <= target.y <= region.height - radius
        for region in regions:
            region_targets = [target for target in first if target.region_id == region.region_id]
            assert any(abs(target.x - target.diameter / 2.0) < 1e-9 for target in region_targets)
            assert any(
                abs(target.x - (region.width - target.diameter / 2.0)) < 1e-9
                for target in region_targets
            )

    def test_physical_target_size_uses_current_mode_and_safe_fallback(self) -> None:
        """Selected-mode metadata yields equal physical circles across displays."""
        left = DisplayRegion("left", 0, 0, 1920, 1080)
        right = DisplayRegion("right", 1920, 0, 1280, 720)
        left_size = target_diameter_for_region(
            left,
            DisplayModeMetrics(3840, 2160, 600, 337),
            12.0,
            72.0,
        )
        right_size = target_diameter_for_region(
            right,
            DisplayModeMetrics(2560, 1440, 400, 225),
            12.0,
            72.0,
        )
        self.assertAlmostEqual(left_size, right_size, places=1)
        assert target_diameter_for_region(left, None, 12.0, 72.0) == 72.0
        with self.assertRaisesRegex(ValueError, "physical target"):
            target_diameter_for_region(left, None, 0.0, 72.0)

    def test_iterative_defaults_and_terminal_limits_are_finite(self) -> None:
        """Default batches stop at precision or after exactly 55 circles."""
        config = TrainingConfig()
        assert config.batch_size == 5
        assert config.precision_threshold == 100.0
        assert config.maximum_targets == 55
        assert config.physical_target_diameter_mm == 12.0
        assert config.fallback_target_diameter == 72.0
        assert config.preparation_seconds == 2.0
        assert config.transition_overlap_seconds == 1.0
        assert config.measurement_seconds == 2.0
        assert config.surprise_tail_fraction == 0.10
        assert config.surprise_histogram_bins == 1024
        with self.assertRaisesRegex(ValueError, "multiple"):
            TrainingConfig(batch_size=5, maximum_targets=54)
        with self.assertRaisesRegex(ValueError, "physical target"):
            TrainingConfig(physical_target_diameter_mm=0.0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            TrainingConfig(preparation_seconds=1.0, transition_overlap_seconds=1.1)
        with self.assertRaisesRegex(ValueError, "surprise"):
            TrainingConfig(surprise_tail_fraction=0.0)
        with self.assertRaisesRegex(ValueError, "surprise"):
            TrainingConfig(surprise_tail_fraction=0.20)
        with self.assertRaisesRegex(ValueError, "surprise"):
            TrainingConfig(surprise_histogram_bins=8)

    def test_head_loss_pauses_fixed_measurement_without_discarding_samples(self) -> None:
        """Head loss pauses time while pupil-independent samples remain accepted."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        pointer = FakePointer((region,))
        clock = FakeClock()
        diagnostic = FakeTrainingSurface()
        _samples, model = base_calibration()
        result = asyncio.run(
            _collect_target(
                SyntheticCamera(),
                InterruptingVision(),
                FakeStatus(),
                pointer,
                topology,
                TrainingTarget("selected", 500.0, 350.0, 40.0, False),
                model,
                model,
                TrainingConfig(
                    batch_size=5,
                    maximum_targets=5,
                    preparation_seconds=0.03,
                    transition_overlap_seconds=0.01,
                    measurement_seconds=0.05,
                    fallback_target_diameter=40.0,
                ),
                _PointerCadence(10.0),
                asyncio.Event(),
                lambda _regions: diagnostic,
                0.03,
                0.10,
                0.0,
                0.01,
                None,
                clock,
                clock.sleep,
            )
        )
        assert result is not None
        assert result.noise.sample_count >= 5
        assert clock.value >= 0.06
        assert diagnostic.diagnostics == []
        assert diagnostic.diagnostic_hides == 0
        assert len(pointer.moves) == 1

    def test_training_separates_raw_estimates_from_rendered_cursor(self) -> None:
        """Every raw observation is scored while only position medians are rendered."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        pointer = FakePointer((region,))
        surface = FakeTrainingSurface()
        surface.current = ("selected", 500.0, 350.0)
        clock = FakeClock()
        model = SequenceModel((Point(100.0, 100.0), Point(900.0, 600.0)))
        incumbent = SequenceModel((Point(500.0, 350.0),))
        result = asyncio.run(
            _collect_target(
                SyntheticCamera(),
                TargetVision(surface, region.width, region.height),
                FakeStatus(),
                pointer,
                topology,
                TrainingTarget("selected", 500.0, 350.0, 40.0, False),
                model,
                incumbent,
                TrainingConfig(
                    batch_size=5,
                    maximum_targets=5,
                    preparation_seconds=0.03,
                    transition_overlap_seconds=0.01,
                    measurement_seconds=0.03,
                    fallback_target_diameter=40.0,
                ),
                _PointerCadence(0.0),
                asyncio.Event(),
                lambda _regions: FakeTrainingSurface(),
                0.0,
                0.10,
                0.0,
                0.01,
                None,
                clock,
                clock.sleep,
            )
        )
        assert result is not None
        assert len(model.inputs) == result.noise.sample_count
        self.assertAlmostEqual(model.inputs[0][0], 0.4996)
        self.assertAlmostEqual(model.inputs[1][0], 0.5004)
        assert pointer.moves[0] == ("selected", 100.0, 100.0)
        assert pointer.moves[1] == ("selected", 500.0, 350.0)
        assert result.noise.horizontal_dispersion > 300.0
        assert result.noise.vertical_dispersion > 200.0

    def test_metrics_keep_edge_and_response_measurements_separate(self) -> None:
        """Aggregate reporting preserves holdout error categories."""
        metrics = training_metrics(
            (
                TargetMeasurement(10.0, False, 0.4),
                TargetMeasurement(30.0, True, None),
                TargetMeasurement(20.0, True, 0.2),
            )
        )
        assert metrics.target_count == 3
        assert metrics.hit_count == 2
        assert metrics.median_error == 20.0
        assert metrics.edge_error == 25.0
        self.assertAlmostEqual(metrics.median_response or 0.0, 0.3)

    def test_adaptive_training_improves_independent_validation(self) -> None:
        """Stable training samples improve a deliberately biased model."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        pointer = FakePointer((region,))
        surface = FakeTrainingSurface()
        vision = TargetVision(surface, region.width, region.height)
        status = FakeStatus()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        config = TrainingConfig(
            batch_size=5,
            maximum_targets=15,
            precision_threshold=250.0,
            preparation_seconds=0.03,
            transition_overlap_seconds=0.01,
            measurement_seconds=0.03,
            fallback_target_diameter=40.0,
        )
        result = asyncio.run(
            run_adaptive_training(
                SyntheticCamera(),
                vision,
                pointer,
                topology,
                surface,
                status,
                asyncio.Event(),
                samples,
                initial_model,
                config,
                diagnostic_factory=lambda _regions: FakeTrainingSurface(),
                head_recovery_timeout=0.10,
                failure_panel_seconds=0.0,
                pointer_interval=0.0,
                frame_interval=0.01,
                clock=clock,
                sleep=clock.sleep,
            )
        )
        assert result is not None
        assert result.after.median_error < result.before.median_error
        assert not result.precision_met
        assert len(surface.targets) == 15
        assert not result.rounds[0].regions_equalized
        assert result.rounds[0].observed_regions == 5
        assert result.rounds[1].observed_regions == 9
        assert len(surface.preparations) == len(surface.targets) * 4
        assert [stage[-1] for stage in surface.preparations[:4]] == [1.0, 0.5, 0.0, 0.0]
        assert all(stage[3] == 40.0 for stage in surface.preparations)
        assert surface.messages[:3] == [
            "Training starts in 3",
            "Training starts in 2",
            "Training starts in 1",
        ]
        assert "Training completed!" in surface.messages[-1]
        assert all(move[0] == "selected" for move in pointer.moves)
        states = [report[0] for report in status.reports]
        assert states.count(RuntimeStatus.TRAINING_VALIDATING) >= 4
        assert RuntimeStatus.ADAPTIVE_TRAINING in states

    def test_each_target_is_unseen_then_updates_the_next_target_model(self) -> None:
        """Every target uses the pre-target model and immediately advances its sample count."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        surface = FakeTrainingSurface()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        observed_sample_counts: list[int] = []

        async def collect(
            _camera: object,
            _vision: object,
            _status: object,
            _pointer: object,
            _topology: object,
            target: TrainingTarget,
            model: object,
            _incumbent: object,
            *_args: object,
            **_kwargs: object,
        ) -> _TargetResult:
            observed_sample_counts.append(model.sample_count)  # type: ignore[attr-defined]
            features = (target.x / region.width, target.y / region.height)
            noise = CursorNoiseSummary(1, 0.0, 0.0, 0.0, 0.0, 0.0)
            measurement = TargetMeasurement(1.0, target.edge_or_corner, None)
            return _TargetResult(
                (features,),
                ((0.0, 0.5),),
                measurement,
                measurement,
                noise,
            )

        with patch("gazeebo.training._collect_target", side_effect=collect):
            result = asyncio.run(
                run_adaptive_training(
                    SyntheticCamera(),
                    TargetVision(surface, region.width, region.height),
                    FakePointer((region,)),
                    topology,
                    surface,
                    FakeStatus(),
                    asyncio.Event(),
                    samples,
                    initial_model,
                    TrainingConfig(
                        batch_size=5,
                        maximum_targets=5,
                        precision_threshold=10000.0,
                        preparation_seconds=0.03,
                        transition_overlap_seconds=0.01,
                        measurement_seconds=0.03,
                        fallback_target_diameter=40.0,
                    ),
                    diagnostic_factory=lambda _regions: FakeTrainingSurface(),
                    head_recovery_timeout=0.10,
                    failure_panel_seconds=0.0,
                    pointer_interval=0.0,
                    frame_interval=0.01,
                    clock=clock,
                    sleep=clock.sleep,
                )
            )
        assert result is not None
        assert observed_sample_counts == [3, 4, 5, 6, 7]
        assert result.model.sample_count == 8  # type: ignore[attr-defined]
        assert result.rounds[0].target_count == 5

    def test_each_next_target_uses_the_updated_high_surprise_region(self) -> None:
        """Selection reacts after each unseen update instead of precomputing a batch."""
        region = DisplayRegion("selected", 0, 0, 900, 600)
        topology = DisplayTopology((region,))
        scheduler = RegionSurpriseScheduler(topology, 100.0)
        high_key = RegionKey("selected", 2, 2)
        for row in range(3):
            for column in range(3):
                key = RegionKey("selected", row, column)
                target = PointerTarget(
                    "selected",
                    (column + 0.5) * region.width / 3.0,
                    (row + 0.5) * region.height / 3.0,
                )
                error = 900.0 if key == high_key else 40.0
                scheduler.observe(target, error, 0.0, 0.0)
                scheduler.observe(target, error, 0.0, 0.0)
        surface = FakeTrainingSurface()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        selected: list[tuple[float, float]] = []

        async def collect(
            _camera: object,
            _vision: object,
            _status: object,
            _pointer: object,
            _topology: object,
            target: TrainingTarget,
            _model: object,
            _incumbent: object,
            *_args: object,
            **_kwargs: object,
        ) -> _TargetResult:
            selected.append((target.x, target.y))
            features = (target.x / region.width, target.y / region.height)
            noise = CursorNoiseSummary(1, 0.0, 0.0, 0.0, 0.0, 0.0)
            measurement = TargetMeasurement(
                900.0,
                target.edge_or_corner,
                None,
                predictive_uncertainty=25.0,
            )
            return _TargetResult(
                (features,),
                ((0.0, 0.5),),
                measurement,
                measurement,
                noise,
            )

        status = FakeStatus()
        with patch("gazeebo.training._collect_target", side_effect=collect):
            result = asyncio.run(
                run_adaptive_training(
                    SyntheticCamera(),
                    TargetVision(surface, region.width, region.height),
                    FakePointer((region,)),
                    topology,
                    surface,
                    status,
                    asyncio.Event(),
                    samples,
                    initial_model,
                    TrainingConfig(
                        batch_size=5,
                        maximum_targets=5,
                        precision_threshold=1.0,
                        preparation_seconds=0.03,
                        transition_overlap_seconds=0.01,
                        measurement_seconds=0.03,
                        fallback_target_diameter=40.0,
                    ),
                    diagnostic_factory=lambda _regions: FakeTrainingSurface(),
                    head_recovery_timeout=0.10,
                    failure_panel_seconds=0.0,
                    pointer_interval=0.0,
                    frame_interval=0.01,
                    clock=clock,
                    sleep=clock.sleep,
                    surprise_scheduler=scheduler,
                )
            )
        assert result is not None
        assert len(set(selected)) == 5
        assert all(
            scheduler.region_for_target(PointerTarget("selected", x, y)) == high_key
            for x, y in selected
        )
        assert result.rounds[0].selected_regions == ("selected:2,2",) * 5
        assert result.rounds[0].observed_regions == 9
        assert all(target.unseen_error == 900.0 for target in result.completed_targets)
        assert all(target.predictive_uncertainty == 25.0 for target in result.completed_targets)
        assert any(
            state is RuntimeStatus.TARGET_PREPARATION
            and "high-surprise" in detail
            and "surprise" in detail
            for state, detail in status.reports
        )

    def test_interrupted_partial_batch_reports_every_retained_target(self) -> None:
        """Terminal progress includes completed targets outside a five-target report."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        surface = FakeTrainingSurface()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        stop = asyncio.Event()
        retained: list[object] = []

        async def collect(
            _camera: object,
            _vision: object,
            _status: object,
            _pointer: object,
            _topology: object,
            target: TrainingTarget,
            _model: object,
            _incumbent: object,
            *_args: object,
            **_kwargs: object,
        ) -> _TargetResult:
            features = (target.x / region.width, target.y / region.height)
            noise = CursorNoiseSummary(1, 0.0, 0.0, 0.0, 0.0, 0.0)
            measurement = TargetMeasurement(100.0, target.edge_or_corner, None)
            return _TargetResult(
                (features,),
                ((0.0, 0.5),),
                measurement,
                measurement,
                noise,
            )

        def retain(target: object) -> None:
            retained.append(target)
            if len(retained) == 3:
                stop.set()

        with patch("gazeebo.training._collect_target", side_effect=collect):
            result = asyncio.run(
                run_adaptive_training(
                    SyntheticCamera(),
                    TargetVision(surface, region.width, region.height),
                    FakePointer((region,)),
                    topology,
                    surface,
                    FakeStatus(),
                    stop,
                    samples,
                    initial_model,
                    TrainingConfig(
                        batch_size=5,
                        maximum_targets=5,
                        preparation_seconds=0.03,
                        transition_overlap_seconds=0.01,
                        measurement_seconds=0.03,
                        fallback_target_diameter=40.0,
                    ),
                    diagnostic_factory=lambda _regions: FakeTrainingSurface(),
                    head_recovery_timeout=0.10,
                    failure_panel_seconds=0.0,
                    pointer_interval=0.0,
                    frame_interval=0.01,
                    clock=clock,
                    sleep=clock.sleep,
                    completed_target_sink=retain,
                )
            )
        assert result is None
        assert len(retained) == 3
        assert "3/5 circles" in surface.messages[-1]

    def test_no_regression_rejects_tail_hidden_by_global_and_region_medians(self) -> None:
        """One bad regional tail cannot hide behind otherwise equal medians."""
        incumbent = TrainingMetrics(
            20,
            0,
            50.0,
            50.0,
            None,
            0.0,
            maximum_region_error=80.0,
            maximum_region_cvar90=100.0,
            maximum_region_upper=120.0,
        )
        candidate = replace(
            incumbent,
            maximum_region_error=70.0,
            maximum_region_cvar90=300.0,
            maximum_region_upper=350.0,
        )
        assert not _candidate_metrics_are_acceptable(incumbent, candidate)
        assert _candidate_metrics_are_acceptable(
            incumbent,
            replace(candidate, maximum_region_cvar90=90.0, maximum_region_upper=110.0),
        )

    def test_no_regression_rejects_one_bad_region_hidden_by_global_medians(self) -> None:
        """A strong global median cannot replace an incumbent after local regression."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        surface = FakeTrainingSurface()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        completed = 0

        async def collect(
            _camera: object,
            _vision: object,
            _status: object,
            _pointer: object,
            _topology: object,
            target: TrainingTarget,
            _model: object,
            _incumbent: object,
            *_args: object,
            **_kwargs: object,
        ) -> _TargetResult:
            nonlocal completed
            candidate_error = 500.0 if completed == 0 else 10.0
            completed += 1
            features = (target.x / region.width, target.y / region.height)
            noise = CursorNoiseSummary(1, 0.0, 0.0, 0.0, 0.0, 0.0)
            return _TargetResult(
                (features,),
                ((0.0, 0.5),),
                TargetMeasurement(candidate_error, target.edge_or_corner, None),
                TargetMeasurement(100.0, target.edge_or_corner, None),
                noise,
            )

        with patch("gazeebo.training._collect_target", side_effect=collect):
            result = asyncio.run(
                run_adaptive_training(
                    SyntheticCamera(),
                    TargetVision(surface, region.width, region.height),
                    FakePointer((region,)),
                    topology,
                    surface,
                    FakeStatus(),
                    asyncio.Event(),
                    samples,
                    initial_model,
                    TrainingConfig(
                        batch_size=5,
                        maximum_targets=5,
                        precision_threshold=1000.0,
                        preparation_seconds=0.03,
                        transition_overlap_seconds=0.01,
                        measurement_seconds=0.03,
                        fallback_target_diameter=40.0,
                    ),
                    incumbent_model=initial_model,
                    diagnostic_factory=lambda _regions: FakeTrainingSurface(),
                    head_recovery_timeout=0.10,
                    failure_panel_seconds=0.0,
                    pointer_interval=0.0,
                    frame_interval=0.01,
                    clock=clock,
                    sleep=clock.sleep,
                )
            )
        assert result is not None
        assert result.aggregate_metrics.median_error == 10.0
        assert result.aggregate_metrics.edge_error == 10.0
        assert result.aggregate_metrics.maximum_region_error == 500.0
        assert result.incumbent_metrics is not None
        assert result.incumbent_metrics.maximum_region_error == 100.0
        assert not result.model_accepted

    def test_no_regression_uses_every_unseen_target_not_only_last_batch(self) -> None:
        """A strong final report cannot hide poor earlier edge coverage."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        surface = FakeTrainingSurface()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        completed = 0

        async def collect(
            _camera: object,
            _vision: object,
            _status: object,
            _pointer: object,
            _topology: object,
            target: TrainingTarget,
            _model: object,
            _incumbent: object,
            *_args: object,
            **_kwargs: object,
        ) -> _TargetResult:
            nonlocal completed
            candidate_error = 1000.0 if completed < 5 else 10.0
            completed += 1
            features = (target.x / region.width, target.y / region.height)
            noise = CursorNoiseSummary(1, 0.0, 0.0, 0.0, 0.0, 0.0)
            return _TargetResult(
                (features,),
                ((0.0, 0.5),),
                TargetMeasurement(candidate_error, target.edge_or_corner, None),
                TargetMeasurement(100.0, target.edge_or_corner, None),
                noise,
            )

        with patch("gazeebo.training._collect_target", side_effect=collect):
            result = asyncio.run(
                run_adaptive_training(
                    SyntheticCamera(),
                    TargetVision(surface, region.width, region.height),
                    FakePointer((region,)),
                    topology,
                    surface,
                    FakeStatus(),
                    asyncio.Event(),
                    samples,
                    initial_model,
                    TrainingConfig(
                        batch_size=5,
                        maximum_targets=10,
                        precision_threshold=50.0,
                        preparation_seconds=0.03,
                        transition_overlap_seconds=0.01,
                        measurement_seconds=0.03,
                        fallback_target_diameter=40.0,
                    ),
                    diagnostic_factory=lambda _regions: FakeTrainingSurface(),
                    head_recovery_timeout=0.10,
                    failure_panel_seconds=0.0,
                    pointer_interval=0.0,
                    frame_interval=0.01,
                    clock=clock,
                    sleep=clock.sleep,
                )
            )
        assert result is not None
        assert result.after.median_error == 10.0
        assert result.aggregate_metrics.median_error == 505.0
        assert result.incumbent_metrics is not None
        assert result.incumbent_metrics.median_error == 100.0
        assert len(result.validation_targets) == 10
        assert not result.model_accepted

    def test_explicit_training_refits_every_completed_target(self) -> None:
        """On-demand training folds every reported target into the final fit."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        surface = FakeTrainingSurface()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        result = asyncio.run(
            run_adaptive_training(
                SyntheticCamera(),
                TargetVision(surface, region.width, region.height),
                FakePointer((region,)),
                topology,
                surface,
                FakeStatus(),
                asyncio.Event(),
                samples,
                initial_model,
                TrainingConfig(
                    batch_size=5,
                    maximum_targets=10,
                    precision_threshold=10000.0,
                    preparation_seconds=0.03,
                    transition_overlap_seconds=0.01,
                    measurement_seconds=0.03,
                    fallback_target_diameter=40.0,
                ),
                force_adaptation=True,
                diagnostic_factory=lambda _regions: FakeTrainingSurface(),
                head_recovery_timeout=0.10,
                failure_panel_seconds=0.0,
                pointer_interval=0.0,
                frame_interval=0.01,
                clock=clock,
                sleep=clock.sleep,
            )
        )
        assert result is not None
        assert not result.precision_met
        assert not result.aggregate_metrics.regions_precise
        assert len(result.completed_targets) == 10
        assert len(result.validation_targets) == 10
        assert isinstance(result.model, CalibrationModel)
        assert result.model.sample_count == len(samples) + 10
        assert len(surface.targets) == 10

    def test_maximum_failure_refits_and_retains_terminal_batch(self) -> None:
        """The last reported batch joins the final all-data candidate and corpus."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        surface = FakeTrainingSurface()
        status = FakeStatus()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        config = TrainingConfig(
            batch_size=5,
            maximum_targets=10,
            precision_threshold=1e-9,
            preparation_seconds=0.03,
            transition_overlap_seconds=0.01,
            measurement_seconds=0.03,
            fallback_target_diameter=40.0,
        )
        result = asyncio.run(
            run_adaptive_training(
                SyntheticCamera(),
                TargetVision(surface, region.width, region.height),
                FakePointer((region,)),
                topology,
                surface,
                status,
                asyncio.Event(),
                samples,
                initial_model,
                config,
                diagnostic_factory=lambda _regions: FakeTrainingSurface(),
                head_recovery_timeout=0.10,
                failure_panel_seconds=0.0,
                pointer_interval=0.0,
                frame_interval=0.01,
                clock=clock,
                sleep=clock.sleep,
            )
        )
        assert result is not None
        assert not result.precision_met
        assert len(result.completed_targets) == 10
        assert len(result.validation_targets) == 10
        assert isinstance(result.model, CalibrationModel)
        assert result.model.sample_count == len(samples) + 10
        assert len(surface.targets) == 10
        assert surface.targets[-1][-1] == "Training 10/10"
        assert "10/10 circles" in surface.messages[-1]
        assert "Precision target not met" in surface.messages[-1]
        states = [item[0] for item in status.reports]
        assert states.count(RuntimeStatus.ADAPTIVE_TRAINING) == 1
        assert states.count(RuntimeStatus.ALL_DATA_REFITTING) == 11
        assert RuntimeStatus.TRAINING_RECOMMENDED in states
        assert RuntimeStatus.TRAINING_COMPLETED in states

    def test_prior_initial_targets_count_toward_invocation_maximum(self) -> None:
        """Initial anchors leave only complete unseen batches inside the 55-target cap."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        surface = FakeTrainingSurface()
        status = FakeStatus()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        result = asyncio.run(
            run_adaptive_training(
                SyntheticCamera(),
                TargetVision(surface, region.width, region.height),
                FakePointer((region,)),
                topology,
                surface,
                status,
                asyncio.Event(),
                samples,
                initial_model,
                TrainingConfig(
                    batch_size=5,
                    maximum_targets=10,
                    precision_threshold=1e-9,
                    preparation_seconds=0.03,
                    transition_overlap_seconds=0.01,
                    measurement_seconds=0.03,
                    fallback_target_diameter=40.0,
                ),
                target_offset=5,
                diagnostic_factory=lambda _regions: FakeTrainingSurface(),
                head_recovery_timeout=0.10,
                failure_panel_seconds=0.0,
                pointer_interval=0.0,
                frame_interval=0.01,
                clock=clock,
                sleep=clock.sleep,
            )
        )
        assert result is not None
        assert not result.precision_met
        assert len(surface.targets) == 5
        assert any("total 10/10" in detail for _state, detail in status.reports)
        assert surface.targets[0][-1] == "Training 6/10"

    def test_noisy_pupil_samples_do_not_gate_training(self) -> None:
        """Prediction noise is summarized without retrying or rejecting a target."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        surface = FakeTrainingSurface()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        config = TrainingConfig(
            batch_size=5,
            maximum_targets=5,
            precision_threshold=10000.0,
            preparation_seconds=0.03,
            transition_overlap_seconds=0.01,
            measurement_seconds=0.03,
            fallback_target_diameter=40.0,
        )
        status = FakeStatus()
        result = asyncio.run(
            run_adaptive_training(
                SyntheticCamera(),
                TargetVision(surface, region.width, region.height, unstable=True),
                FakePointer((region,)),
                topology,
                surface,
                status,
                asyncio.Event(),
                samples,
                initial_model,
                config,
                diagnostic_factory=lambda _regions: FakeTrainingSurface(),
                head_recovery_timeout=0.10,
                failure_panel_seconds=0.0,
                pointer_interval=0.1,
                frame_interval=0.01,
                clock=clock,
                sleep=clock.sleep,
            )
        )
        assert result is not None
        assert result.rounds[0].median_noise_spread > 0.0
        assert all("retry" not in detail for _state, detail in status.reports)
        assert status.reports[-1][0] is RuntimeStatus.TRAINING_COMPLETED
