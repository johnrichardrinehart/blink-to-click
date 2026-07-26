"""Tests for persisted model reuse in the foreground runtime."""

from __future__ import annotations

import asyncio
import math
import tempfile
import unittest
from collections import deque
from dataclasses import replace
from pathlib import Path

from gazeebo.calibration import CalibrationModel, CalibrationSample
from gazeebo.contexts import build_router
from gazeebo.contracts import DisplayRegion, EyeObservation, RuntimeStatus
from gazeebo.geometry import DisplayTopology, Point, calibration_targets
from gazeebo.runtime import (
    FEATURE_SCHEMA,
    TrackingConfig,
    _persist_targets,
    _report_persistent_metrics,
    run_owned_session,
)
from gazeebo.state import TrainingState, TrainingStore, ValidationSummary
from gazeebo.training import CollectedTarget, TrainingConfig, TrainingMetrics
from tests.fakes import FakeCamera, FakePointer, FakeStatus, FakeTraining, FakeVision


def collected(topology: DisplayTopology) -> tuple[CollectedTarget, ...]:
    """Create a complete one-display training set with normalized context."""
    return tuple(
        CollectedTarget(
            (
                topology.to_global(target).x,
                topology.to_global(target).y,
            ),
            (0.0, 0.5),
            target,
            "center" if index == 2 else "corner",
        )
        for index, target in enumerate(calibration_targets(topology))
    )


def persisted_state(store: TrainingStore, topology: DisplayTopology) -> TrainingState:
    """Persist one synthetic accepted all-data fixture model."""
    targets = collected(topology)
    model = CalibrationModel.fit(
        [
            CalibrationSample(target.features, topology.to_global(target.target))
            for target in targets
        ]
    )
    persisted = _persist_targets(
        store,
        TrainingState(),
        topology,
        "fixture-camera",
        FEATURE_SCHEMA,
        targets,
        0.0,
        0.0,
        validation_targets=targets,
        validated_model=model,
    )
    assert persisted is not None
    return persisted[1]


class _ContextIncumbent:
    """Decode test holdout coordinates supplied only to the incumbent."""

    kind = "fixture"

    def predict(
        self,
        _features: tuple[float, ...],
        context: tuple[float, ...] | None = None,
    ) -> Point:
        assert context is not None
        return Point(context[0] * 1000.0, context[1] * 700.0)


def fast_reuse() -> TrackingConfig:
    """Use one passive context and one tracking observation."""
    return TrackingConfig(
        calibration_settle_seconds=0.0,
        calibration_samples_per_target=1,
        calibration_attempts_per_target=1,
        startup_context_samples=1,
        startup_context_attempts=1,
        frame_interval_seconds=0.0,
    )


class _Clock:
    """Advance deterministic adaptive-training time through awaited sleeps."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    async def sleep(self, delay: float) -> None:
        self.value += max(delay, 0.01)
        await asyncio.sleep(0)


class _TargetVision:
    """Observe the active fake target with one stable routing context."""

    def __init__(self, training: FakeTraining, context: tuple[float, ...] = (0.0, 0.5)) -> None:
        self.training = training
        self.context = context
        self.closed = False

    def observe(self, _frame: object, timestamp: float) -> EyeObservation:
        if self.training.targets:
            _region, x, y, _diameter, _label = self.training.targets[-1]
        else:
            x, y = 500.0, 350.0
        context = self.context if len(self.training.messages) >= 2 else (0.0, 0.5)
        return EyeObservation(timestamp, 1.0, 1.0, (x, y), 1.0, context)

    def close(self) -> None:
        self.closed = True


class _EndlessCamera:
    """Supply disposable frames until the runtime closes it."""

    camera_id = "fixture-camera"

    def __init__(self) -> None:
        self.closed = False

    def read(self) -> object:
        return object()

    def close(self) -> None:
        self.closed = True


class PersistenceRuntimeTests(unittest.TestCase):
    """Lock atomic candidate commits and calibration-free repeated startup."""

    def test_unvalidated_targets_are_retained_without_a_model_anchor(self) -> None:
        """Rejected model evidence still appends every completed target aggregate."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gazeebo" / "training-v1.json"
            targets = collected(topology)
            targets = (
                replace(
                    targets[0],
                    unseen_error=123.0,
                    predictive_uncertainty=45.0,
                ),
                *targets[1:],
            )
            persisted = _persist_targets(
                TrainingStore(path),
                TrainingState(),
                topology,
                "fixture-camera",
                "gaze-v3",
                targets,
                80.0,
                90.0,
            )
            assert persisted is not None
            assert len(persisted[1].targets) == 5
            assert persisted[1].targets[0].unseen_error == 123.0
            assert persisted[1].targets[0].predictive_uncertainty == 45.0
            assert persisted[1].anchors == []
            assert TrainingStore(path).load().targets == persisted[1].targets

    def test_all_data_training_persists_a_context_model_anchor(self) -> None:
        """Each accepted invocation retains its final all-data model and context."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        targets = collected(topology)
        validated = CalibrationModel.fit(
            [
                CalibrationSample(target.features, topology.to_global(target.target))
                for target in targets
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            persisted = _persist_targets(
                store,
                TrainingState(),
                topology,
                "fixture-camera",
                "gaze-v3",
                targets,
                80.0,
                90.0,
                123.0,
                150.0,
                180.0,
                validation_targets=targets[:3],
                validated_model=validated,
            )
            assert persisted is not None
            _router, candidate = persisted
            assert len(candidate.anchors) == 1
            anchor = candidate.anchors[0]
            expected_record = validated.to_record()
            expected_record["validation_target_count"] = 3
            assert anchor.model == expected_record
            assert anchor.context_centroid == (0.0, 0.5)
            assert anchor.median_error == candidate.validations[-1].median_error
            assert candidate.validations[-1].maximum_region_error == 123.0
            assert candidate.validations[-1].maximum_region_cvar90 == 150.0
            assert candidate.validations[-1].maximum_region_upper == 180.0
            assert store.load().anchors == candidate.anchors

    def test_exact_restart_loads_the_terminally_validated_coefficients(self) -> None:
        """Exact topology reuse does not replace measured coefficients by refitting."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        targets = collected(topology)
        validated = CalibrationModel.fit(
            [
                CalibrationSample(
                    target.features,
                    Point(
                        topology.to_global(target.target).x + 40.0,
                        topology.to_global(target.target).y + 20.0,
                    ),
                )
                for target in targets
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            persisted = _persist_targets(
                store,
                TrainingState(),
                topology,
                "fixture-camera",
                "gaze-v3",
                targets,
                80.0,
                90.0,
                validation_targets=targets,
                validated_model=validated,
            )
            assert persisted is not None
            before = persisted[0].predict((500.0, 350.0), (0.0, 0.5))
            after = build_router(
                store.load(),
                topology,
                camera_id="fixture-camera",
                feature_schema="gaze-v3",
            ).predict((500.0, 350.0), (0.0, 0.5))
            assert math.isclose(after.x, before.x)
            assert math.isclose(after.y, before.y)
            assert after.x > 500.0
            assert after.y > 350.0

    def test_rejected_candidate_retains_targets_but_keeps_model_anchor(self) -> None:
        """No-regression failure preserves new evidence without replacing the model."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        bad_holdout = tuple(
            CollectedTarget(
                (0.0, 0.0),
                (target.x / 1000.0, target.y / 700.0),
                target,
                "corner",
            )
            for target in calibration_targets(topology)[:3]
        )
        bad_model = CalibrationModel.fit(
            [CalibrationSample(target.features, Point(0.0, 0.0)) for target in collected(topology)]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gazeebo" / "training-v1.json"
            store = TrainingStore(path)
            existing = persisted_state(store, topology)
            result = _persist_targets(
                store,
                existing,
                topology,
                "fixture-camera",
                "gaze-v3",
                collected(topology),
                500.0,
                500.0,
                validation_targets=bad_holdout,
                incumbent_metrics=TrainingMetrics(3, 0, 10.0, 10.0, None, 0.0),
                model_accepted=False,
                validated_model=bad_model,
            )
            assert result is not None
            assert len(result[1].targets) == len(existing.targets) + 5
            assert result[1].anchors == existing.anchors
            assert store.load().anchors == existing.anchors

    def test_finite_terminal_result_can_establish_first_model(self) -> None:
        """A first below-threshold run remains available for later improvement."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        holdout = tuple(
            CollectedTarget((0.0, 0.0), (0.0, 0.5), target, "corner")
            for target in calibration_targets(topology)[:3]
        )
        zero_model = CalibrationModel.fit(
            [CalibrationSample(target.features, Point(0.0, 0.0)) for target in collected(topology)]
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            result = _persist_targets(
                store,
                TrainingState(),
                topology,
                "fixture-camera",
                "gaze-v3",
                collected(topology),
                500.0,
                600.0,
                validation_targets=holdout,
                validated_model=zero_model,
            )
            assert result is not None
            assert store.load().validations[-1].median_error == 500.0

    def test_saved_router_reports_its_actual_aggregate_holdout(self) -> None:
        """A training-model score cannot be mistaken for the persisted router's score."""
        state = TrainingState(
            validations=[ValidationSummary(5, "camera", "topology", "global", 123.4, 234.5)]
        )
        status = FakeStatus()
        _report_persistent_metrics(status, state)
        assert status.reports == [
            (
                RuntimeStatus.TRAINING_VALIDATING,
                "persistent routing: median error 123px, edge error 234px, "
                "worst region 0px, worst CVaR90 n/a, maximum bound n/a",
            )
        ]

    def test_repeated_run_uses_passive_context_without_calibration_targets(self) -> None:
        """A compatible store starts navigation after passive model routing."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            state = persisted_state(store, topology)
            observations: deque[EyeObservation | None] = deque(
                (
                    EyeObservation(0.0, 1.0, 1.0, (500.0, 350.0), 1.0, (0.0, 0.5)),
                    EyeObservation(1.0, 1.0, 1.0, (500.0, 350.0), 1.0, (0.0, 0.5)),
                )
            )
            camera = FakeCamera(deque((object(), object())))
            pointer = FakePointer(topology.regions)
            status = FakeStatus()
            stop = asyncio.Event()

            async def stop_after_move(_delay: float) -> None:
                if pointer.moves and status.reports[-1][0] is RuntimeStatus.ACTIVE:
                    stop.set()
                await asyncio.sleep(0)

            result = asyncio.run(
                run_owned_session(
                    camera,
                    FakeVision(observations),
                    pointer,
                    status,
                    stop,
                    tracking=fast_reuse(),
                    training_state=state,
                    sleep=stop_after_move,
                )
            )
            assert result == 0
            states = [item[0] for item in status.reports]
            assert RuntimeStatus.SELECTING_MODEL in states
            assert RuntimeStatus.INITIAL_TRAINING not in states
            assert pointer.moves
            assert camera.closed
            assert pointer.closed

    def test_active_request_transitions_navigation_into_training_and_back(self) -> None:
        """An active event runs on-demand targets without replacing the process."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            state = persisted_state(store, topology)
        training = FakeTraining()
        camera = _EndlessCamera()
        vision = _TargetVision(training, (0.2, 0.5))
        pointer = FakePointer(topology.regions)
        status = FakeStatus()
        stop = asyncio.Event()
        request = asyncio.Event()
        request.set()
        clock = _Clock()

        async def stop_after_second_active(delay: float) -> None:
            await clock.sleep(delay)
            active_count = sum(item[0] is RuntimeStatus.ACTIVE for item in status.reports)
            if active_count >= 2 and pointer.moves:
                stop.set()

        result = asyncio.run(
            run_owned_session(
                camera,
                vision,
                pointer,
                status,
                stop,
                tracking=fast_reuse(),
                training_config=TrainingConfig(
                    batch_size=5,
                    maximum_targets=10,
                    precision_threshold=10000.0,
                    preparation_seconds=0.03,
                    transition_overlap_seconds=0.01,
                    measurement_seconds=0.03,
                    fallback_target_diameter=40.0,
                ),
                training_state=state,
                training_requested_event=request,
                training_factory=lambda _regions: training,
                clock=clock,
                sleep=stop_after_second_active,
            )
        )
        assert result == 0
        assert len(training.targets) == 10
        assert training.messages[:3] == [
            "Training starts in 3",
            "Training starts in 2",
            "Training starts in 1",
        ]
        assert "Training completed!" in training.messages[-1]
        assert not any("posture" in message.lower() for message in training.messages)
        assert sum(item[0] is RuntimeStatus.ACTIVE for item in status.reports) == 2
        assert not request.is_set()
        assert training.closed
        assert camera.closed
        assert vision.closed
        assert pointer.closed

    def test_added_output_uses_weak_best_effort_model_without_forced_training(self) -> None:
        """A changed topology remains usable but is explicitly unvalidated."""
        source = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            state = persisted_state(store, source)
        current = DisplayTopology(
            (
                DisplayRegion("stable", 0, 0, 1000, 700),
                DisplayRegion("added", 1200, 0, 800, 600),
            )
        )
        observations: deque[EyeObservation | None] = deque(
            (
                EyeObservation(0.0, 1.0, 1.0, (500.0, 350.0), 1.0, (0.0, 0.5)),
                EyeObservation(1.0, 1.0, 1.0, (500.0, 350.0), 1.0, (0.0, 0.5)),
            )
        )
        pointer = FakePointer(current.regions)
        status = FakeStatus()
        stop = asyncio.Event()

        async def stop_after_move(_delay: float) -> None:
            if pointer.moves and status.reports[-1][0] is RuntimeStatus.ACTIVE:
                stop.set()
            await asyncio.sleep(0)

        result = asyncio.run(
            run_owned_session(
                FakeCamera(deque((object(), object()))),
                FakeVision(observations),
                pointer,
                status,
                stop,
                tracking=fast_reuse(),
                training_state=state,
                sleep=stop_after_move,
            )
        )
        assert result == 0
        states = [item[0] for item in status.reports]
        assert RuntimeStatus.TOPOLOGY_UNVALIDATED in states
        assert RuntimeStatus.INITIAL_TRAINING not in states
        assert all(
            any(
                region.region_id == region_id and 0 <= x < region.width and 0 <= y < region.height
                for region in current.regions
            )
            for region_id, x, y in pointer.moves
        )


if __name__ == "__main__":
    unittest.main()
