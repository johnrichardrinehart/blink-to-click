"""Tests for foreground session lifecycle and cleanup."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from collections import deque
from dataclasses import replace
from pathlib import Path

from gazeebo.contracts import DisplayRegion, EyeObservation, RuntimeStatus
from gazeebo.geometry import DisplayTopology, Point, calibration_targets
from gazeebo.runtime import (
    DISPLAY_REAUTHORIZATION_RESULT,
    TrackingConfig,
    _track,
    run_owned_session,
)
from gazeebo.state import TrainingState, TrainingStore
from gazeebo.training import TrainingConfig
from tests.fakes import (
    FakeCamera,
    FakeHud,
    FakePointer,
    FakeStatus,
    FakeTraining,
    FakeVision,
)


def observation(index: float) -> EyeObservation:
    """Create one reliable head/face calibration observation."""
    value = float(index)
    return EyeObservation(value, 1.0, 1.0, (value,), 1.0, (0.0, 0.5))


class DirectFeatureModel:
    """Map one fixture feature directly to a horizontal cursor coordinate."""

    kind = "direct-feature"

    def predict(
        self,
        features: tuple[float, ...],
        _context: tuple[float, ...] | None = None,
    ) -> Point:
        """Return the exact current estimate without animation."""
        return Point(features[0], 50.0)


class RecordingModel:
    """Record the context selected by the refresh cadence."""

    kind = "recording"

    def __init__(self, events: list[str]) -> None:
        """Initialize context and event recording."""
        self.contexts: list[tuple[float, ...] | None] = []
        self.events = events

    def predict(
        self,
        _features: tuple[float, ...],
        context: tuple[float, ...] | None = None,
    ) -> Point:
        """Record the selected context and return one safe fixture point."""
        self.events.append("predict")
        self.contexts.append(context)
        return Point(50.0, 50.0)


class FakeDisplayMonitor:
    """Return stable live geometry and record refresh ordering."""

    def __init__(self, events: list[str]) -> None:
        """Initialize refresh event recording."""
        self.events = events
        self.closed = False

    def snapshot(self) -> tuple[tuple[int, int, int, int], ...]:
        """Return stable fixture geometry."""
        self.events.append("topology")
        return ((0, 0, 100, 100),)

    def close(self) -> None:
        """Record monitor cleanup."""
        self.closed = True


class ChangingDisplayMonitor:
    """Expose one topology change between startup and the first refresh."""

    def __init__(
        self,
        before: tuple[tuple[int, int, int, int], ...],
        after: tuple[tuple[int, int, int, int], ...],
    ) -> None:
        """Retain deterministic before/after snapshots."""
        self._snapshots = [before, after]
        self.closed = False

    def snapshot(self) -> tuple[tuple[int, int, int, int], ...]:
        """Return the changed snapshot after the initial read."""
        if len(self._snapshots) > 1:
            return self._snapshots.pop(0)
        return self._snapshots[0]

    def close(self) -> None:
        """Record monitor cleanup."""
        self.closed = True


def fast_tracking() -> TrackingConfig:
    """Use one deterministic observation per calibration target."""
    return TrackingConfig(
        calibration_settle_seconds=0.0,
        calibration_samples_per_target=1,
        calibration_attempts_per_target=1,
        frame_interval_seconds=0.0,
    )


class RuntimeTests(unittest.TestCase):
    """Lock normal, failed, and topology-invalidated cleanup."""

    def test_head_diagnostic_minimum_is_finite_and_configurable(self) -> None:
        """Runtime accepts a custom minimum but rejects unsafe values."""
        assert (
            TrackingConfig(head_diagnostic_minimum_seconds=4.5).head_diagnostic_minimum_seconds
            == 4.5
        )
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            TrackingConfig(head_diagnostic_minimum_seconds=-1.0)

    def test_live_context_refresh_checks_topology_first_at_one_hertz(self) -> None:
        """Topology refresh precedes posture and lighting route updates."""
        timestamps = (0.0, 0.2, 0.9, 1.0, 1.5)
        observations: deque[EyeObservation | None] = deque(
            EyeObservation(
                timestamp,
                1.0,
                1.0,
                (0.5,),
                1.0,
                (timestamp, 0.5),
            )
            for timestamp in timestamps
        )
        events: list[str] = []
        model = RecordingModel(events)
        monitor = FakeDisplayMonitor(events)
        pointer = FakePointer((DisplayRegion("only", 0, 0, 100, 100),))
        stop = asyncio.Event()

        async def stop_after_observations(_delay: float) -> None:
            if len(pointer.moves) == len(timestamps):
                stop.set()
            await asyncio.sleep(0)

        result = asyncio.run(
            _track(
                FakeCamera(deque(object() for _ in timestamps)),
                FakeVision(observations),
                pointer,
                DisplayTopology(pointer.regions),
                model,
                None,
                FakeStatus(),
                stop,
                TrackingConfig(
                    frame_interval_seconds=0.0,
                    pointer_update_interval_seconds=0.0,
                    context_refresh_interval_seconds=1.0,
                ),
                None,
                TrainingState(),
                "fixture-camera",
                "gaze-v1",
                monitor,
                lambda _regions: FakeTraining(),
                lambda: 0.0,
                stop_after_observations,
            )
        )
        assert result == 0
        assert model.contexts == [
            (0.0, 0.5),
            (0.0, 0.5),
            (0.0, 0.5),
            (1.0, 0.5),
            (1.0, 0.5),
        ]
        assert events[:2] == ["topology", "topology"]
        assert events[2] == "predict"

    def test_added_display_pauses_before_any_new_pointer_motion(self) -> None:
        """Default reauthorization exits before predicting against stale authorization."""
        original = ((0, 0, 100, 100),)
        monitor = ChangingDisplayMonitor(original, (*original, (100, 0, 100, 100)))
        pointer = FakePointer((DisplayRegion("only", 0, 0, 100, 100),))
        result = asyncio.run(
            _track(
                FakeCamera(deque([object()])),
                FakeVision(deque([observation(0.0)])),
                pointer,
                DisplayTopology(pointer.regions),
                RecordingModel([]),
                None,
                FakeStatus(),
                asyncio.Event(),
                TrackingConfig(frame_interval_seconds=0.0),
                None,
                TrainingState(),
                "fixture-camera",
                "gaze-v1",
                monitor,
                lambda _regions: FakeTraining(),
                lambda: 0.0,
                asyncio.sleep,
            )
        )
        assert result == DISPLAY_REAUTHORIZATION_RESULT
        assert pointer.moves == []

    def test_no_pause_addition_retains_only_the_authorized_union(self) -> None:
        """Disabling pause cannot grant motion on a newly discovered output."""
        original = ((0, 0, 100, 100),)
        monitor = ChangingDisplayMonitor(original, (*original, (100, 0, 100, 100)))
        pointer = FakePointer((DisplayRegion("only", 0, 0, 100, 100),))
        stop = asyncio.Event()

        async def stop_after_move(_delay: float) -> None:
            if pointer.moves:
                stop.set()
            await asyncio.sleep(0)

        result = asyncio.run(
            _track(
                FakeCamera(deque([object()])),
                FakeVision(deque([observation(0.0)])),
                pointer,
                DisplayTopology(pointer.regions),
                RecordingModel([]),
                None,
                FakeStatus(),
                stop,
                TrackingConfig(
                    frame_interval_seconds=0.0,
                    allow_display_reauthorization_pause=False,
                ),
                None,
                TrainingState(),
                "fixture-camera",
                "gaze-v1",
                monitor,
                lambda _regions: FakeTraining(),
                lambda: 0.0,
                stop_after_move,
            )
        )
        assert result == 0
        assert pointer.moves == [("only", 50.0, 50.0)]

    def test_removed_authorized_display_stops_before_motion_without_pause(self) -> None:
        """Invalid authorized geometry stops instead of predicting on stale regions."""
        original = ((0, 0, 100, 100), (100, 0, 100, 100))
        monitor = ChangingDisplayMonitor(original, (original[0],))
        regions = (
            DisplayRegion("left", 0, 0, 100, 100),
            DisplayRegion("right", 100, 0, 100, 100),
        )
        pointer = FakePointer(regions)
        result = asyncio.run(
            _track(
                FakeCamera(deque([object()])),
                FakeVision(deque([observation(0.0)])),
                pointer,
                DisplayTopology(pointer.regions),
                RecordingModel([]),
                None,
                FakeStatus(),
                asyncio.Event(),
                TrackingConfig(
                    frame_interval_seconds=0.0,
                    allow_display_reauthorization_pause=False,
                ),
                None,
                TrainingState(),
                "fixture-camera",
                "gaze-v1",
                monitor,
                lambda _regions: FakeTraining(),
                lambda: 0.0,
                asyncio.sleep,
            )
        )
        assert result == 3
        assert pointer.moves == []

    def test_tracking_stops_and_releases_every_owned_resource(self) -> None:
        """A stop event exits with no camera, model, or pointer owner left open."""
        camera = FakeCamera(deque(object() for _ in range(6)))
        vision = FakeVision(deque(observation(index) for index in range(6)))
        pointer = FakePointer((DisplayRegion("only", 100, 200, 1000, 700),))
        hud = FakeHud()
        status = FakeStatus()
        stop = asyncio.Event()

        async def controlled_sleep(_delay: float) -> None:
            if status.reports and status.reports[-1][0] is RuntimeStatus.ACTIVE:
                stop.set()
            await asyncio.sleep(0)

        result = asyncio.run(
            run_owned_session(
                camera,
                vision,
                pointer,
                status,
                stop,
                hud=hud,
                tracking=fast_tracking(),
                sleep=controlled_sleep,
            )
        )
        assert result == 0
        assert len(pointer.moves) == 6
        assert camera.closed
        assert vision.closed
        assert pointer.closed
        assert hud.closed
        assert hud.updates[-1][0] == "only"
        assert hud.updates[-1][1] >= 100.0
        assert hud.updates[-1][2] >= 200.0
        assert [item[0] for item in status.reports][-2:] == [
            RuntimeStatus.ACTIVE,
            RuntimeStatus.STOPPED,
        ]

    def test_calibration_and_tracking_cross_authorized_displays(self) -> None:
        """One fitted session can move between every authorized region."""
        regions = (
            DisplayRegion("left", 0, 0, 1000, 700),
            DisplayRegion("right", 1000, 0, 1000, 700),
        )
        topology = DisplayTopology(regions)
        targets = calibration_targets(topology)
        observations: deque[EyeObservation | None] = deque(
            EyeObservation(
                float(index),
                1.0,
                1.0,
                (
                    topology.to_global(target).x,
                    topology.to_global(target).y,
                ),
                1.0,
                (0.0, 0.5),
            )
            for index, target in enumerate(targets)
        )
        observations.append(EyeObservation(20.0, 1.0, 1.0, (1500.0, 350.0), 1.0, (0.0, 0.5)))
        pointer = FakePointer(regions)
        status = FakeStatus()
        stop = asyncio.Event()

        async def stop_after_tracking_move(_delay: float) -> None:
            if (
                status.reports
                and status.reports[-1][0] is RuntimeStatus.ACTIVE
                and len(pointer.moves) > len(targets)
            ):
                stop.set()
            await asyncio.sleep(0)

        result = asyncio.run(
            run_owned_session(
                FakeCamera(deque(object() for _ in observations)),
                FakeVision(observations),
                pointer,
                status,
                stop,
                tracking=fast_tracking(),
                sleep=stop_after_tracking_move,
            )
        )
        assert result == 0
        assert {move[0] for move in pointer.moves[: len(targets)]} == {"left", "right"}
        assert pointer.moves[-1][0] == "right"

    def test_navigation_medians_estimated_positions_without_filtering_features(self) -> None:
        """Rendering suppresses one-frame jumps without altering model inputs."""
        observations: deque[EyeObservation | None] = deque(
            (
                EyeObservation(0.0, 1.0, 1.0, (10.0,), 1.0, (0.0, 0.5)),
                EyeObservation(1.0, 1.0, 1.0, (90.0,), 1.0, (0.0, 0.5)),
                EyeObservation(2.0, 1.0, 1.0, (90.0,), 1.0, (0.0, 0.5)),
            )
        )
        pointer = FakePointer((DisplayRegion("only", 0, 0, 100, 100),))
        stop = asyncio.Event()

        async def stop_after_three_moves(_delay: float) -> None:
            if len(pointer.moves) == 3:
                stop.set()
            await asyncio.sleep(0)

        result = asyncio.run(
            _track(
                FakeCamera(deque((object(), object(), object()))),
                FakeVision(observations),
                pointer,
                DisplayTopology(pointer.regions),
                DirectFeatureModel(),
                None,
                FakeStatus(),
                stop,
                TrackingConfig(
                    frame_interval_seconds=0.0,
                    pointer_update_interval_seconds=0.0,
                ),
                None,
                TrainingState(),
                "fixture-camera",
                "gaze-v1",
                None,
                lambda _regions: FakeTraining(),
                lambda: 0.0,
                stop_after_three_moves,
            )
        )
        assert result == 0
        assert pointer.moves == [
            ("only", 10.0, 50.0),
            ("only", 50.0, 50.0),
            ("only", 90.0, 50.0),
        ]

    def test_pointer_updates_are_rate_limited_without_throttling_observations(self) -> None:
        """The default cadence emits at most ten pointer moves per second."""
        timestamps = (0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 10.02, 10.09, 10.11)
        camera = FakeCamera(deque(object() for _ in timestamps))
        vision = FakeVision(deque(observation(value) for value in timestamps))
        pointer = FakePointer((DisplayRegion("only", 0, 0, 1000, 700),))
        status = FakeStatus()
        stop = asyncio.Event()

        async def stop_after_second_tracking_move(_delay: float) -> None:
            if (
                status.reports
                and status.reports[-1][0] is RuntimeStatus.ACTIVE
                and len(pointer.moves) == 7
            ):
                stop.set()
            await asyncio.sleep(0)

        result = asyncio.run(
            run_owned_session(
                camera,
                vision,
                pointer,
                status,
                stop,
                tracking=fast_tracking(),
                sleep=stop_after_second_tracking_move,
            )
        )
        assert result == 0
        assert len(pointer.moves) == 7
        assert len(vision.observations) == 0

    def test_calibration_failure_still_releases_resources(self) -> None:
        """An exhausted camera cannot bypass the shared cleanup path."""
        camera = FakeCamera(deque())
        vision = FakeVision(deque())
        pointer = FakePointer((DisplayRegion("only", 0, 0, 100, 100),))
        status = FakeStatus()
        training = FakeTraining()

        with self.assertRaisesRegex(EOFError, "exhausted"):
            asyncio.run(
                run_owned_session(
                    camera,
                    vision,
                    pointer,
                    status,
                    asyncio.Event(),
                    training=training,
                    tracking=fast_tracking(),
                )
            )
        assert camera.closed
        assert vision.closed
        assert pointer.closed
        assert training.closed
        assert status.reports[-1][0] is RuntimeStatus.STOPPED

    def test_completed_initial_target_survives_later_camera_failure(self) -> None:
        """A later target failure cannot discard an already completed aggregate."""
        camera = FakeCamera(deque((object(), object())))
        vision = FakeVision(deque((observation(1.0), observation(2.0))))
        now = 0.0

        def clock() -> float:
            return now

        async def advance(_delay: float) -> None:
            nonlocal now
            now += 3.0
            await asyncio.sleep(0)

        pointer = FakePointer((DisplayRegion("only", 0, 0, 100, 100),))
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            with self.assertRaisesRegex(EOFError, "exhausted"):
                asyncio.run(
                    run_owned_session(
                        camera,
                        vision,
                        pointer,
                        FakeStatus(),
                        asyncio.Event(),
                        training=FakeTraining(),
                        tracking=replace(
                            fast_tracking(),
                            calibration_attempts_per_target=3,
                        ),
                        training_config=TrainingConfig(),
                        training_store=store,
                        clock=clock,
                        sleep=advance,
                    )
                )
            persisted = store.load()
            assert len(persisted.targets) == 1
            assert persisted.targets[0].sequence == 0

    def test_closed_desktop_session_requires_recalibration(self) -> None:
        """Invalidated authorized geometry ends tracking without pointer guesses."""
        camera = FakeCamera(deque(object() for _ in range(5)))
        vision = FakeVision(deque(observation(index) for index in range(5)))
        pointer = FakePointer((DisplayRegion("only", 0, 0, 100, 100),))
        status = FakeStatus()

        async def invalidate_after_calibration(_delay: float) -> None:
            if len(pointer.moves) == 5:
                pointer.closed = True
            await asyncio.sleep(0)

        result = asyncio.run(
            run_owned_session(
                camera,
                vision,
                pointer,
                status,
                asyncio.Event(),
                tracking=fast_tracking(),
                sleep=invalidate_after_calibration,
            )
        )
        assert result == 3
        assert RuntimeStatus.RECALIBRATION_REQUIRED in [item[0] for item in status.reports]
        assert camera.closed
        assert vision.closed
        assert pointer.closed


if __name__ == "__main__":
    unittest.main()
