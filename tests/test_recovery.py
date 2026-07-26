"""Tests for bounded head-tracking recovery and transient diagnostics."""

from __future__ import annotations

import asyncio
import unittest
from collections import deque
from dataclasses import dataclass

from gazeebo.contracts import DisplayRegion, EyeObservation, HeadTrackingFailure, RuntimeStatus
from gazeebo.recovery import HeadTrackingError, observe_with_head_recovery
from tests.fakes import FakeCamera, FakeStatus, FakeTraining, FakeVision


@dataclass(slots=True)
class Clock:
    """Advance deterministic time only through injected sleeps."""

    value: float = 0.0

    def __call__(self) -> float:
        """Return current fixture time."""
        return self.value

    async def sleep(self, delay: float) -> None:
        """Advance fixture time without blocking."""
        self.value += delay
        await asyncio.sleep(0)


class RecoveryTests(unittest.TestCase):
    """Lock automatic recovery, timeout, rendering, and cleanup."""

    def test_one_frame_miss_pauses_without_flashing_diagnostic(self) -> None:
        """A transient detector miss is not presented as sustained head loss."""
        failure = HeadTrackingFailure("recenter")
        observation = EyeObservation(0.01, 1.0, 1.0, (0.5, 0.5), 0.9, (0.0, 0.5))
        clock = Clock()
        factory_calls: list[object] = []

        def diagnostic_factory(_regions: object) -> FakeTraining:
            factory_calls.append(object())
            return FakeTraining()

        recovered = asyncio.run(
            observe_with_head_recovery(
                FakeCamera(deque((object(), object()))),
                FakeVision(deque((failure, observation))),
                (DisplayRegion("only", 0, 0, 100, 100),),
                diagnostic_factory,
                FakeStatus(),
                asyncio.Event(),
                3.0,
                10.0,
                0.01,
                0.0,
                clock,
                clock.sleep,
            )
        )
        assert recovered is not None
        self.assertAlmostEqual(recovered.paused_seconds, 0.01)
        assert factory_calls == []

    def test_subsecond_miss_pauses_without_starting_sustained_warning(self) -> None:
        """Six measured transient misses recover before warning UI or capture."""
        failure = HeadTrackingFailure("tracking lost")
        observation = EyeObservation(0.6, 1.0, 1.0, (0.5, 0.5), 0.9, (0.0, 0.5))
        clock = Clock()
        factory_calls: list[object] = []
        status = FakeStatus()

        def diagnostic_factory(_regions: object) -> FakeTraining:
            factory_calls.append(object())
            return FakeTraining()

        recovered = asyncio.run(
            observe_with_head_recovery(
                FakeCamera(deque(object() for _ in range(7))),
                FakeVision(deque((*([failure] * 6), observation))),
                (DisplayRegion("only", 0, 0, 100, 100),),
                diagnostic_factory,
                status,
                asyncio.Event(),
                3.0,
                10.0,
                0.1,
                0.0,
                clock,
                clock.sleep,
            )
        )
        assert recovered is not None
        self.assertAlmostEqual(recovered.paused_seconds, 0.6)
        assert factory_calls == []
        assert status.reports == []

    def test_recovery_renders_then_returns_head_only_observation(self) -> None:
        """Missing pupils do not block recovery once head geometry is reliable."""
        failure = HeadTrackingFailure("recenter", (0.1, 0.1, 0.5, 0.7), (1.0, 2.0, 3.0))
        head_only = EyeObservation(
            0.01,
            0.0,
            0.0,
            (0.5, 0.5),
            0.9,
            (0.0, 0.5),
            pupil_available=False,
            pupil_confidence=0.0,
        )
        clock = Clock()
        surface = FakeTraining()
        recovered = asyncio.run(
            observe_with_head_recovery(
                FakeCamera(deque(object() for _ in range(4))),
                FakeVision(deque((failure, head_only, head_only, head_only))),
                (DisplayRegion("only", 0, 0, 100, 100),),
                lambda _regions: surface,
                FakeStatus(),
                asyncio.Event(),
                0.03,
                0.1,
                0.01,
                0.0,
                clock,
                clock.sleep,
                0.0,
            )
        )
        assert recovered is not None
        assert not recovered.observation.pupil_available
        self.assertAlmostEqual(recovered.paused_seconds, 0.03)
        assert len(surface.diagnostics) == 4
        assert surface.diagnostic_hides == 1
        assert surface.closed

    def test_timeout_keeps_motion_blocked_and_closes_diagnostic(self) -> None:
        """Unrecovered head loss shows guidance for the finite timeout then fails."""
        failure = HeadTrackingFailure("move fully into frame")
        clock = Clock()
        surface = FakeTraining()
        status = FakeStatus()
        with self.assertRaisesRegex(HeadTrackingError, "move fully"):
            asyncio.run(
                observe_with_head_recovery(
                    FakeCamera(deque(object() for _ in range(5))),
                    FakeVision(deque(failure for _ in range(5))),
                    (DisplayRegion("only", 0, 0, 100, 100),),
                    lambda _regions: surface,
                    status,
                    asyncio.Event(),
                    0.03,
                    0.03,
                    0.01,
                    0.02,
                    clock,
                    clock.sleep,
                    0.0,
                )
            )
        assert len(surface.diagnostics) == 4
        assert surface.diagnostics[-1][2] == 0.0
        assert surface.closed
        assert clock.value >= 0.05
        assert [item[0] for item in status.reports] == [
            RuntimeStatus.HEAD_TRACKING_RECOVERY,
            RuntimeStatus.CAMERA_ERROR,
        ]


if __name__ == "__main__":
    unittest.main()
