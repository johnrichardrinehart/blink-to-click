"""Tests for bounded confidence regions and post-rough-in refinement."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

from gazeebo.adaptation import make_stored_target
from gazeebo.contracts import DisplayRegion, RuntimeStatus
from gazeebo.control import ControlCommand
from gazeebo.geometry import DisplayTopology, Point, PointerTarget
from gazeebo.refinement import (
    ConfidenceRectangle,
    ConfidenceRegionEstimator,
    RefinementConfig,
    RefinementController,
    RefinementSession,
    SettledPositionRecorder,
    refinement_rows,
)
from tests.fakes import FakeHud, FakePointer, FakeStatus, FakeTraining

if TYPE_CHECKING:
    from collections.abc import Callable

    from gazeebo.state import StoredTarget


class ConfidenceRegionTests(unittest.TestCase):
    """Lock p99 evidence hierarchy, migration fallback, and grid geometry."""

    def setUp(self) -> None:
        """Create two authorized outputs separated by partial gaps."""
        self.topology = DisplayTopology(
            (
                DisplayRegion("left", 0, 0, 900, 600),
                DisplayRegion("right", 900, 100, 600, 400),
            )
        )

    def test_region_component_p99_doubles_conservative_histogram_edges(self) -> None:
        """A sufficiently sampled cell supplies independent full width and height."""
        targets = [
            self._target(index, "left", 100.0, 100.0, horizontal=50.0, vertical=-25.0)
            for index in range(3)
        ]
        estimator = ConfidenceRegionEstimator(
            self.topology,
            targets,
            camera_id="camera",
            feature_schema="schema",
            config=RefinementConfig(
                minimum_samples=3,
                histogram_bins=100,
                maximum_residual=100.0,
            ),
        )
        rectangle = estimator.rectangle(Point(100.0, 100.0))
        assert rectangle.source == "region+authorized-union-intersection"
        assert rectangle.samples == 3
        assert rectangle.width == 102.0
        assert rectangle.height == 52.0
        assert rectangle.center == Point(100.0, 100.0)

    def test_sparse_region_falls_back_to_output_then_global_components(self) -> None:
        """Specific evidence is used only after its deterministic sample minimum."""
        targets = [
            self._target(index, "left", 100.0, 100.0, horizontal=60.0, vertical=30.0)
            for index in range(3)
        ]
        output = ConfidenceRegionEstimator(
            self.topology,
            targets,
            camera_id="camera",
            feature_schema="schema",
            config=RefinementConfig(minimum_samples=3),
        ).rectangle(Point(800.0, 500.0))
        assert output.source == "output+authorized-union-intersection"
        right = ConfidenceRegionEstimator(
            self.topology,
            targets,
            camera_id="camera",
            feature_schema="schema",
            config=RefinementConfig(minimum_samples=3),
        ).rectangle(Point(1200.0, 300.0))
        assert right.source.startswith("global+")

    def test_legacy_radial_evidence_is_a_conservative_square(self) -> None:
        """Schema-eight targets remain useful without fabricated components."""
        targets = [self._target(index, "left", 100.0, 100.0, radial=200.0) for index in range(3)]
        rectangle = ConfidenceRegionEstimator(
            self.topology,
            targets,
            camera_id="camera",
            feature_schema="schema",
            config=RefinementConfig(
                minimum_samples=3,
                histogram_bins=100,
                maximum_residual=1000.0,
            ),
        ).rectangle(Point(100.0, 100.0))
        assert rectangle.source.startswith("legacy-region+")
        assert rectangle.width == rectangle.height == 420.0

    def test_overrides_and_empty_fallback_are_finite_and_projected(self) -> None:
        """Explicit dimensions win while absent evidence uses topology bounds."""
        fallback = ConfidenceRegionEstimator(
            self.topology,
            (),
            camera_id="camera",
            feature_schema="schema",
        ).rectangle(Point(700.0, 900.0))
        assert fallback.source.startswith("topology-fallback+")
        assert fallback.width == 1500.0
        assert fallback.height == 600.0
        assert fallback.center == Point(700.0, 599.9999999999999)
        overridden = ConfidenceRegionEstimator(
            self.topology,
            (),
            camera_id="camera",
            feature_schema="schema",
            config=RefinementConfig(width_override=300.0, height_override=150.0),
        ).rectangle(Point(100.0, 100.0))
        assert overridden.width == 300.0
        assert overridden.height == 150.0
        assert overridden.source.startswith("override-height+override-width+")

    def test_topology_scaling_remaps_component_residuals_per_axis(self) -> None:
        """A resized output scales horizontal and vertical evidence independently."""
        source = DisplayTopology((DisplayRegion("same", 0, 0, 900, 600),))
        target = make_stored_target(
            0,
            "camera",
            "schema",
            (0.1,),
            (0.2,),
            source,
            PointerTarget("same", 450.0, 300.0),
            "center",
            horizontal_residual=50.0,
            vertical_residual=-50.0,
        )
        current = DisplayTopology((DisplayRegion("same", 0, 0, 1800, 900),))
        rectangle = ConfidenceRegionEstimator(
            current,
            (target,),
            camera_id="camera",
            feature_schema="schema",
            config=RefinementConfig(
                minimum_samples=1,
                histogram_bins=100,
                maximum_residual=200.0,
            ),
        ).rectangle(Point(900.0, 450.0))
        assert rectangle.width == 204.0
        assert rectangle.height == 152.0

    def test_configured_matrix_recurses_row_major_and_projects_gap_centers(self) -> None:
        """A 3x4 keyboard matrix subdivides safely and stays in the union."""
        config = RefinementConfig(
            maximum_depth=2,
            minimum_cell_size=10.0,
            rows=("i,.p", "aoeu", ";qjk"),
        )
        session = RefinementSession(self.topology, config)
        start = session.start(ConfidenceRectangle(750.0, 0.0, 600.0, 600.0, "test", 100))
        assert self.topology.region_containing(start) is not None
        selected = session.select("k")
        assert session.depth == 1
        assert session.rectangle == ConfidenceRectangle(
            1200.0,
            400.0,
            150.0,
            200.0,
            "test",
            100,
        )
        assert self.topology.region_containing(selected) is not None
        session.select("i")
        with self.assertRaisesRegex(ValueError, "maximum depth"):
            session.select("i")
        accepted = session.accept()
        assert self.topology.region_containing(accepted) is not None
        assert not session.active

    def test_matrix_config_validates_shape_labels_and_cli_precedence(self) -> None:
        """TOML rows are replaceable only by one complete valid CLI matrix."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[refinement]\nrows = ["i,.p", "aoeu", ";qjk"]\n')
            assert refinement_rows(None, path=path) == ("i,.p", "aoeu", ";qjk")
            assert refinement_rows(("12", "34"), path=path) == ("12", "34")
        custom = RefinementConfig(rows=("i,.p", "aoeu", ";qjk"))
        assert custom.locate_label(";") == (2, 0)
        with self.assertRaisesRegex(ValueError, "not active"):
            custom.locate_label("1")
        for rows in (
            ("12",),
            ("12", "345"),
            ("11", "23"),
            ("1 ", "23"),
            ("é2", "34"),
            ("1234567", "abcdefg"),
        ):
            with self.assertRaises(ValueError):
                RefinementConfig(rows=rows)

    def _target(  # noqa: PLR0913
        self,
        sequence: int,
        output: str,
        x: float,
        y: float,
        *,
        horizontal: float | None = None,
        vertical: float | None = None,
        radial: float | None = None,
    ) -> StoredTarget:
        return make_stored_target(
            sequence,
            "camera",
            "schema",
            (0.1,),
            (0.2,),
            self.topology,
            PointerTarget(output, x, y),
            "center",
            unseen_error=radial,
            horizontal_residual=horizontal,
            vertical_residual=vertical,
        )


class RefinementControllerTests(unittest.IsolatedAsyncioTestCase):
    """Verify command sequencing, pointer-only motion, and transient surfaces."""

    async def test_unsupported_capture_is_nonfatal_and_reports_socket_fallback(self) -> None:
        """Missing portal support cannot stop an active manual refinement."""
        topology = DisplayTopology((DisplayRegion("only", 0, 0, 900, 600),))
        status = FakeStatus()
        controller = RefinementController(
            topology,
            ConfidenceRegionEstimator(
                topology,
                (),
                camera_id="camera",
                feature_schema="schema",
            ),
            FakePointer(topology.regions),
            status,
            lambda _regions: FakeTraining(),
            config=RefinementConfig(),
        )
        controller.update_rough(Point(450.0, 300.0))
        assert await controller.handle(ControlCommand("refine"))
        assert await controller.handle(ControlCommand("accept"))
        assert not await controller.handle(ControlCommand("capture"))
        assert controller.held
        assert "socket motion reports" in status.reports[-1][1]
        await controller.close()

    async def test_capture_motion_and_disconnect_keep_socket_fallback_active(self) -> None:
        """Captured pointer motion re-emits safely and disconnect remains nonfatal."""
        topology = DisplayTopology((DisplayRegion("only", 0, 0, 900, 600),))
        pointer = FakePointer(topology.regions)
        status = FakeStatus()
        motions: list[Callable[[int, float, float], None]] = []
        disconnections: list[Callable[[str], None]] = []

        class Capture:
            barrier_count = 4
            closed = False

            async def close(self) -> None:
                self.closed = True

        capture = Capture()

        async def authorize(
            motion: Callable[[int, float, float], None],
            disconnected: Callable[[str], None],
        ) -> Capture:
            motions.append(motion)
            disconnections.append(disconnected)
            return capture

        controller = RefinementController(
            topology,
            ConfidenceRegionEstimator(
                topology,
                (),
                camera_id="camera",
                feature_schema="schema",
            ),
            pointer,
            status,
            lambda _regions: FakeTraining(),
            config=RefinementConfig(),
            capture_authorizer=authorize,
        )
        controller.update_rough(Point(450.0, 300.0))
        assert await controller.handle(ControlCommand("refine"))
        assert await controller.handle(ControlCommand("accept"))
        assert await controller.handle(ControlCommand("capture"))
        motions[0](0, 10.0, -5.0)
        assert pointer.moves[-1] == ("only", 460.0, 295.0)
        disconnections[0]("fixture EIS disconnected")
        await asyncio.sleep(0)
        assert capture.closed
        assert "socket motion reports" in status.reports[-1][1]
        assert controller.held
        await controller.close()

    async def test_grid_accepts_manual_motion_settles_and_cancels(self) -> None:
        """The owner socket drives a complete bounded rough-in transaction."""
        topology = DisplayTopology((DisplayRegion("only", 0, 0, 900, 600),))
        pointer = FakePointer(topology.regions)
        status = FakeStatus()
        surface = FakeTraining()
        hud = FakeHud()
        waiters: list[asyncio.Future[None]] = []

        async def controlled_sleep(_delay: float) -> None:
            future = asyncio.get_running_loop().create_future()
            waiters.append(future)
            await future

        controller = RefinementController(
            topology,
            ConfidenceRegionEstimator(
                topology,
                (),
                camera_id="camera",
                feature_schema="schema",
                config=RefinementConfig(maximum_depth=2),
            ),
            pointer,
            status,
            lambda _regions: surface,
            config=RefinementConfig(maximum_depth=2),
            sleep=controlled_sleep,
            hud=hud,
        )
        controller.update_rough(Point(450.0, 300.0))
        assert await controller.handle(ControlCommand("refine"))
        assert controller.held
        assert surface.refinements[-1][:4] == (0.0, 0.0, 900.0, 600.0)
        assert surface.refinements[-1][6] == ("123", "456", "789")
        assert await controller.handle(ControlCommand("cell", label="1"))
        assert surface.refinements[-1][4] == 1
        assert await controller.handle(ControlCommand("accept"))
        assert controller.manual
        assert surface.refinement_hides == 0
        assert await controller.handle(ControlCommand("move", (10.0, 20.0)))
        assert surface.refinements[-1][:4] == (10.0, 20.0, 300.0, 200.0)
        await asyncio.sleep(0)
        waiters[-1].set_result(None)
        await asyncio.sleep(0)
        assert status.reports[-1][0] is RuntimeStatus.REFINEMENT_SETTLED
        assert hud.refinement_context.startswith("settled ")
        assert surface.refinement_hides == 1
        assert await controller.handle(ControlCommand("cancel"))
        assert not controller.held
        await controller.close()
        assert surface.closed


class SettledPositionTests(unittest.IsolatedAsyncioTestCase):
    """Verify transient authorized motion and restartable settling."""

    async def test_latest_position_commits_only_after_full_debounce(self) -> None:
        """Only the latest position survives a restarted debounce interval."""
        topology = DisplayTopology((DisplayRegion("only", 0, 0, 100, 100),))
        waiters: list[asyncio.Future[None]] = []
        settled: list[Point] = []

        async def controlled_sleep(_delay: float) -> None:
            future = asyncio.get_running_loop().create_future()
            waiters.append(future)
            await future

        recorder = SettledPositionRecorder(
            topology,
            Point(50.0, 50.0),
            settled.append,
            sleep=controlled_sleep,
        )
        recorder.report_relative(10.0, 5.0)
        await asyncio.sleep(0)
        recorder.report_absolute(Point(500.0, 500.0))
        await asyncio.sleep(0)
        assert len(waiters) == 2
        waiters[-1].set_result(None)
        await asyncio.sleep(0)
        assert settled == [Point(99.99999999999999, 99.99999999999999)]
        assert recorder.settled_position == settled[0]
        await recorder.close()

    async def test_close_cancels_unsettled_motion_without_history(self) -> None:
        """Closing drops an active timer and retains no movement history."""
        topology = DisplayTopology((DisplayRegion("only", 0, 0, 100, 100),))
        settled: list[Point] = []
        recorder = SettledPositionRecorder(topology, Point(0.0, 0.0), settled.append, delay=60.0)
        recorder.report_relative(1.0, 2.0)
        await recorder.close()
        assert settled == []
        assert recorder.position == Point(1.0, 2.0)


if __name__ == "__main__":
    unittest.main()
