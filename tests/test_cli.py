"""Tests for the command-line boundary."""

from __future__ import annotations

import asyncio
import io
import json
import threading
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from gazeebo.cli import (
    _camera_device,
    _load_startup_inputs,
    _open_startup_resources,
    _refinement_command,
    _refinement_config,
    build_parser,
    main,
)
from gazeebo.state import TrainingState, TrainingStore, TrainingStoreError
from gazeebo.training import TrainingConfig


class CliTests(unittest.TestCase):
    """Lock argument parsing without opening runtime resources."""

    def test_defaults_define_one_local_foreground_session(self) -> None:
        """An empty argument list selects safe runtime defaults."""
        arguments = build_parser().parse_args([])
        assert arguments.command == "run"
        assert arguments.camera is None
        assert arguments.camera_codec is None
        assert arguments.width == 640
        assert arguments.height == 480
        assert arguments.calibration_samples > 0
        assert arguments.head_diagnostic_minimum == 3.0
        assert arguments.head_recovery_timeout == 10.0
        assert arguments.pointer_update_interval == 0.10
        assert arguments.smoothing_alpha == 1.0
        assert arguments.smoothing_maximum_step == 10000.0
        assert arguments.noise_minimum_alpha == 1.0
        assert arguments.noise_maximum_alpha == 1.0
        assert not arguments.ephemeral
        assert arguments.training_batch_size == 5
        assert arguments.training_precision_threshold == 100.0
        assert arguments.training_maximum_targets == 55
        assert arguments.training_preparation == 2.0
        assert arguments.training_transition_overlap == 1.0
        assert arguments.training_measurement == 2.0
        assert arguments.training_target_size_mm == 12.0
        assert arguments.training_fallback_diameter == 72.0
        assert arguments.context_refresh_interval == 1.0
        assert arguments.allow_display_reauthorization_pause
        assert arguments.diagnostic_capture is None
        assert not arguments.debug_hud

    def test_dump_and_stats_commands_need_no_camera_or_portal(self) -> None:
        """Read-only introspection returns stable JSON before runtime startup."""
        dump_output = io.StringIO()
        stats_output = io.StringIO()
        with patch("gazeebo.cli._open_startup_resources") as startup:
            with redirect_stdout(dump_output):
                assert main(["dump-training", "--ephemeral"]) == 0
            with redirect_stdout(stats_output):
                assert main(["training-stats", "--ephemeral"]) == 0
        startup.assert_not_called()
        assert json.loads(dump_output.getvalue())["version"] >= 5
        statistics = json.loads(stats_output.getvalue())
        assert statistics["schema_version"] >= 5
        assert statistics["target_count"] == 0

    def test_refinement_commands_map_to_bounded_socket_protocol(self) -> None:
        """CLI cell and motion values become explicit owner requests."""
        arguments = build_parser().parse_args(["refine-cell", ";"])
        assert _refinement_command(arguments.command, arguments.control_values) == "cell ;"
        arguments = build_parser().parse_args(["refine-move", "-2.5", "3"])
        assert _refinement_command(arguments.command, arguments.control_values) == "move -2.5 3"
        with self.assertRaises(ValueError):
            _refinement_command("refine-cell", ())
        configured = build_parser().parse_args(
            [
                "--refinement-row",
                "i,.p",
                "--refinement-row",
                "aoeu",
                "--refinement-row",
                ";qjk",
            ]
        )
        assert _refinement_config(configured).rows == ("i,.p", "aoeu", ";qjk")

    def test_training_commands_do_not_expose_profiles(self) -> None:
        """Users request training or reset one automatic local corpus."""
        train = build_parser().parse_args(["train"])
        reset = build_parser().parse_args(["reset-training"])
        assert train.command == "train"
        assert reset.command == "reset-training"
        assert "profile" not in vars(train)

    def test_camera_index_and_path_remain_distinct(self) -> None:
        """Numeric indices and device paths reach OpenCV in their native forms."""
        assert _camera_device(None) is None
        assert _camera_device("2") == 2
        assert _camera_device("/dev/video2") == "/dev/video2"
        assert build_parser().parse_args(["--camera-codec", "MJPG"]).camera_codec == "MJPG"

    def test_default_face_confidence_allows_valid_profile_geometry(self) -> None:
        """The runtime uses the tracker's profile-capable landmark threshold."""
        arguments = build_parser().parse_args([])
        assert arguments.vision_confidence == 0.20

    def test_diagnostic_capture_cli_is_explicitly_disableable(self) -> None:
        """CLI policy overrides the default-enabled configuration path."""
        enabled = build_parser().parse_args(["--diagnostic-capture"])
        disabled = build_parser().parse_args(["--no-diagnostic-capture"])
        reset = build_parser().parse_args(["reset-diagnostics"])
        statistics = build_parser().parse_args(["diagnostic-stats"])
        assert enabled.diagnostic_capture
        assert not disabled.diagnostic_capture
        assert reset.command == "reset-diagnostics"
        assert statistics.command == "diagnostic-stats"

    def test_zero_pointer_interval_requests_continuous_updates(self) -> None:
        """A zero interval remains available for explicit development tuning."""
        arguments = build_parser().parse_args(["--pointer-update-interval", "0"])
        assert arguments.pointer_update_interval == 0.0

    def test_store_load_overlaps_portal_and_vision_startup(self) -> None:
        """A slow local store cannot serialize independent authorization work."""
        loading = threading.Event()
        release = threading.Event()

        class BlockingStore(TrainingStore):
            def load(self) -> TrainingState:
                loading.set()
                assert release.wait(timeout=1.0)
                return TrainingState()

        async def open_resources(
            _arguments: object,
            _stop: asyncio.Event,
        ) -> None:
            await asyncio.to_thread(loading.wait)
            release.set()

        with patch("gazeebo.cli._open_startup_resources", side_effect=open_resources):
            result = asyncio.run(
                _load_startup_inputs(
                    build_parser().parse_args([]),
                    BlockingStore(ephemeral=True),
                    asyncio.Event(),
                )
            )
        assert result is None

    def test_store_failure_cancels_parallel_startup(self) -> None:
        """Unsafe state stops portal and camera startup instead of waiting for input."""
        stopped = False

        class BrokenStore(TrainingStore):
            def load(self) -> TrainingState:
                msg = "fixture store failure"
                raise TrainingStoreError(msg)

        async def open_resources(
            _arguments: object,
            stop: asyncio.Event,
        ) -> None:
            nonlocal stopped
            await stop.wait()
            stopped = True

        with (
            patch("gazeebo.cli._open_startup_resources", side_effect=open_resources),
            self.assertRaisesRegex(TrainingStoreError, "fixture store failure"),
        ):
            asyncio.run(
                _load_startup_inputs(
                    build_parser().parse_args([]),
                    BrokenStore(ephemeral=True),
                    asyncio.Event(),
                )
            )
        assert stopped

    def test_startup_stop_closes_parallel_camera_resources(self) -> None:
        """A signal during portal authorization cannot leak camera or vision state."""

        class Resource:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        camera = Resource()
        vision = Resource()
        portal_started = asyncio.Event()

        async def authorize() -> None:
            portal_started.set()
            await asyncio.Event().wait()

        async def scenario() -> object:
            stop = asyncio.Event()
            task = asyncio.create_task(_open_startup_resources(build_parser().parse_args([]), stop))
            await portal_started.wait()
            stop.set()
            return await task

        with (
            patch("gazeebo.cli.PortalPointerController.authorize", side_effect=authorize),
            patch("gazeebo.cli._open_vision", return_value=(camera, vision)),
        ):
            assert asyncio.run(scenario()) is None
        assert camera.closed
        assert vision.closed

    def test_non_terminating_training_values_are_rejected(self) -> None:
        """Invalid training timing fails before runtime resources can open."""
        arguments = build_parser().parse_args(["--training-measurement", "0"])
        with self.assertRaisesRegex(ValueError, "windows"):
            TrainingConfig(
                batch_size=arguments.training_batch_size,
                precision_threshold=arguments.training_precision_threshold,
                maximum_targets=arguments.training_maximum_targets,
                preparation_seconds=arguments.training_preparation,
                transition_overlap_seconds=arguments.training_transition_overlap,
                measurement_seconds=arguments.training_measurement,
                physical_target_diameter_mm=arguments.training_target_size_mm,
                fallback_target_diameter=arguments.training_fallback_diameter,
            )


if __name__ == "__main__":
    unittest.main()
