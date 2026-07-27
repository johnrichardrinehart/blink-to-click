"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import TYPE_CHECKING

from gazeebo import __version__
from gazeebo.camera import CameraError, OpenCVCamera
from gazeebo.contexts import build_router
from gazeebo.contracts import RuntimeStatus
from gazeebo.control import ControlError, TrainingControl, request_command, request_training
from gazeebo.diagnostics import (
    CapturingVisionEstimator,
    DiagnosticArchive,
    DiagnosticArchiveError,
    diagnostic_archive_stats,
    diagnostic_capture_enabled,
    reset_diagnostic_archive,
)
from gazeebo.display import DisplayMonitorError, NativeDisplayMonitor
from gazeebo.geometry import DisplayTopology
from gazeebo.hud import LayerShellDebugHud
from gazeebo.input_capture import PortalInputCapture
from gazeebo.portal import PortalError, PortalPointerController
from gazeebo.refinement import RefinementConfig, refinement_rows
from gazeebo.runtime import (
    DISPLAY_REAUTHORIZATION_RESULT,
    FEATURE_SCHEMA,
    ConsoleStatus,
    TrackingConfig,
    TrackingError,
    install_signal_handlers,
    run_owned_session,
)
from gazeebo.state import TrainingState, TrainingStore, TrainingStoreError
from gazeebo.training import LayerShellTraining, TrainingConfig, TrainingError
from gazeebo.vision import OpenSeeFaceEstimator, VisionError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gazeebo.contracts import VisionEstimator


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="gazeebo",
        description="Local gaze-driven cursor navigation",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "run",
            "train",
            "reset-training",
            "dump-training",
            "training-stats",
            "reset-diagnostics",
            "diagnostic-stats",
            "refine-start",
            "refine-cell",
            "refine-accept",
            "refine-cancel",
            "refine-capture",
            "refine-move",
            "refine-position",
        ),
        default="run",
    )
    parser.add_argument("control_values", nargs="*")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--camera", help="V4L2 device path or numeric index")
    parser.add_argument(
        "--camera-codec",
        choices=("YUYV", "MJPG"),
        help="request an uncompressed YUYV or camera-compressed MJPEG transport",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--vision-confidence", type=float, default=0.20)
    parser.add_argument("--head-diagnostic-minimum", type=float, default=3.0)
    parser.add_argument("--head-recovery-timeout", type=float, default=10.0)
    parser.add_argument("--head-failure-panel-seconds", type=float, default=2.0)
    parser.add_argument("--calibration-settle", type=float, default=1.00)
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--startup-context-samples", type=int, default=8)
    parser.add_argument("--smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--smoothing-dead-zone", type=float, default=6.0)
    parser.add_argument("--smoothing-maximum-step", type=float, default=10000.0)
    parser.add_argument(
        "--pointer-update-interval",
        type=float,
        default=0.10,
        help="minimum seconds between pointer moves; zero updates continuously",
    )
    parser.add_argument("--training-batch-size", type=int, default=5)
    parser.add_argument("--training-precision-threshold", type=float, default=100.0)
    parser.add_argument("--training-maximum-targets", type=int, default=55)
    parser.add_argument("--training-preparation", type=float, default=2.0)
    parser.add_argument("--training-transition-overlap", type=float, default=1.0)
    parser.add_argument("--training-measurement", type=float, default=2.0)
    parser.add_argument("--training-target-size-mm", type=float, default=12.0)
    parser.add_argument("--training-fallback-diameter", type=float, default=72.0)
    parser.add_argument("--training-countdown-interval", type=float, default=1.0)
    parser.add_argument("--training-completion-seconds", type=float, default=2.0)
    parser.add_argument("--context-refresh-interval", type=float, default=1.0)
    parser.add_argument("--rough-in-width", type=float)
    parser.add_argument("--rough-in-height", type=float)
    parser.add_argument("--rough-in-minimum-samples", type=int, default=100)
    parser.add_argument("--refinement-maximum-depth", type=int, default=6)
    parser.add_argument("--refinement-minimum-cell-size", type=float, default=12.0)
    parser.add_argument("--refinement-settle-seconds", type=float, default=0.5)
    parser.add_argument(
        "--refinement-row",
        action="append",
        help="replace the refinement matrix with complete repeated row strings",
    )
    parser.add_argument("--noise-minimum-alpha", type=float, default=1.0)
    parser.add_argument("--noise-maximum-alpha", type=float, default=1.0)
    parser.add_argument("--noise-minimum-dead-zone", type=float, default=2.0)
    parser.add_argument("--noise-maximum-dead-zone", type=float, default=40.0)
    parser.add_argument("--noise-minimum-samples", type=int, default=20)
    parser.add_argument("--noise-maximum-targets", type=int, default=32)
    parser.add_argument(
        "--allow-display-reauthorization-pause",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pause and reacquire portal authorization after display additions",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="do not read or write local target-level training data",
    )
    parser.add_argument(
        "--diagnostic-capture",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "retain lossless frames from 3 seconds before through 3 seconds after "
            "head warnings (enabled by default; sensitive; 2 GiB quota)"
        ),
    )
    parser.add_argument(
        "--debug-hud",
        action="store_true",
        help="show authorized regions, routing, training surprise, and cursor coordinates",
    )
    return parser


def _camera_device(value: str | None) -> str | int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _tracking_config(arguments: argparse.Namespace) -> TrackingConfig:
    return TrackingConfig(
        calibration_settle_seconds=arguments.calibration_settle,
        calibration_samples_per_target=arguments.calibration_samples,
        startup_context_samples=arguments.startup_context_samples,
        smoothing_alpha=arguments.smoothing_alpha,
        smoothing_dead_zone=arguments.smoothing_dead_zone,
        smoothing_maximum_step=arguments.smoothing_maximum_step,
        pointer_update_interval_seconds=arguments.pointer_update_interval,
        head_diagnostic_minimum_seconds=arguments.head_diagnostic_minimum,
        head_recovery_timeout_seconds=arguments.head_recovery_timeout,
        head_failure_panel_seconds=arguments.head_failure_panel_seconds,
        context_refresh_interval_seconds=arguments.context_refresh_interval,
        allow_display_reauthorization_pause=arguments.allow_display_reauthorization_pause,
        noise_minimum_alpha=arguments.noise_minimum_alpha,
        noise_maximum_alpha=arguments.noise_maximum_alpha,
        noise_minimum_dead_zone=arguments.noise_minimum_dead_zone,
        noise_maximum_dead_zone=arguments.noise_maximum_dead_zone,
        noise_minimum_samples=arguments.noise_minimum_samples,
        noise_maximum_targets=arguments.noise_maximum_targets,
    )


def _training_config(arguments: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        batch_size=arguments.training_batch_size,
        precision_threshold=arguments.training_precision_threshold,
        maximum_targets=arguments.training_maximum_targets,
        preparation_seconds=arguments.training_preparation,
        transition_overlap_seconds=arguments.training_transition_overlap,
        measurement_seconds=arguments.training_measurement,
        physical_target_diameter_mm=arguments.training_target_size_mm,
        fallback_target_diameter=arguments.training_fallback_diameter,
        countdown_interval_seconds=arguments.training_countdown_interval,
        completion_seconds=arguments.training_completion_seconds,
    )


def _refinement_config(arguments: argparse.Namespace) -> RefinementConfig:
    return RefinementConfig(
        width_override=arguments.rough_in_width,
        height_override=arguments.rough_in_height,
        minimum_samples=arguments.rough_in_minimum_samples,
        maximum_depth=arguments.refinement_maximum_depth,
        minimum_cell_size=arguments.refinement_minimum_cell_size,
        settle_seconds=arguments.refinement_settle_seconds,
        rows=refinement_rows(arguments.refinement_row),
    )


def _open_vision(arguments: argparse.Namespace) -> tuple[OpenCVCamera, OpenSeeFaceEstimator]:
    camera = OpenCVCamera(
        _camera_device(arguments.camera),
        width=arguments.width,
        height=arguments.height,
        frames_per_second=arguments.fps,
        codec=arguments.camera_codec,
    )
    try:
        vision = OpenSeeFaceEstimator(
            *camera.dimensions,
            minimum_confidence=arguments.vision_confidence,
        )
    except Exception:
        camera.close()
        raise
    return camera, vision


async def _open_startup_resources(  # noqa: C901, PLR0912
    arguments: argparse.Namespace,
    stop: asyncio.Event,
) -> tuple[PortalPointerController, OpenCVCamera, OpenSeeFaceEstimator] | None:
    """Authorize geometry while camera and vision initialize, unless stopped."""
    portal_task = asyncio.create_task(PortalPointerController.authorize())
    vision_task = asyncio.create_task(asyncio.to_thread(_open_vision, arguments))
    stop_task = asyncio.create_task(stop.wait())
    pending: set[asyncio.Task[object]] = {portal_task, vision_task, stop_task}
    while not portal_task.done() or not vision_task.done():
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done:
            if not portal_task.done():
                portal_task.cancel()
            break
    if not vision_task.done():
        await asyncio.wait((vision_task,))
    if not portal_task.done():
        portal_task.cancel()
    portal_result, vision_result = await asyncio.gather(
        portal_task,
        vision_task,
        return_exceptions=True,
    )
    stop_task.cancel()
    await asyncio.gather(stop_task, return_exceptions=True)
    pointer = portal_result if isinstance(portal_result, PortalPointerController) else None
    resources = vision_result if isinstance(vision_result, tuple) else None
    if stop.is_set():
        if resources is not None:
            resources[1].close()
            resources[0].close()
        if pointer is not None:
            await pointer.close()
        return None
    if isinstance(portal_result, BaseException) or isinstance(vision_result, BaseException):
        if resources is not None:
            resources[1].close()
            resources[0].close()
        if pointer is not None:
            await pointer.close()
        error = portal_result if isinstance(portal_result, BaseException) else vision_result
        if not isinstance(error, BaseException):
            msg = "parallel startup failed without an exception"
            raise TypeError(msg)
        raise error
    if pointer is None or resources is None:
        msg = "parallel startup returned incomplete resources"
        raise RuntimeError(msg)
    return pointer, resources[0], resources[1]


async def _load_startup_inputs(
    arguments: argparse.Namespace,
    store: TrainingStore,
    stop: asyncio.Event,
) -> (
    tuple[
        TrainingState,
        PortalPointerController,
        OpenCVCamera,
        OpenSeeFaceEstimator,
    ]
    | None
):
    """Load private state while portal and vision startup proceed independently."""
    state_task = asyncio.create_task(asyncio.to_thread(store.load))
    resources_task = asyncio.create_task(_open_startup_resources(arguments, stop))
    await asyncio.wait((state_task, resources_task), return_when=asyncio.FIRST_EXCEPTION)
    if state_task.done() and not state_task.cancelled() and state_task.exception() is not None:
        stop.set()
    state_result, resources_result = await asyncio.gather(
        state_task,
        resources_task,
        return_exceptions=True,
    )
    if isinstance(state_result, BaseException):
        raise state_result
    if isinstance(resources_result, BaseException):
        raise resources_result
    if resources_result is None:
        return None
    return state_result, *resources_result


def _needs_training(
    state: TrainingState,
    pointer: PortalPointerController,
    camera: OpenCVCamera,
) -> bool:
    try:
        build_router(
            state,
            DisplayTopology(pointer.regions),
            camera_id=camera.camera_id,
            feature_schema=FEATURE_SCHEMA,
        )
    except ValueError:
        return True
    return False


async def _run_session(  # noqa: C901
    arguments: argparse.Namespace,
    status: ConsoleStatus,
    store: TrainingStore,
    stop: asyncio.Event,
    *,
    train_requested: bool,
) -> int:
    """Own one portal authorization epoch inside the foreground process."""
    tracking = _tracking_config(arguments)
    training_config = _training_config(arguments)
    refinement_config = _refinement_config(arguments)
    training_requested = asyncio.Event()
    control = TrainingControl(training_requested)
    pointer: PortalPointerController | None = None
    hud: LayerShellDebugHud | None = None
    training: LayerShellTraining | None = None
    monitor: NativeDisplayMonitor | None = None
    camera: OpenCVCamera | None = None
    vision: VisionEstimator | None = None
    session_started = False
    try:
        await control.start()
        status.report(RuntimeStatus.AUTHORIZING)
        startup = await _load_startup_inputs(arguments, store, stop)
        if startup is None:
            return 0
        state, pointer, camera, vision = startup
        if arguments.diagnostic_capture:
            vision = CapturingVisionEstimator(
                vision,
                DiagnosticArchive(status=status),
            )
        monitor = NativeDisplayMonitor.create()
        if arguments.debug_hud:
            hud = LayerShellDebugHud.create(pointer.regions)
        if train_requested or _needs_training(state, pointer, camera):
            training = LayerShellTraining.create(pointer.regions)
        session_started = True
        return await run_owned_session(
            camera,
            vision,
            pointer,
            status,
            stop,
            hud=hud,
            training=training,
            tracking=tracking,
            training_config=training_config,
            training_store=store,
            training_state=state,
            train_requested=train_requested,
            training_requested_event=training_requested,
            training_factory=LayerShellTraining.create,
            diagnostic_factory=LayerShellTraining.create,
            training_control=control,
            display_monitor=monitor,
            refinement_config=refinement_config,
            refinement_factory=LayerShellTraining.create,
            input_capture_authorizer=PortalInputCapture.authorize,
        )
    finally:
        if not session_started:
            await control.close()
            if monitor is not None:
                monitor.close()
            if vision is not None:
                vision.close()
            if camera is not None:
                camera.close()
            if training is not None:
                await training.close()
            if hud is not None:
                await hud.close()
            if pointer is not None:
                await pointer.close()


def _refinement_command(command: str, values: Sequence[str]) -> str | None:
    """Translate explicit CLI refinement commands into socket protocol lines."""
    fixed = {
        "refine-start": ("refine", 0),
        "refine-cell": ("cell", 1),
        "refine-accept": ("accept", 0),
        "refine-cancel": ("cancel", 0),
        "refine-capture": ("capture", 0),
        "refine-move": ("move", 2),
        "refine-position": ("position", 2),
    }
    specification = fixed.get(command)
    if specification is None:
        if values:
            msg = "control values require a refinement command"
            raise ValueError(msg)
        return None
    name, count = specification
    if len(values) != count:
        msg = f"{command} requires {count} value(s)"
        raise ValueError(msg)
    return " ".join((name, *values))


async def _run(arguments: argparse.Namespace) -> int:  # noqa: C901, PLR0911, PLR0912, PLR0915
    """Run one foreground process across any required authorization epochs."""
    status = ConsoleStatus()
    status.report(RuntimeStatus.STARTING)
    store = TrainingStore(ephemeral=arguments.ephemeral)
    stop = asyncio.Event()
    install_signal_handlers(stop)
    try:
        if arguments.command == "dump-training":
            sys.stdout.write(store.dump_json())
            status.report(RuntimeStatus.STOPPED)
            return 0
        if arguments.command == "training-stats":
            statistics = store.stats()
            sys.stdout.write(
                json.dumps(
                    {
                        "bytes_per_target": statistics.bytes_per_target,
                        "compression_ratio": statistics.compression_ratio,
                        "logical_bytes": statistics.logical_bytes,
                        "on_disk_bytes": statistics.on_disk_bytes,
                        "schema_version": statistics.schema_version,
                        "target_count": statistics.target_count,
                    },
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )
            status.report(RuntimeStatus.STOPPED)
            return 0
        if arguments.command == "reset-training":
            store.reset()
            status.report(RuntimeStatus.STOPPED)
            return 0
        if arguments.command == "diagnostic-stats":
            diagnostic_statistics = diagnostic_archive_stats()
            sys.stdout.write(
                json.dumps(
                    {
                        "event_count": diagnostic_statistics.event_count,
                        "maximum_bytes": diagnostic_statistics.maximum_bytes,
                        "on_disk_bytes": diagnostic_statistics.on_disk_bytes,
                        "schema_version": diagnostic_statistics.schema_version,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            status.report(RuntimeStatus.STOPPED)
            return 0
        if arguments.command == "reset-diagnostics":
            reset_diagnostic_archive()
            status.report(RuntimeStatus.STOPPED)
            return 0
        if arguments.command == "train" and await request_training():
            status.report(RuntimeStatus.STOPPED, "active session accepted training request")
            return 0
        refinement_command = _refinement_command(arguments.command, arguments.control_values)
        if refinement_command is not None:
            if await request_command(refinement_command):
                status.report(RuntimeStatus.STOPPED, "active session accepted refinement request")
                return 0
            status.report(RuntimeStatus.INPUT_ERROR, "no active Gazeebo owner")
            return 2
        status.report(RuntimeStatus.LOADING)
        train_requested = arguments.command == "train"
        while not stop.is_set():
            result = await _run_session(
                arguments,
                status,
                store,
                stop,
                train_requested=train_requested,
            )
            if result != DISPLAY_REAUTHORIZATION_RESULT:
                return result
            train_requested = False
            status.report(
                RuntimeStatus.DISPLAY_CHANGE,
                "portal authorization refresh starting",
            )
        return 0  # noqa: TRY300
    except ControlError as error:
        status.report(RuntimeStatus.INPUT_ERROR, str(error))
        return 2
    except (PortalError, DisplayMonitorError) as error:
        status.report(RuntimeStatus.INPUT_ERROR, str(error))
        return 2
    except TrainingStoreError as error:
        status.report(RuntimeStatus.STORE_ERROR, str(error))
        return 1
    except DiagnosticArchiveError as error:
        status.report(RuntimeStatus.DIAGNOSTIC_CAPTURE, str(error))
        return 1
    except TrainingError as error:
        status.report(RuntimeStatus.TRAINING_ERROR, str(error))
        return 1
    except (CameraError, VisionError, TrackingError) as error:
        if stop.is_set():
            return 0
        status.report(RuntimeStatus.CAMERA_ERROR, str(error))
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run one foreground navigation or training command."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command in {"run", "train"}:
        try:
            arguments.diagnostic_capture = diagnostic_capture_enabled(
                cli_value=arguments.diagnostic_capture
            )
        except DiagnosticArchiveError as error:
            ConsoleStatus().report(RuntimeStatus.DIAGNOSTIC_CAPTURE, str(error))
            return 1
    try:
        return asyncio.run(_run(arguments))
    except ValueError as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        return 130
