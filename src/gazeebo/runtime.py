"""Foreground model selection, training, navigation, and cleanup lifecycle."""

from __future__ import annotations

import asyncio
import copy
import math
import signal
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gazeebo.adaptation import TopologyQuality, describe_topology, make_stored_target
from gazeebo.calibration import (
    MINIMUM_CALIBRATION_SAMPLES,
    CalibrationModel,
    CalibrationSample,
    IncrementalCalibration,
    aggregate_feature_dispersion,
    aggregate_features,
    target_fit_weight,
)
from gazeebo.contexts import (
    ModelRouter,
    SmoothingBounds,
    SmoothingSettings,
    ValidationMetrics,
    add_target,
    build_router,
    calibration_samples_for,
    candidate_is_acceptable,
    noise_smoothing_for,
)
from gazeebo.contracts import RuntimeStatus
from gazeebo.geometry import (
    DisplayTopology,
    Point,
    PointerSmoother,
    PointerTarget,
    calibration_targets,
    rolling_point_median,
)
from gazeebo.recovery import HeadTrackingError, observe_with_head_recovery
from gazeebo.state import (
    MAXIMUM_MODEL_ANCHORS,
    ModelAnchor,
    TrainingState,
    TrainingStore,
    ValidationSummary,
)
from gazeebo.surprise import RegionSurpriseScheduler
from gazeebo.training import (
    CollectedTarget,
    GazePredictor,
    TrainingConfig,
    TrainingError,
    TrainingMetrics,
    cursor_noise_summary,
    run_adaptive_training,
    show_target_preparation,
    show_training_completion,
    show_training_countdown,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from gazeebo.contracts import (
        CameraCapture,
        ContextVector,
        DebugHud,
        DisplayRegion,
        FeatureVector,
        HeadDiagnosticSurface,
        PointerController,
        StatusSink,
        TrainingSurface,
        VisionEstimator,
    )
    from gazeebo.control import TrainingControl
    from gazeebo.display import DisplayMonitor, OutputGeometry

FEATURE_SCHEMA = "gaze-v4"
TRAINING_REQUESTED_RESULT = 4
DISPLAY_REAUTHORIZATION_RESULT = 5
CALIBRATION_EDGE_BOUNDARY = 0.25
ANCHOR_VARIANCE_FLOOR = 0.0025
SMOOTHING_ALPHA_REPORT_THRESHOLD = 0.01
SMOOTHING_DEAD_ZONE_REPORT_THRESHOLD = 0.5
BAYESIAN_UNCERTAINTY_SCALE = 300.0


class TrackingError(RuntimeError):
    """Model selection, training, or navigation cannot continue safely."""


def _diagnostic_factory(
    factory: Callable[[Sequence[DisplayRegion]], HeadDiagnosticSurface] | None,
) -> Callable[[Sequence[DisplayRegion]], HeadDiagnosticSurface]:
    if factory is not None:
        return factory

    def unavailable(_regions: Sequence[DisplayRegion]) -> HeadDiagnosticSurface:
        msg = "head-tracking diagnostic surface is unavailable"
        raise TrackingError(msg)

    return unavailable


@dataclass(frozen=True, slots=True)
class _BaseCalibration:
    model: CalibrationModel
    samples: tuple[CalibrationSample, ...]
    targets: tuple[CollectedTarget, ...]


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Session timing and failure thresholds."""

    calibration_settle_seconds: float = 1.00
    calibration_samples_per_target: int = 8
    calibration_attempts_per_target: int = 2000
    startup_context_samples: int = 8
    startup_context_attempts: int = 80
    frame_interval_seconds: float = 0.01
    head_diagnostic_minimum_seconds: float = 3.0
    head_recovery_timeout_seconds: float = 10.0
    head_failure_panel_seconds: float = 2.0
    smoothing_alpha: float = 1.0
    smoothing_dead_zone: float = 6.0
    smoothing_maximum_step: float = 10000.0
    pointer_update_interval_seconds: float = 0.10
    context_refresh_interval_seconds: float = 1.0
    allow_display_reauthorization_pause: bool = True
    noise_minimum_alpha: float = 1.0
    noise_maximum_alpha: float = 1.0
    noise_minimum_dead_zone: float = 2.0
    noise_maximum_dead_zone: float = 40.0
    noise_minimum_samples: int = 20
    noise_maximum_targets: int = 32

    def __post_init__(self) -> None:
        """Reject unsafe or non-terminating runtime settings."""
        intervals = (
            self.calibration_settle_seconds,
            self.frame_interval_seconds,
            self.pointer_update_interval_seconds,
            self.context_refresh_interval_seconds,
            self.head_diagnostic_minimum_seconds,
            self.head_recovery_timeout_seconds,
            self.head_failure_panel_seconds,
        )
        if any(not math.isfinite(interval) or interval < 0.0 for interval in intervals):
            msg = "tracking intervals must be finite and non-negative"
            raise ValueError(msg)
        counts = (
            self.calibration_samples_per_target,
            self.calibration_attempts_per_target,
            self.startup_context_samples,
            self.startup_context_attempts,
        )
        if any(count <= 0 for count in counts):
            msg = "tracking sample and failure counts must be positive"
            raise ValueError(msg)
        if self.calibration_attempts_per_target < self.calibration_samples_per_target:
            msg = "calibration attempts must cover the requested samples"
            raise ValueError(msg)
        if self.startup_context_attempts < self.startup_context_samples:
            msg = "startup context attempts must cover the requested samples"
            raise ValueError(msg)
        if self.head_recovery_timeout_seconds <= 0.0:
            msg = "head recovery timeout must be positive"
            raise ValueError(msg)
        SmoothingBounds(
            self.noise_minimum_alpha,
            self.noise_maximum_alpha,
            self.noise_minimum_dead_zone,
            self.noise_maximum_dead_zone,
            self.noise_minimum_samples,
            self.noise_maximum_targets,
        )


class ConsoleStatus:
    """Report state transitions to standard error."""

    def report(self, status: RuntimeStatus, detail: str = "") -> None:
        """Write one line immediately without retaining it."""
        suffix = f": {detail}" if detail else ""
        sys.stderr.write(f"gazeebo: {status.value}{suffix}\n")
        sys.stderr.flush()


async def run_owned_session(  # noqa: C901, PLR0912, PLR0913, PLR0915
    camera: CameraCapture,
    vision: VisionEstimator,
    pointer: PointerController,
    status: StatusSink,
    stop: asyncio.Event,
    *,
    hud: DebugHud | None = None,
    training: TrainingSurface | None = None,
    tracking: TrackingConfig | None = None,
    training_config: TrainingConfig | None = None,
    training_store: TrainingStore | None = None,
    training_state: TrainingState | None = None,
    train_requested: bool = False,
    training_requested_event: asyncio.Event | None = None,
    training_factory: Callable[[Sequence[DisplayRegion]], TrainingSurface] | None = None,
    diagnostic_factory: Callable[[Sequence[DisplayRegion]], HeadDiagnosticSurface] | None = None,
    training_control: TrainingControl | None = None,
    display_monitor: DisplayMonitor | None = None,
    feature_schema: str = FEATURE_SCHEMA,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> int:
    """Select or train a model, navigate, and release all owned resources."""
    tracking_config = tracking or TrackingConfig()
    active_training_config = training_config or TrainingConfig()
    head_diagnostic_factory = diagnostic_factory or training_factory
    state = training_state or TrainingState()
    persist_completed_targets = training_store is not None and training is not None

    def retain_completed_target(target: CollectedTarget) -> None:
        nonlocal state
        if training_store is not None:
            state = _append_unvalidated_target(
                training_store,
                state,
                DisplayTopology(pointer.regions),
                camera.camera_id,
                feature_schema,
                target,
            )

    try:
        topology = DisplayTopology(pointer.regions)
        stored_router = _stored_router(
            state,
            topology,
            camera.camera_id,
            feature_schema,
            status,
        )
        incumbent: GazePredictor | None = (
            stored_router
            if any(
                anchor.camera_id == camera.camera_id and anchor.feature_schema == feature_schema
                for anchor in state.anchors
            )
            else None
        )
        model: GazePredictor
        initial_targets: tuple[CollectedTarget, ...] = ()

        if stored_router is None:
            if training is None:
                status.report(
                    RuntimeStatus.TRAINING_RECOMMENDED,
                    "no compatible stored model; initial training is required",
                )
            status.report(
                RuntimeStatus.INITIAL_TRAINING,
                f"{len(topology.regions)} authorized displays",
            )
            if training is not None and not await show_training_countdown(
                training,
                status,
                stop,
                active_training_config.countdown_interval_seconds,
                sleep,
            ):
                return 0
            calibration = await _calibrate(
                camera,
                vision,
                pointer,
                topology,
                training,
                hud,
                status,
                stop,
                tracking_config,
                active_training_config,
                _diagnostic_factory(head_diagnostic_factory),
                clock,
                sleep,
                retain_completed_target if persist_completed_targets else None,
            )
            if stop.is_set() or calibration is None:
                return 0
            model = calibration.model
            base_samples = list(calibration.samples)
            initial_targets = calibration.targets
        else:
            model = stored_router
            base_samples = calibration_samples_for(
                state,
                topology,
                camera_id=camera.camera_id,
                feature_schema=feature_schema,
            )
            await _prime_router(
                camera,
                vision,
                model,
                status,
                stop,
                topology.regions,
                _diagnostic_factory(head_diagnostic_factory),
                tracking_config,
                clock,
                sleep,
            )

        should_train = training is not None and (train_requested or incumbent is None)
        if should_train and training is not None:
            training_result = await run_adaptive_training(
                camera,
                vision,
                pointer,
                topology,
                training,
                status,
                stop,
                base_samples,
                model,
                active_training_config,
                incumbent_model=incumbent,
                force_adaptation=train_requested and incumbent is not None,
                establishing_model=incumbent is None,
                target_offset=len(initial_targets),
                diagnostic_factory=_diagnostic_factory(head_diagnostic_factory),
                head_diagnostic_minimum=tracking_config.head_diagnostic_minimum_seconds,
                head_recovery_timeout=tracking_config.head_recovery_timeout_seconds,
                failure_panel_seconds=tracking_config.head_failure_panel_seconds,
                pointer_interval=tracking_config.pointer_update_interval_seconds,
                frame_interval=tracking_config.frame_interval_seconds,
                hud=hud,
                show_countdown=incumbent is not None,
                clock=clock,
                sleep=sleep,
                completed_target_sink=(
                    retain_completed_target if persist_completed_targets else None
                ),
                surprise_scheduler=_region_surprise_scheduler(
                    state,
                    topology,
                    camera.camera_id,
                    feature_schema,
                    active_training_config,
                    seed_targets=(() if persist_completed_targets else initial_targets),
                ),
            )
            await training.close()
            if stop.is_set() or training_result is None:
                return 0
            model = (
                training_result.model
                if training_result.model_accepted or incumbent is None
                else incumbent
            )
            pending = (*initial_targets, *training_result.completed_targets)
            if training_store is not None and pending:
                persisted = _persist_targets(
                    training_store,
                    state,
                    topology,
                    camera.camera_id,
                    feature_schema,
                    pending,
                    training_result.aggregate_metrics.median_error,
                    training_result.aggregate_metrics.edge_error,
                    training_result.aggregate_metrics.maximum_region_error,
                    training_result.aggregate_metrics.maximum_region_cvar90,
                    training_result.aggregate_metrics.maximum_region_upper,
                    validation_targets=training_result.validation_targets,
                    incumbent_metrics=(
                        None if incumbent is None else training_result.incumbent_metrics
                    ),
                    model_accepted=training_result.model_accepted,
                    validated_model=(
                        training_result.model
                        if training_result.model_accepted
                        and isinstance(training_result.model, CalibrationModel)
                        else None
                    ),
                    append_targets=not persist_completed_targets,
                )
                if persisted is not None:
                    model, state = persisted
                    if training_result.model_accepted:
                        _report_persistent_metrics(status, state)
                else:
                    status.report(
                        RuntimeStatus.TRAINING_RECOMMENDED,
                        "persistent context model failed unseen acceptance",
                    )

        _set_hud_model_context(hud, model)
        while True:
            status.report(RuntimeStatus.ACTIVE)
            result = await _track(
                camera,
                vision,
                pointer,
                topology,
                model,
                hud,
                status,
                stop,
                tracking_config,
                training_requested_event,
                state,
                camera.camera_id,
                feature_schema,
                display_monitor,
                _diagnostic_factory(head_diagnostic_factory),
                clock,
                sleep,
            )
            if result != TRAINING_REQUESTED_RESULT:
                return result
            if training_requested_event is not None:
                training_requested_event.clear()
            if training_factory is None:
                msg = "active training request has no target-surface factory"
                raise TrackingError(msg)
            active_training = training_factory(pointer.regions)
            try:
                base_samples = calibration_samples_for(
                    state,
                    topology,
                    camera_id=camera.camera_id,
                    feature_schema=feature_schema,
                )
                training_result = await run_adaptive_training(
                    camera,
                    vision,
                    pointer,
                    topology,
                    active_training,
                    status,
                    stop,
                    base_samples,
                    model,
                    active_training_config,
                    incumbent_model=model,
                    force_adaptation=True,
                    diagnostic_factory=_diagnostic_factory(head_diagnostic_factory),
                    head_diagnostic_minimum=tracking_config.head_diagnostic_minimum_seconds,
                    head_recovery_timeout=tracking_config.head_recovery_timeout_seconds,
                    failure_panel_seconds=tracking_config.head_failure_panel_seconds,
                    pointer_interval=tracking_config.pointer_update_interval_seconds,
                    frame_interval=tracking_config.frame_interval_seconds,
                    hud=hud,
                    clock=clock,
                    sleep=sleep,
                    completed_target_sink=(
                        retain_completed_target if persist_completed_targets else None
                    ),
                    surprise_scheduler=_region_surprise_scheduler(
                        state,
                        topology,
                        camera.camera_id,
                        feature_schema,
                        active_training_config,
                    ),
                )
                if training_result is None:
                    return 0
                if training_store is not None and training_result.completed_targets:
                    persisted = _persist_targets(
                        training_store,
                        state,
                        topology,
                        camera.camera_id,
                        feature_schema,
                        training_result.completed_targets,
                        training_result.aggregate_metrics.median_error,
                        training_result.aggregate_metrics.edge_error,
                        training_result.aggregate_metrics.maximum_region_error,
                        training_result.aggregate_metrics.maximum_region_cvar90,
                        training_result.aggregate_metrics.maximum_region_upper,
                        validation_targets=training_result.validation_targets,
                        incumbent_metrics=training_result.incumbent_metrics,
                        model_accepted=training_result.model_accepted,
                        validated_model=(
                            training_result.model
                            if training_result.model_accepted
                            and isinstance(training_result.model, CalibrationModel)
                            else None
                        ),
                        append_targets=not persist_completed_targets,
                    )
                    if persisted is not None:
                        model, state = persisted
                        _report_persistent_metrics(status, state)
                    else:
                        model = training_result.model
                        status.report(
                            RuntimeStatus.TRAINING_RECOMMENDED,
                            "persistent context model failed unseen acceptance",
                        )
                else:
                    model = training_result.model if training_result.model_accepted else model
                _set_hud_model_context(hud, model)
            finally:
                await active_training.close()
    finally:
        vision.close()
        camera.close()
        if training is not None:
            await training.close()
        if hud is not None:
            await hud.close()
        await pointer.close()
        if training_control is not None:
            await training_control.close()
        if display_monitor is not None:
            display_monitor.close()
        status.report(RuntimeStatus.STOPPED)


def _stored_router(
    state: TrainingState,
    topology: DisplayTopology,
    camera_id: str,
    feature_schema: str,
    status: StatusSink,
) -> ModelRouter | None:
    if not state.targets:
        return None
    status.report(RuntimeStatus.SELECTING_MODEL)
    try:
        router = build_router(
            state,
            topology,
            camera_id=camera_id,
            feature_schema=feature_schema,
        )
    except ValueError:
        return None
    quality = router.decide(state.targets[-1].context).topology_quality
    if quality is TopologyQuality.WEAK:
        status.report(
            RuntimeStatus.TOPOLOGY_UNVALIDATED,
            "using best-effort remapping and authorized-union clipping",
        )
    return router


async def _prime_router(  # noqa: PLR0913
    camera: CameraCapture,
    vision: VisionEstimator,
    router: ModelRouter,
    status: StatusSink,
    stop: asyncio.Event,
    regions: Sequence[DisplayRegion],
    diagnostic_factory: Callable[[Sequence[DisplayRegion]], HeadDiagnosticSurface],
    tracking: TrackingConfig,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
) -> None:
    """Choose an implicit context from a short passive startup window."""
    contexts: list[ContextVector] = []
    while len(contexts) < tracking.startup_context_samples and not stop.is_set():
        try:
            recovered = await observe_with_head_recovery(
                camera,
                vision,
                regions,
                diagnostic_factory,
                status,
                stop,
                tracking.head_diagnostic_minimum_seconds,
                tracking.head_recovery_timeout_seconds,
                tracking.frame_interval_seconds,
                tracking.head_failure_panel_seconds,
                clock,
                sleep,
            )
        except HeadTrackingError as error:
            raise TrackingError(str(error)) from error
        if recovered is None:
            return
        contexts.append(recovered.observation.context)
        await sleep(tracking.frame_interval_seconds)
    if stop.is_set():
        return
    decision = router.decide(aggregate_features(contexts))
    status.report(
        RuntimeStatus.SELECTING_MODEL,
        f"{decision.label}; confidence {decision.confidence_label}",
    )
    if decision.out_of_distribution:
        status.report(
            RuntimeStatus.TRAINING_RECOMMENDED,
            "startup posture or illumination is outside learned contexts",
        )


async def _calibrate(  # noqa: C901, PLR0912, PLR0913, PLR0915
    camera: CameraCapture,
    vision: VisionEstimator,
    pointer: PointerController,
    topology: DisplayTopology,
    training: TrainingSurface | None,
    hud: DebugHud | None,
    status: StatusSink,
    stop: asyncio.Event,
    tracking: TrackingConfig,
    training_config: TrainingConfig,
    diagnostic_factory: Callable[[Sequence[DisplayRegion]], HeadDiagnosticSurface],
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
    completed_target_sink: Callable[[CollectedTarget], None] | None = None,
) -> _BaseCalibration | None:
    samples: list[CalibrationSample] = []
    collected: list[CollectedTarget] = []
    windows: list[tuple[tuple[FeatureVector, ...], tuple[ContextVector, ...]]] = []
    interim_model: CalibrationModel | None = None
    interim_calibration: IncrementalCalibration | None = None
    interim_smoother = PointerSmoother(
        alpha=tracking.smoothing_alpha,
        dead_zone=tracking.smoothing_dead_zone,
        maximum_step=tracking.smoothing_maximum_step,
    )
    last_pointer_update: float | None = None
    initial_diameters: dict[str, float] = {}
    if training is None:
        targets = calibration_targets(topology)
    else:
        initial_diameters = {
            region.region_id: training.target_diameter(
                region.region_id,
                training_config.physical_target_diameter_mm,
                training_config.fallback_target_diameter,
            )
            for region in topology.regions
        }
        seed_scheduler = RegionSurpriseScheduler(
            topology,
            training_config.precision_threshold,
            confidence_z=training_config.surprise_confidence_z,
            maximum_surprise=training_config.surprise_maximum,
            decay=training_config.surprise_decay,
            maximum_outputs=training_config.surprise_maximum_outputs,
            tail_fraction=training_config.surprise_tail_fraction,
            histogram_bins=training_config.surprise_histogram_bins,
        )
        selected_targets: list[PointerTarget] = []
        for _ in range(training_config.batch_size):
            selection = seed_scheduler.select(initial_diameters)
            selected_targets.append(selection.target)
            seed_scheduler.mark_seed(selection.target)
        targets = tuple(selected_targets)
    if (
        training is not None
        and len(targets) + training_config.batch_size > training_config.maximum_targets
    ):
        msg = "training maximum must cover initial anchors and one unseen batch"
        raise TrackingError(msg)
    for target_index, target in enumerate(targets, start=1):
        if stop.is_set():
            break
        if training is None:
            pointer.move(target.region_id, target.x, target.y)
            await _update_hud(hud, topology, target)
        else:
            diameter = initial_diameters[target.region_id]
            status.report(
                RuntimeStatus.TARGET_PREPARATION,
                f"circle {target_index}/{training_config.maximum_targets}",
            )
            if not await show_target_preparation(
                training,
                target,
                diameter,
                f"Prepare for circle: {target_index}/{training_config.maximum_targets}",
                training_config.preparation_seconds,
                training_config.transition_overlap_seconds,
                stop,
                sleep,
            ):
                break
            status.report(
                RuntimeStatus.TARGET_MEASUREMENT,
                f"circle {target_index}/{training_config.maximum_targets}",
            )
            training.show_target(
                target.region_id,
                target.x,
                target.y,
                diameter,
                f"Training {target_index}/{training_config.maximum_targets}",
            )
        status.report(
            RuntimeStatus.INITIAL_TRAINING,
            f"target {target_index}/{training_config.maximum_targets}",
        )
        if training is None:
            await sleep(tracking.calibration_settle_seconds)
        target_features: list[FeatureVector] = []
        target_contexts: list[ContextVector] = []
        rendered_predictions: list[Point] = []
        unseen_predictions: list[Point] = []
        unseen_errors: list[float] = []
        unseen_uncertainties: list[float] = []
        attempts = 0
        started = clock()
        paused = 0.0
        measurement_complete = False
        global_target = topology.to_global(target)
        while attempts < tracking.calibration_attempts_per_target and not stop.is_set():
            try:
                recovered = await observe_with_head_recovery(
                    camera,
                    vision,
                    topology.regions,
                    diagnostic_factory,
                    status,
                    stop,
                    tracking.head_diagnostic_minimum_seconds,
                    tracking.head_recovery_timeout_seconds,
                    tracking.frame_interval_seconds,
                    tracking.head_failure_panel_seconds,
                    clock,
                    sleep,
                )
            except HeadTrackingError as error:
                if training is not None:
                    await show_training_completion(
                        training,
                        status,
                        target_index - 1,
                        training_config.maximum_targets,
                        "Training failed: head tracking did not recover",
                        training_config.completion_seconds,
                        stop,
                        sleep,
                    )
                    raise TrainingError(str(error)) from error
                raise TrackingError(str(error)) from error
            if recovered is None:
                break
            attempts += 1
            paused += recovered.paused_seconds
            observation = recovered.observation
            if hud is not None:
                hud.set_evidence_context(observation.evidence_class)
            target_features.append(observation.features)
            target_contexts.append(observation.context)
            if training is not None and interim_model is not None:
                estimated, uncertainty = _predict_with_uncertainty(
                    interim_model,
                    observation.features,
                    observation.context,
                )
                clipped = topology.to_global(topology.locate(estimated))
                unseen_predictions.append(clipped)
                unseen_errors.append(
                    math.hypot(clipped.x - global_target.x, clipped.y - global_target.y)
                )
                if uncertainty is not None:
                    unseen_uncertainties.append(uncertainty)
                rendered = rolling_point_median(rendered_predictions, estimated)
                pointer_due = (
                    last_pointer_update is None
                    or observation.timestamp - last_pointer_update
                    >= tracking.pointer_update_interval_seconds
                )
                if pointer_due:
                    visible_target = topology.locate(interim_smoother.update(rendered))
                    pointer.move(
                        visible_target.region_id,
                        visible_target.x,
                        visible_target.y,
                    )
                    await _update_hud(hud, topology, visible_target)
                    last_pointer_update = observation.timestamp
            if training is None:
                measurement_complete = (
                    len(target_features) >= tracking.calibration_samples_per_target
                )
            else:
                measurement_complete = (
                    clock() - started - paused >= training_config.measurement_seconds
                )
            await sleep(tracking.frame_interval_seconds)
            if measurement_complete:
                break
        if not stop.is_set():
            if not measurement_complete or not target_features:
                msg = f"could not obtain reliable head tracking for initial target {target_index}"
                if training is not None:
                    raise TrainingError(msg)
                raise TrackingError(msg)
            features = aggregate_features(target_features)
            feature_dispersion = aggregate_feature_dispersion(target_features)
            context = aggregate_features(target_contexts)
            unseen_noise = cursor_noise_summary(unseen_predictions) if unseen_predictions else None
            samples.append(
                CalibrationSample(
                    features,
                    global_target,
                    target_fit_weight(unseen_noise),
                    context,
                    feature_dispersion,
                )
            )
            completed_target = CollectedTarget(
                features,
                context,
                target,
                _calibration_zone(topology, target),
                noise=unseen_noise,
                feature_dispersion=feature_dispersion,
                unseen_error=(statistics.median(unseen_errors) if unseen_errors else None),
                predictive_uncertainty=(
                    statistics.median(unseen_uncertainties) if unseen_uncertainties else None
                ),
            )
            collected.append(completed_target)
            if completed_target_sink is not None:
                completed_target_sink(completed_target)
            windows.append((tuple(target_features), tuple(target_contexts)))
            if training is not None and len(samples) == MINIMUM_CALIBRATION_SAMPLES:
                interim_calibration = IncrementalCalibration(samples, topology=topology)
                interim_model = interim_calibration.model
            elif training is not None and interim_calibration is not None:
                interim_model = interim_calibration.add(samples[-1])
    if stop.is_set():
        if training is not None:
            await show_training_completion(
                training,
                status,
                len(collected),
                training_config.maximum_targets,
                "Training interrupted",
                0.0,
                stop,
                sleep,
            )
        return None
    model = CalibrationModel.fit(samples, topology=topology)
    if training is not None:
        collected = [
            replace(
                target,
                noise=cursor_noise_summary(
                    tuple(
                        topology.to_global(topology.locate(model.predict(feature, context)))
                        for feature, context in zip(features, contexts, strict=True)
                    )
                ),
            )
            for target, (features, contexts) in zip(collected, windows, strict=True)
        ]
        samples = [
            replace(sample, weight=target_fit_weight(target.noise))
            for sample, target in zip(samples, collected, strict=True)
        ]
        model = CalibrationModel.fit(
            samples,
            routing_contexts=tuple(target.context for target in collected),
            topology=topology,
        )
    return _BaseCalibration(model, tuple(samples), tuple(collected))


def _make_model_anchor(  # noqa: PLR0913
    sequence: int,
    camera_id: str,
    feature_schema: str,
    topology: DisplayTopology,
    model: CalibrationModel,
    contexts: Sequence[tuple[float, ...]],
    metrics: ValidationMetrics,
    validation_target_count: int,
) -> ModelAnchor:
    """Bind one accepted final all-data model to aggregate training context."""
    if (
        not contexts
        or validation_target_count <= 0
        or any(len(context) != len(contexts[0]) for context in contexts)
    ):
        msg = "validated model anchor requires consistent aggregate context and validation"
        raise ValueError(msg)
    dimensions = tuple(zip(*contexts, strict=True))
    model_record = model.to_record()
    model_record["validation_target_count"] = validation_target_count
    return ModelAnchor(
        sequence,
        camera_id,
        feature_schema,
        topology.topology_id,
        describe_topology(topology),
        tuple(statistics.fmean(dimension) for dimension in dimensions),
        tuple(
            max(statistics.pvariance(dimension), ANCHOR_VARIANCE_FLOOR) for dimension in dimensions
        ),
        model_record,
        metrics.median_error,
        metrics.edge_error,
    )


def _retain_model_anchors(anchors: Sequence[ModelAnchor]) -> list[ModelAnchor]:
    """Keep a bounded, deterministic, context-diverse set of validated models."""
    if len(anchors) <= MAXIMUM_MODEL_ANCHORS:
        return list(anchors)
    selected = [
        min(
            anchors,
            key=lambda anchor: (
                max(anchor.median_error, anchor.edge_error),
                anchor.median_error + anchor.edge_error,
                -anchor.sequence,
            ),
        )
    ]
    remaining = [anchor for anchor in anchors if anchor is not selected[0]]
    while remaining and len(selected) < MAXIMUM_MODEL_ANCHORS:
        chosen = max(
            remaining,
            key=lambda anchor: (
                min(_anchor_distance(anchor, prior) for prior in selected),
                -max(anchor.median_error, anchor.edge_error),
                anchor.sequence,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return sorted(selected, key=lambda anchor: anchor.sequence)


def _anchor_distance(left: ModelAnchor, right: ModelAnchor) -> float:
    if (
        left.camera_id != right.camera_id
        or left.feature_schema != right.feature_schema
        or left.topology_id != right.topology_id
        or len(left.context_centroid) != len(right.context_centroid)
    ):
        return math.inf
    return math.sqrt(
        statistics.fmean(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(
                left.context_centroid,
                right.context_centroid,
                strict=True,
            )
        )
    )


def _region_surprise_scheduler(  # noqa: PLR0913
    state: TrainingState,
    topology: DisplayTopology,
    camera_id: str,
    feature_schema: str,
    config: TrainingConfig,
    *,
    seed_targets: Sequence[CollectedTarget] = (),
) -> RegionSurpriseScheduler:
    """Rebuild one bounded scheduler and include non-persistent seed visits."""
    scheduler = RegionSurpriseScheduler.from_stored_targets(
        topology,
        config.precision_threshold,
        state.targets,
        camera_id=camera_id,
        feature_schema=feature_schema,
        confidence_z=config.surprise_confidence_z,
        maximum_surprise=config.surprise_maximum,
        decay=config.surprise_decay,
        maximum_outputs=config.surprise_maximum_outputs,
        tail_fraction=config.surprise_tail_fraction,
        histogram_bins=config.surprise_histogram_bins,
    )
    for target in seed_targets:
        scheduler.mark_seed(target.target)
        if target.unseen_error is not None:
            scheduler.observe(
                target.target,
                target.unseen_error,
                target.predictive_uncertainty,
                None if target.noise is None else target.noise.p95_radial_spread,
                mark_visit=False,
            )
    return scheduler


def _optional_pixels(value: float | None) -> str:
    """Format optional migrated validation evidence without inventing a value."""
    return "n/a" if value is None else f"{value:.0f}px"


def _report_persistent_metrics(status: StatusSink, state: TrainingState) -> None:
    """Report the aggregate holdout score for the router that was actually saved."""
    validation = state.validations[-1]
    status.report(
        RuntimeStatus.TRAINING_VALIDATING,
        (
            "persistent routing: "
            f"median error {validation.median_error:.0f}px, "
            f"edge error {validation.edge_error:.0f}px, "
            f"worst region {validation.maximum_region_error:.0f}px, "
            f"worst CVaR90 {_optional_pixels(validation.maximum_region_cvar90)}, "
            f"maximum bound {_optional_pixels(validation.maximum_region_upper)}"
        ),
    )


def _append_unvalidated_target(  # noqa: PLR0913
    store: TrainingStore,
    existing: TrainingState,
    topology: DisplayTopology,
    camera_id: str,
    feature_schema: str,
    target: CollectedTarget,
) -> TrainingState:
    """Atomically retain one completed aggregate before the next target starts."""
    candidate = copy.deepcopy(existing)
    persistent = make_stored_target(
        candidate.next_sequence,
        camera_id,
        feature_schema,
        target.features,
        target.context,
        topology,
        target.target,
        target.zone,
        target.noise,
        target.feature_dispersion,
        target.unseen_error,
        target.predictive_uncertainty,
    )
    add_target(candidate, persistent)
    store.save(candidate)
    return candidate


def _persist_targets(  # noqa: PLR0913
    store: TrainingStore,
    existing: TrainingState,
    topology: DisplayTopology,
    camera_id: str,
    feature_schema: str,
    targets: Sequence[CollectedTarget],
    median_error: float,
    edge_error: float,
    maximum_region_error: float | None = None,
    maximum_region_cvar90: float | None = None,
    maximum_region_upper: float | None = None,
    *,
    validation_targets: Sequence[CollectedTarget] = (),
    incumbent_metrics: TrainingMetrics | None = None,
    model_accepted: bool = True,
    validated_model: CalibrationModel | None = None,
    append_targets: bool = True,
) -> tuple[ModelRouter, TrainingState] | None:
    """Persist every target and update the active model only after acceptance."""
    measured_metrics = ValidationMetrics(median_error, edge_error)
    measured_region_error = (
        max((target.unseen_error or 0.0 for target in validation_targets), default=0.0)
        if maximum_region_error is None
        else maximum_region_error
    )
    prior_metrics = (
        None
        if incumbent_metrics is None
        else ValidationMetrics(
            incumbent_metrics.median_error,
            incumbent_metrics.edge_error,
        )
    )
    accepted = (
        model_accepted
        and validated_model is not None
        and bool(validation_targets)
        and all(math.isfinite(value) for value in (median_error, edge_error))
        and candidate_is_acceptable(prior_metrics, measured_metrics)
        and (
            incumbent_metrics is None
            or (
                measured_region_error <= incumbent_metrics.maximum_region_error
                and maximum_region_cvar90 is not None
                and maximum_region_cvar90 <= incumbent_metrics.maximum_region_cvar90
                and maximum_region_upper is not None
                and maximum_region_upper <= incumbent_metrics.maximum_region_upper
            )
        )
    )
    candidate = copy.deepcopy(existing)
    assigned_clusters: list[str] = []
    for collected in targets if append_targets else ():
        persistent = make_stored_target(
            candidate.next_sequence,
            camera_id,
            feature_schema,
            collected.features,
            collected.context,
            topology,
            collected.target,
            collected.zone,
            collected.noise,
            collected.feature_dispersion,
            collected.unseen_error,
            collected.predictive_uncertainty,
        )
        assigned_clusters.append(add_target(candidate, persistent))
    if accepted and validated_model is not None:
        anchor_contexts = [target.context for target in targets]
        candidate.anchors.append(
            _make_model_anchor(
                candidate.next_sequence,
                camera_id,
                feature_schema,
                topology,
                validated_model,
                anchor_contexts,
                measured_metrics,
                len(validation_targets),
            )
        )
        candidate.anchors = _retain_model_anchors(candidate.anchors)
    router = build_router(
        candidate,
        topology,
        camera_id=camera_id,
        feature_schema=feature_schema,
    )
    dominant_cluster = (
        Counter(assigned_clusters).most_common(1)[0][0] if assigned_clusters else None
    )
    if accepted and dominant_cluster is not None:
        candidate.clusters = [
            replace(
                cluster,
                median_error=median_error,
                edge_error=edge_error,
            )
            if cluster.cluster_id == dominant_cluster
            else cluster
            for cluster in candidate.clusters
        ]
    if accepted:
        candidate.models = {}
        decision = router.decide(validation_targets[-1].context)
        candidate.validations.append(
            ValidationSummary(
                candidate.next_sequence,
                camera_id,
                topology.topology_id,
                decision.label,
                median_error,
                edge_error,
                measured_region_error,
                maximum_region_cvar90,
                maximum_region_upper,
            )
        )
        candidate.validations = candidate.validations[-64:]
    store.save(candidate)
    return router, candidate


def _smoothing_selection_changed(
    previous: SmoothingSettings | None,
    current: SmoothingSettings,
) -> bool:
    """Rate-limit observable noise-route changes to meaningful differences."""
    if previous is None:
        return current.confidence != "default"
    return (
        previous.confidence != current.confidence
        or abs(previous.alpha - current.alpha) >= SMOOTHING_ALPHA_REPORT_THRESHOLD
        or abs(previous.dead_zone - current.dead_zone) >= SMOOTHING_DEAD_ZONE_REPORT_THRESHOLD
    )


def _display_change_action(
    previous: Sequence[OutputGeometry],
    current: Sequence[OutputGeometry],
    authorized: Sequence[OutputGeometry],
    *,
    allow_pause: bool,
) -> tuple[int | None, str]:
    """Prioritize topology safety before refreshing camera-derived context."""
    previous_set = set(previous)
    current_set = set(current)
    authorized_set = set(authorized)
    removed_authorized = bool(authorized_set - current_set)
    additions_only = previous_set < current_set
    if additions_only:
        if allow_pause:
            return (
                DISPLAY_REAUTHORIZATION_RESULT,
                "new output detected; pausing to refresh portal authorization",
            )
        return (
            None,
            "new output detected; continuing only on the existing authorized union",
        )
    if removed_authorized:
        if allow_pause:
            return (
                DISPLAY_REAUTHORIZATION_RESULT,
                "authorized output geometry changed; pausing to refresh authorization",
            )
        return (
            3,
            "authorized output geometry changed; pause disabled, stopping motion",
        )
    return (None, "non-authorized output geometry changed; authorized union retained")


def _predict_with_uncertainty(
    model: GazePredictor,
    features: FeatureVector,
    context: ContextVector | None,
) -> tuple[Point, float | None]:
    """Use posterior uncertainty when the selected model provides it."""
    if isinstance(model, (CalibrationModel, ModelRouter)):
        return model.predict_with_uncertainty(features, context)
    return model.predict(features, context), None


def _uncertainty_adjusted_alpha(
    base_alpha: float,
    minimum_alpha: float,
    uncertainty: float | None,
) -> float:
    """Slow uncertain posterior motion within configured smoothing bounds."""
    confidence_scale = (
        1.0 if uncertainty is None else 1.0 / (1.0 + uncertainty / BAYESIAN_UNCERTAINTY_SCALE)
    )
    return max(minimum_alpha, base_alpha * confidence_scale)


async def _track(  # noqa: C901, PLR0912, PLR0913, PLR0915
    camera: CameraCapture,
    vision: VisionEstimator,
    pointer: PointerController,
    topology: DisplayTopology,
    calibration: GazePredictor,
    hud: DebugHud | None,
    status: StatusSink,
    stop: asyncio.Event,
    tracking: TrackingConfig,
    training_requested: asyncio.Event | None,
    state: TrainingState,
    camera_id: str,
    feature_schema: str,
    display_monitor: DisplayMonitor | None,
    diagnostic_factory: Callable[[Sequence[DisplayRegion]], HeadDiagnosticSurface],
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
) -> int:
    smoother = PointerSmoother(
        alpha=tracking.smoothing_alpha,
        dead_zone=tracking.smoothing_dead_zone,
        maximum_step=tracking.smoothing_maximum_step,
    )
    last_pointer_update: float | None = None
    last_context_refresh: float | None = None
    routing_context: ContextVector | None = None
    rendered_predictions: list[Point] = []
    last_smoothing: SmoothingSettings | None = None
    base_smoothing_alpha = tracking.smoothing_alpha
    smoothing_bounds = SmoothingBounds(
        tracking.noise_minimum_alpha,
        tracking.noise_maximum_alpha,
        tracking.noise_minimum_dead_zone,
        tracking.noise_maximum_dead_zone,
        tracking.noise_minimum_samples,
        tracking.noise_maximum_targets,
    )
    available_geometry = display_monitor.snapshot() if display_monitor is not None else None
    authorized_geometry = tuple(
        sorted((region.x, region.y, region.width, region.height) for region in topology.regions)
    )
    while not stop.is_set():
        if training_requested is not None and training_requested.is_set():
            return TRAINING_REQUESTED_RESULT
        if bool(getattr(pointer, "closed", False)):
            status.report(RuntimeStatus.RECALIBRATION_REQUIRED, "desktop authorization closed")
            return 3
        try:
            recovered = await observe_with_head_recovery(
                camera,
                vision,
                topology.regions,
                diagnostic_factory,
                status,
                stop,
                tracking.head_diagnostic_minimum_seconds,
                tracking.head_recovery_timeout_seconds,
                tracking.frame_interval_seconds,
                tracking.head_failure_panel_seconds,
                clock,
                sleep,
            )
        except HeadTrackingError as error:
            raise TrackingError(str(error)) from error
        if recovered is None:
            return 0
        observation = recovered.observation
        if hud is not None:
            hud.set_evidence_context(observation.evidence_class)
        context_due = (
            last_context_refresh is None
            or observation.timestamp - last_context_refresh
            >= tracking.context_refresh_interval_seconds
        )
        if context_due:
            if display_monitor is not None and available_geometry is not None:
                refreshed_geometry = display_monitor.snapshot()
                if refreshed_geometry != available_geometry:
                    action, detail = _display_change_action(
                        available_geometry,
                        refreshed_geometry,
                        authorized_geometry,
                        allow_pause=tracking.allow_display_reauthorization_pause,
                    )
                    status.report(RuntimeStatus.DISPLAY_CHANGE, detail)
                    available_geometry = refreshed_geometry
                    if action is not None:
                        return action
            routing_context = observation.context
            selected_smoothing = noise_smoothing_for(
                state,
                topology,
                camera_id=camera_id,
                feature_schema=feature_schema,
                context=routing_context,
                defaults=(tracking.smoothing_alpha, tracking.smoothing_dead_zone),
                bounds=smoothing_bounds,
            )
            base_smoothing_alpha = selected_smoothing.alpha
            smoother.alpha = base_smoothing_alpha
            smoother.dead_zone = selected_smoothing.dead_zone
            if _smoothing_selection_changed(last_smoothing, selected_smoothing):
                status.report(
                    RuntimeStatus.NOISE_MODEL_SELECTION,
                    (
                        f"{selected_smoothing.confidence}; alpha "
                        f"{selected_smoothing.alpha:.2f}; dead-zone "
                        f"{selected_smoothing.dead_zone:.1f}px"
                    ),
                )
                if hud is not None:
                    hud.set_noise_context(
                        selected_smoothing.confidence,
                        selected_smoothing.alpha,
                        selected_smoothing.dead_zone,
                    )
            last_smoothing = selected_smoothing
            last_context_refresh = observation.timestamp
        estimated, uncertainty = _predict_with_uncertainty(
            calibration,
            observation.features,
            routing_context,
        )
        rendered = rolling_point_median(rendered_predictions, estimated)
        pointer_due = (
            last_pointer_update is None
            or observation.timestamp - last_pointer_update
            >= tracking.pointer_update_interval_seconds
        )
        if pointer_due:
            smoother.alpha = _uncertainty_adjusted_alpha(
                base_smoothing_alpha,
                tracking.noise_minimum_alpha,
                uncertainty,
            )
            _set_hud_model_context(hud, calibration)
            target = topology.locate(smoother.update(rendered))
            pointer.move(target.region_id, target.x, target.y)
            await _update_hud(hud, topology, target)
            last_pointer_update = observation.timestamp
        await sleep(tracking.frame_interval_seconds)
    return 0


def _calibration_zone(topology: DisplayTopology, target: PointerTarget) -> str:
    region = topology.region(target.region_id)
    horizontal = target.x / region.width
    vertical = target.y / region.height
    far_edge = 1.0 - CALIBRATION_EDGE_BOUNDARY
    x_edge = horizontal < CALIBRATION_EDGE_BOUNDARY or horizontal > far_edge
    y_edge = vertical < CALIBRATION_EDGE_BOUNDARY or vertical > far_edge
    if x_edge and y_edge:
        return "corner"
    if x_edge or y_edge:
        return "edge"
    return "center"


def _set_hud_model_context(
    hud: DebugHud | None,
    model: GazePredictor,
) -> None:
    if hud is None:
        return
    if isinstance(model, ModelRouter) and model.last_decision is not None:
        decision = model.last_decision
        hud.set_model_context(
            decision.label,
            decision.topology_quality.name.lower(),
            decision.confidence_label,
        )
    else:
        hud.set_model_context(model.kind, "current-session", "session-only")


async def _update_hud(
    hud: DebugHud | None,
    topology: DisplayTopology,
    target: PointerTarget,
) -> None:
    """Publish one global coordinate when diagnostics are enabled."""
    if hud is None:
        return
    point = topology.to_global(target)
    await hud.update(target.region_id, point.x, point.y)


def install_signal_handlers(stop: asyncio.Event) -> None:
    """Map interactive and release-triggered termination onto one stop event."""
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
