"""Head/face tracking recovery without pupil-required gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gazeebo.contracts import EyeObservation, RuntimeStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from gazeebo.contracts import (
        CameraCapture,
        DisplayRegion,
        HeadDiagnosticSurface,
        StatusSink,
        VisionEstimator,
    )


SUSTAINED_FAILURE_SECONDS = 0.75


class HeadTrackingError(RuntimeError):
    """Reliable head/face tracking did not recover before its deadline."""


@dataclass(frozen=True, slots=True)
class RecoveredObservation:
    """One reliable observation and time excluded from active measurement."""

    observation: EyeObservation
    paused_seconds: float


async def observe_with_head_recovery(  # noqa: PLR0913
    camera: CameraCapture,
    vision: VisionEstimator,
    regions: Sequence[DisplayRegion],
    diagnostic_factory: Callable[[Sequence[DisplayRegion]], HeadDiagnosticSurface],
    status: StatusSink,
    stop: object,
    minimum_duration: float,
    recovery_timeout: float,
    frame_interval: float,
    failure_panel_seconds: float,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
    confirmation_duration: float = SUSTAINED_FAILURE_SECONDS,
) -> RecoveredObservation | None:
    """Pause on missing head evidence, render guidance, and recover or fail."""
    frame = camera.read()
    result = vision.observe(frame, clock())
    if isinstance(result, EyeObservation):
        return RecoveredObservation(result, 0.0)

    started = clock()
    while clock() - started < confirmation_duration and not _stop_is_set(stop):
        await sleep(frame_interval)
        frame = camera.read()
        candidate = vision.observe(frame, clock())
        if isinstance(candidate, EyeObservation):
            return RecoveredObservation(candidate, clock() - started)
        result = candidate
    if _stop_is_set(stop):
        return None
    capture_warning = getattr(vision, "capture_warning", None)
    if callable(capture_warning):
        capture_warning(result, clock())
    surface = diagnostic_factory(regions)
    display_started = clock()
    recovered: EyeObservation | None = None
    status.report(RuntimeStatus.HEAD_TRACKING_RECOVERY, result.reason)
    try:
        while not _stop_is_set(stop):
            elapsed = clock() - started
            recovery_remaining = max(0.0, recovery_timeout - elapsed)
            minimum_remaining = max(0.0, minimum_duration - (clock() - display_started))
            remaining = minimum_remaining if recovered is not None else recovery_remaining
            surface.show_head_diagnostic(frame, result, remaining)
            if recovered is not None and minimum_remaining <= 0.0:
                surface.hide_head_diagnostic()
                return RecoveredObservation(recovered, elapsed)
            if recovered is None and recovery_remaining <= 0.0:
                status.report(RuntimeStatus.CAMERA_ERROR, result.reason)
                await sleep(max(failure_panel_seconds, minimum_remaining))
                raise HeadTrackingError(result.reason)
            await sleep(frame_interval)
            frame = camera.read()
            candidate = vision.observe(frame, clock())
            if isinstance(candidate, EyeObservation):
                recovered = candidate
            else:
                recovered = None
                result = candidate
        return None
    finally:
        await surface.close()


def _stop_is_set(stop: object) -> bool:
    method = getattr(stop, "is_set", None)
    return bool(method()) if callable(method) else False
