"""Deterministic in-memory implementations of runtime boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections import deque

    from gazeebo.contracts import (
        DisplayRegion,
        EyeObservation,
        Frame,
        HeadTrackingFailure,
        RuntimeStatus,
        TrainingRegionStatus,
    )


class _ObservationQueue(Protocol):
    def popleft(self) -> EyeObservation | HeadTrackingFailure | None:
        """Return one configured vision result."""

    def __len__(self) -> int:
        """Return the number of pending fixture results."""


@dataclass(slots=True)
class FakeCamera:
    """Yield a fixed frame sequence and record cleanup."""

    frames: deque[Frame]
    camera_id: str = "fixture-camera"
    closed: bool = False

    def read(self) -> Frame:
        """Return the next configured frame."""
        if self.closed:
            msg = "camera is closed"
            raise RuntimeError(msg)
        if not self.frames:
            msg = "camera fixture is exhausted"
            raise EOFError(msg)
        return self.frames.popleft()

    def close(self) -> None:
        """Record idempotent closure."""
        self.closed = True


@dataclass(slots=True)
class FakeVision:
    """Yield configured observations without inspecting fixture frames."""

    observations: _ObservationQueue
    closed: bool = False

    def observe(self, frame: Frame, timestamp: float) -> EyeObservation | HeadTrackingFailure:
        """Return the next configured observation or generic head-loss guidance."""
        del frame, timestamp
        if self.closed:
            msg = "vision estimator is closed"
            raise RuntimeError(msg)
        result = self.observations.popleft()
        if result is None:
            from gazeebo.contracts import HeadTrackingFailure  # noqa: PLC0415

            return HeadTrackingFailure("fixture head tracking unavailable")
        return result

    def close(self) -> None:
        """Record idempotent closure."""
        self.closed = True


@dataclass(slots=True)
class FakePointer:
    """Record pointer events within fixed authorized geometry."""

    regions: tuple[DisplayRegion, ...]
    topology_id: str = "fixture-layout"
    moves: list[tuple[str, float, float]] = field(default_factory=list)
    closed: bool = False

    def move(self, region_id: str, x: float, y: float) -> None:
        """Record a validated region-local movement."""
        region = next((item for item in self.regions if item.region_id == region_id), None)
        if region is None:
            msg = f"unknown fixture region: {region_id}"
            raise ValueError(msg)
        if not 0 <= x < region.width or not 0 <= y < region.height:
            msg = "fixture movement is outside its region"
            raise ValueError(msg)
        self.moves.append((region_id, x, y))

    async def close(self) -> None:
        """Record idempotent closure."""
        self.closed = True


@dataclass(slots=True)
class FakeHud:
    """Record opt-in pointer diagnostics and cleanup."""

    updates: list[tuple[str, float, float]] = field(default_factory=list)
    model_context: tuple[str, str, str] = ("unselected", "unknown", "unknown")
    noise_context: tuple[str, float, float] = ("default", 0.0, 0.0)
    evidence_context: str = "head+face"
    training_region: tuple[str, float, float, float, int, int, str] = (
        "inactive",
        0.0,
        0.0,
        0.0,
        0,
        0,
        "inactive",
    )
    closed: bool = False

    def set_model_context(
        self,
        routing: str,
        topology_quality: str,
        model_confidence: str,
    ) -> None:
        """Record safe inferred routing and quality labels."""
        self.model_context = (routing, topology_quality, model_confidence)

    def set_noise_context(self, confidence: str, alpha: float, dead_zone: float) -> None:
        """Record bounded inferred smoothing diagnostics."""
        self.noise_context = (confidence, alpha, dead_zone)

    def set_evidence_context(self, evidence_class: str) -> None:
        """Record the current head-only or pupil-refined evidence class."""
        self.evidence_context = evidence_class

    def set_training_region(self, context: TrainingRegionStatus) -> None:
        """Record one bounded region-aware target selection."""
        self.training_region = (
            context.region,
            context.cvar90,
            context.surprise_lower,
            context.surprise_upper,
            context.observed_regions,
            context.total_regions,
            context.mode,
        )

    async def update(self, region_id: str, x: float, y: float) -> None:
        """Record one global pointer diagnostic."""
        self.updates.append((region_id, x, y))

    async def close(self) -> None:
        """Record idempotent closure."""
        self.closed = True


@dataclass(slots=True)
class FakeTraining:
    """Record transient calibration targets and cleanup."""

    targets: list[tuple[str, float, float, float, str]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    preparations: list[tuple[str, float, float, float, str, float]] = field(default_factory=list)
    diagnostics: list[tuple[Frame, HeadTrackingFailure, float]] = field(default_factory=list)
    diagnostic_hides: int = 0
    diameters: dict[str, float] = field(default_factory=dict)
    closed: bool = False

    def show_target(
        self,
        region_id: str,
        x: float,
        y: float,
        diameter: float,
        label: str,
    ) -> None:
        """Record one target without opening a surface."""
        self.targets.append((region_id, x, y, diameter, label))

    def show_message(self, label: str) -> None:
        """Record one message shown across every fixture output."""
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
        """Record one all-output preparation and fade step."""
        self.preparations.append((region_id, x, y, diameter, label, prior_opacity))

    def show_head_diagnostic(
        self,
        frame: Frame,
        failure: HeadTrackingFailure,
        seconds_remaining: float,
    ) -> None:
        """Record one transient head-recovery frame."""
        self.diagnostics.append((frame, failure, seconds_remaining))

    def hide_head_diagnostic(self) -> None:
        """Record removal of transient diagnostic pixels."""
        self.diagnostic_hides += 1

    def target_diameter(
        self,
        region_id: str,
        _physical_millimetres: float,
        fallback_pixels: float,
    ) -> float:
        """Return configured physical sizing or the pixel fallback."""
        return self.diameters.get(region_id, fallback_pixels)

    async def close(self) -> None:
        """Record idempotent surface cleanup."""
        self.closed = True


@dataclass(slots=True)
class FakeStatus:
    """Collect status transitions."""

    reports: list[tuple[RuntimeStatus, str]] = field(default_factory=list)

    def report(self, status: RuntimeStatus, detail: str = "") -> None:
        """Record a transition."""
        self.reports.append((status, detail))
