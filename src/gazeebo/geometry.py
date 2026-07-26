"""Authorized-display geometry and pointer smoothing."""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gazeebo.contracts import DisplayRegion

MAXIMUM_CALIBRATION_INSET = 0.5
PREDICTION_POINT_MEDIAN_WINDOW = 15


@dataclass(frozen=True, slots=True)
class Point:
    """A point in the desktop's global logical coordinate space."""

    x: float
    y: float


def rolling_point_median(
    history: list[Point],
    point: Point,
    window_size: int = PREDICTION_POINT_MEDIAN_WINDOW,
) -> Point:
    """Reject transient coordinate outliers without animating cursor transit."""
    if window_size <= 0:
        msg = "prediction point median window must be positive"
        raise ValueError(msg)
    history.append(point)
    del history[:-window_size]
    return Point(
        statistics.median(item.x for item in history),
        statistics.median(item.y for item in history),
    )


@dataclass(frozen=True, slots=True)
class PointerTarget:
    """A region-local pointer destination."""

    region_id: str
    x: float
    y: float


class DisplayTopology:
    """Project global predictions onto the authorized display union."""

    def __init__(self, regions: tuple[DisplayRegion, ...]) -> None:
        """Build an immutable topology from one or more authorized regions."""
        if not regions:
            msg = "at least one authorized display region is required"
            raise ValueError(msg)
        identifiers = [region.region_id for region in regions]
        if len(set(identifiers)) != len(identifiers):
            msg = "authorized display region IDs must be unique"
            raise ValueError(msg)
        self._regions = regions
        identity = "\n".join(
            f"{region.x}:{region.y}:{region.width}:{region.height}"
            for region in sorted(
                regions,
                key=lambda item: (item.x, item.y, item.width, item.height),
            )
        )
        self._topology_id = hashlib.sha256(identity.encode()).hexdigest()

    @property
    def regions(self) -> tuple[DisplayRegion, ...]:
        """Return every authorized display region."""
        return self._regions

    @property
    def topology_id(self) -> str:
        """Return a stable identity for this exact geometry."""
        return self._topology_id

    def region_containing(self, point: Point) -> DisplayRegion | None:
        """Return the authorized region containing a global point, if any."""
        return next(
            (
                region
                for region in self._regions
                if region.x <= point.x < region.right and region.y <= point.y < region.bottom
            ),
            None,
        )

    def region(self, region_id: str) -> DisplayRegion:
        """Return one authorized region by its opaque ID."""
        region = next(
            (item for item in self._regions if item.region_id == region_id),
            None,
        )
        if region is None:
            msg = f"unknown authorized display region: {region_id}"
            raise ValueError(msg)
        return region

    def locate(self, point: Point) -> PointerTarget:
        """Keep a global point inside the nearest authorized region."""
        candidates: list[tuple[float, DisplayRegion, float, float]] = []
        for region in self._regions:
            local_x = min(
                max(point.x - region.x, 0.0),
                math.nextafter(region.width, 0.0),
            )
            local_y = min(
                max(point.y - region.y, 0.0),
                math.nextafter(region.height, 0.0),
            )
            global_x = region.x + local_x
            global_y = region.y + local_y
            distance = math.hypot(point.x - global_x, point.y - global_y)
            candidates.append((distance, region, local_x, local_y))
        _, region, local_x, local_y = min(
            candidates,
            key=lambda item: (item[0], item[1].region_id),
        )
        return PointerTarget(region.region_id, local_x, local_y)

    def to_global(self, target: PointerTarget) -> Point:
        """Convert a validated local target into global coordinates."""
        region = self.region(target.region_id)
        if not 0 <= target.x < region.width or not 0 <= target.y < region.height:
            msg = "pointer target is outside its authorized region"
            raise ValueError(msg)
        return Point(region.x + target.x, region.y + target.y)


def calibration_targets(
    topology: DisplayTopology,
    inset: float = 0.12,
) -> tuple[PointerTarget, ...]:
    """Return center and inset-corner targets on every authorized display."""
    if not 0.0 < inset < MAXIMUM_CALIBRATION_INSET:
        msg = "calibration inset must be between zero and one half"
        raise ValueError(msg)
    normalized = (
        (inset, inset),
        (1.0 - inset, inset),
        (0.5, 0.5),
        (inset, 1.0 - inset),
        (1.0 - inset, 1.0 - inset),
    )
    return tuple(
        PointerTarget(
            region.region_id,
            min(region.width - 1.0, horizontal * region.width),
            min(region.height - 1.0, vertical * region.height),
        )
        for region in topology.regions
        for horizontal, vertical in normalized
    )


@dataclass(slots=True)
class PointerSmoother:
    """Reduce gaze jitter while bounding sudden movement."""

    alpha: float = 0.35
    dead_zone: float = 6.0
    maximum_step: float = 600.0
    _current: Point | None = None

    def __post_init__(self) -> None:
        """Validate smoothing parameters."""
        if not 0.0 < self.alpha <= 1.0:
            msg = "smoothing alpha must be in (0, 1]"
            raise ValueError(msg)
        if self.dead_zone < 0.0 or self.maximum_step <= 0.0:
            msg = "smoothing distances must be non-negative with a positive maximum step"
            raise ValueError(msg)

    def update(self, target: Point) -> Point:
        """Return a smoothed target and update in-memory state."""
        if self._current is None:
            self._current = target
            return target

        delta_x = target.x - self._current.x
        delta_y = target.y - self._current.y
        distance = math.hypot(delta_x, delta_y)
        if distance <= self.dead_zone:
            return self._current

        step_x = delta_x * self.alpha
        step_y = delta_y * self.alpha
        step_distance = math.hypot(step_x, step_y)
        if step_distance > self.maximum_step:
            scale = self.maximum_step / step_distance
            step_x *= scale
            step_y *= scale

        self._current = Point(self._current.x + step_x, self._current.y + step_y)
        return self._current

    def reset(self) -> None:
        """Discard session-only smoothing state."""
        self._current = None
