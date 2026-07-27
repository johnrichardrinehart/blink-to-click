"""Bounded rough-in confidence regions and transient cursor refinement."""

from __future__ import annotations

import asyncio
import math
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from gazeebo.adaptation import map_stored_target
from gazeebo.contracts import RuntimeStatus
from gazeebo.geometry import DisplayTopology, Point

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from gazeebo.contracts import (
        DebugHud,
        DisplayRegion,
        InputCaptureSession,
        PointerController,
        RefinementSurface,
        StatusSink,
    )
    from gazeebo.control import ControlCommand
    from gazeebo.state import StoredTarget

TRAINING_GRID_SIZE = 3
MINIMUM_REFINEMENT_ROWS = 2
MAXIMUM_REFINEMENT_ROWS = 6
MINIMUM_REFINEMENT_COLUMNS = 2
MAXIMUM_REFINEMENT_COLUMNS = 6
DEFAULT_REFINEMENT_ROWS = ("123", "456", "789")
P99_QUANTILE = 0.99
DEFAULT_MINIMUM_SAMPLES = 100
DEFAULT_HISTOGRAM_BINS = 1024
DEFAULT_MAXIMUM_RESIDUAL = 10000.0
DEFAULT_MAXIMUM_DEPTH = 6
DEFAULT_MINIMUM_CELL_SIZE = 12.0
DEFAULT_SETTLE_SECONDS = 0.5
MINIMUM_HISTOGRAM_BINS = 32
MAXIMUM_HISTOGRAM_BINS = 4096
MAXIMUM_REFINEMENT_DEPTH = 12


def refinement_rows(
    cli_rows: Sequence[str] | None,
    *,
    path: Path | None = None,
) -> tuple[str, ...]:
    """Resolve complete CLI-over-TOML rows with a numeric 3x3 default."""
    if cli_rows is not None:
        rows = tuple(cli_rows)
        _validate_refinement_rows(rows)
        return rows
    config = path or _default_config_path()
    if not config.exists():
        return DEFAULT_REFINEMENT_ROWS
    if config.is_symlink() or not config.is_file():
        msg = "refinement configuration must be a regular file"
        raise ValueError(msg)
    try:
        raw = tomllib.loads(config.read_text())
        section = raw.get("refinement", {})
        configured = section.get("rows", list(DEFAULT_REFINEMENT_ROWS))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, AttributeError) as error:
        msg = "refinement configuration is malformed"
        raise ValueError(msg) from error
    if not isinstance(configured, list) or not all(isinstance(row, str) for row in configured):
        msg = "refinement rows must be a TOML array of strings"
        raise ValueError(msg)
    rows = tuple(configured)
    _validate_refinement_rows(rows)
    return rows


def _validate_refinement_rows(rows: tuple[str, ...]) -> None:
    if not MINIMUM_REFINEMENT_ROWS <= len(rows) <= MAXIMUM_REFINEMENT_ROWS:
        msg = "refinement matrix must have between 2 and 6 rows"
        raise ValueError(msg)
    columns = len(rows[0])
    if not MINIMUM_REFINEMENT_COLUMNS <= columns <= MAXIMUM_REFINEMENT_COLUMNS:
        msg = "refinement matrix must have between 2 and 6 columns"
        raise ValueError(msg)
    if any(len(row) != columns for row in rows):
        msg = "refinement matrix rows must have equal length"
        raise ValueError(msg)
    labels = "".join(rows)
    if any(
        not character.isascii() or not character.isprintable() or character.isspace()
        for character in labels
    ):
        msg = "refinement labels must be printable non-whitespace ASCII characters"
        raise ValueError(msg)
    if len(set(labels)) != len(labels):
        msg = "refinement labels must be distinct"
        raise ValueError(msg)


def _default_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home) if config_home else Path.home() / ".config"
    return root / "gazeebo" / "config.toml"


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    """Finite confidence, recursion, and settling controls."""

    width_override: float | None = None
    height_override: float | None = None
    minimum_samples: int = DEFAULT_MINIMUM_SAMPLES
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS
    maximum_residual: float = DEFAULT_MAXIMUM_RESIDUAL
    maximum_depth: int = DEFAULT_MAXIMUM_DEPTH
    minimum_cell_size: float = DEFAULT_MINIMUM_CELL_SIZE
    settle_seconds: float = DEFAULT_SETTLE_SECONDS
    rows: tuple[str, ...] = DEFAULT_REFINEMENT_ROWS

    def __post_init__(self) -> None:
        """Reject unbounded or unusable refinement policy."""
        optional = (self.width_override, self.height_override)
        if any(
            value is not None and (not math.isfinite(value) or value <= 0.0) for value in optional
        ):
            msg = "rough-in dimension overrides must be finite and positive"
            raise ValueError(msg)
        _validate_refinement_rows(self.rows)
        finite = (self.maximum_residual, self.minimum_cell_size, self.settle_seconds)
        if (
            not all(math.isfinite(value) for value in finite)
            or self.minimum_samples <= 0
            or not MINIMUM_HISTOGRAM_BINS <= self.histogram_bins <= MAXIMUM_HISTOGRAM_BINS
            or self.maximum_residual <= 0.0
            or not 0 <= self.maximum_depth <= MAXIMUM_REFINEMENT_DEPTH
            or self.minimum_cell_size <= 0.0
            or self.settle_seconds < 0.0
        ):
            msg = "rough-in refinement bounds are invalid"
            raise ValueError(msg)

    @property
    def row_count(self) -> int:
        """Return the configured recursive matrix row count."""
        return len(self.rows)

    @property
    def column_count(self) -> int:
        """Return the configured recursive matrix column count."""
        return len(self.rows[0])

    def locate_label(self, label: str) -> tuple[int, int]:
        """Resolve one configured selection character to row and column."""
        for row, labels in enumerate(self.rows):
            column = labels.find(label)
            if column >= 0:
                return row, column
        msg = f"refinement label is not active: {label!r}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ConfidenceRectangle:
    """One global logical rough-in rectangle and its evidence label."""

    left: float
    top: float
    width: float
    height: float
    source: str
    samples: int

    def __post_init__(self) -> None:
        """Reject non-finite or empty confidence geometry."""
        values = (self.left, self.top, self.width, self.height)
        if (
            not all(math.isfinite(value) for value in values)
            or self.width <= 0.0
            or self.height <= 0.0
            or not self.source
            or self.samples < 0
        ):
            msg = "rough-in confidence rectangle is invalid"
            raise ValueError(msg)

    @property
    def center(self) -> Point:
        """Return the rectangle center in global logical coordinates."""
        return Point(self.left + self.width / 2.0, self.top + self.height / 2.0)

    def cell(
        self,
        row: int,
        column: int,
        row_count: int,
        column_count: int,
    ) -> ConfidenceRectangle:
        """Return one row-and-column child of a bounded rectangular matrix."""
        if not 0 <= row < row_count or not 0 <= column < column_count:
            msg = "refinement matrix cell is out of range"
            raise ValueError(msg)
        width = self.width / column_count
        height = self.height / row_count
        return ConfidenceRectangle(
            self.left + column * width,
            self.top + row * height,
            width,
            height,
            self.source,
            self.samples,
        )


@dataclass(slots=True)
class _Histogram:
    """A fixed-size nearest-rank absolute-residual summary."""

    bins: int
    maximum: float
    counts: list[int] = field(init=False)
    observations: int = 0

    def __post_init__(self) -> None:
        self.counts = [0] * self.bins

    def add(self, value: float) -> None:
        bounded = min(abs(value), self.maximum)
        index = min(self.bins - 1, int(bounded * self.bins / self.maximum))
        self.counts[index] += 1
        self.observations += 1

    def upper_p99(self) -> float:
        """Return the conservative upper edge of the nearest-rank p99 bin."""
        if self.observations <= 0:
            msg = "p99 requires residual evidence"
            raise ValueError(msg)
        rank = max(1, math.ceil(P99_QUANTILE * self.observations))
        cumulative = 0
        for index, count in enumerate(self.counts):
            cumulative += count
            if cumulative >= rank:
                return (index + 1) * self.maximum / self.bins
        return self.maximum


@dataclass(slots=True)
class _Evidence:
    horizontal: _Histogram
    vertical: _Histogram
    radial: _Histogram

    @classmethod
    def create(cls, config: RefinementConfig) -> _Evidence:
        return cls(
            _Histogram(config.histogram_bins, config.maximum_residual),
            _Histogram(config.histogram_bins, config.maximum_residual),
            _Histogram(config.histogram_bins, config.maximum_residual),
        )

    @property
    def component_samples(self) -> int:
        return min(self.horizontal.observations, self.vertical.observations)


class ConfidenceRegionEstimator:
    """Build bounded p99 evidence once and answer rough-ins in fixed work."""

    def __init__(
        self,
        topology: DisplayTopology,
        targets: Sequence[StoredTarget],
        *,
        camera_id: str,
        feature_schema: str,
        config: RefinementConfig | None = None,
    ) -> None:
        """Build fixed evidence tables from compatible retained targets."""
        self.topology = topology
        self.config = config or RefinementConfig()
        self._global = _Evidence.create(self.config)
        self._outputs = {
            region.region_id: _Evidence.create(self.config) for region in topology.regions
        }
        self._regions = {
            (region.region_id, row, column): _Evidence.create(self.config)
            for region in topology.regions
            for row in range(TRAINING_GRID_SIZE)
            for column in range(TRAINING_GRID_SIZE)
        }
        self.work_updates = 0
        for target in sorted(targets, key=lambda item: item.sequence):
            if target.camera_id != camera_id or target.feature_schema != feature_schema:
                continue
            self._add_target(target)

    def rectangle(self, rough_point: Point) -> ConfidenceRectangle:
        """Return one projected, bounded confidence rectangle around a rough point."""
        local = self.topology.locate(rough_point)
        center = self.topology.to_global(local)
        output = self.topology.region(local.region_id)
        key = (
            local.region_id,
            min(
                TRAINING_GRID_SIZE - 1,
                int(TRAINING_GRID_SIZE * local.y / output.height),
            ),
            min(
                TRAINING_GRID_SIZE - 1,
                int(TRAINING_GRID_SIZE * local.x / output.width),
            ),
        )
        width, height, source, samples = self._dimensions(key, local.region_id)
        bounds = _topology_bounds(self.topology)
        bounded_width = min(max(width, self.config.minimum_cell_size), bounds.width)
        bounded_height = min(max(height, self.config.minimum_cell_size), bounds.height)
        if bounded_width != width or bounded_height != height:
            source = f"{source}+dimension-clipped"
        source = f"{source}+authorized-union-intersection"
        return ConfidenceRectangle(
            center.x - bounded_width / 2.0,
            center.y - bounded_height / 2.0,
            bounded_width,
            bounded_height,
            source,
            samples,
        )

    def _dimensions(
        self,
        key: tuple[str, int, int],
        output_key: str,
    ) -> tuple[float, float, str, int]:
        selected: tuple[float, float, str, int] | None = None
        hierarchy = (
            (self._regions[key], "region"),
            (self._outputs[output_key], "output"),
            (self._global, "global"),
        )
        for evidence, source in hierarchy:
            if evidence.component_samples >= self.config.minimum_samples:
                selected = (
                    2.0 * evidence.horizontal.upper_p99(),
                    2.0 * evidence.vertical.upper_p99(),
                    source,
                    evidence.component_samples,
                )
                break
        if selected is None:
            for evidence, source in hierarchy:
                if evidence.radial.observations >= self.config.minimum_samples:
                    diameter = 2.0 * evidence.radial.upper_p99()
                    selected = (
                        diameter,
                        diameter,
                        f"legacy-{source}",
                        evidence.radial.observations,
                    )
                    break
        if selected is None:
            bounds = _topology_bounds(self.topology)
            selected = (bounds.width, bounds.height, "topology-fallback", 0)
        width, height, source, samples = selected
        if self.config.width_override is not None:
            width = self.config.width_override
            source = f"override-width+{source}"
        if self.config.height_override is not None:
            height = self.config.height_override
            source = f"override-height+{source}"
        return width, height, source, samples

    def _add_target(self, target: StoredTarget) -> None:
        mapped = map_stored_target(target, self.topology)
        if mapped is None:
            return
        local = self.topology.locate(mapped.point)
        output = self.topology.region(local.region_id)
        row = min(
            TRAINING_GRID_SIZE - 1,
            int(TRAINING_GRID_SIZE * local.y / output.height),
        )
        column = min(
            TRAINING_GRID_SIZE - 1,
            int(TRAINING_GRID_SIZE * local.x / output.width),
        )
        evidence = (
            self._global,
            self._outputs[local.region_id],
            self._regions[(local.region_id, row, column)],
        )
        source = next(item for item in target.outputs if item.key == target.output_key)
        horizontal_scale = output.width / source.width
        vertical_scale = output.height / source.height
        if target.horizontal_residual is not None and target.vertical_residual is not None:
            for item in evidence:
                item.horizontal.add(target.horizontal_residual * horizontal_scale)
                item.vertical.add(target.vertical_residual * vertical_scale)
        if target.unseen_error is not None:
            radial_scale = max(horizontal_scale, vertical_scale)
            for item in evidence:
                item.radial.add(target.unseen_error * radial_scale)
        self.work_updates += 1


@dataclass(slots=True)
class RefinementSession:
    """Hold one finite recursive grid without owning input actions."""

    topology: DisplayTopology
    config: RefinementConfig
    rectangle: ConfidenceRectangle | None = None
    depth: int = 0
    active: bool = False

    def start(self, rectangle: ConfidenceRectangle) -> Point:
        """Start at one confidence rectangle and return its projected center."""
        self.rectangle = rectangle
        self.depth = 0
        self.active = True
        return self._projected_center()

    def select(self, label: str) -> Point:
        """Select one configured matrix character and return its projected center."""
        if not self.active or self.rectangle is None:
            msg = "refinement grid is not active"
            raise ValueError(msg)
        if self.depth >= self.config.maximum_depth:
            msg = "refinement grid reached its maximum depth"
            raise ValueError(msg)
        row, column = self.config.locate_label(label)
        child = self.rectangle.cell(
            row,
            column,
            self.config.row_count,
            self.config.column_count,
        )
        if (
            child.width < self.config.minimum_cell_size
            or child.height < self.config.minimum_cell_size
        ):
            msg = "refinement cell is below the configured minimum size"
            raise ValueError(msg)
        self.rectangle = child
        self.depth += 1
        return self._projected_center()

    def accept(self) -> Point:
        """Finish refinement and return the projected current center."""
        if not self.active or self.rectangle is None:
            msg = "refinement grid is not active"
            raise ValueError(msg)
        point = self._projected_center()
        self.active = False
        return point

    def cancel(self) -> None:
        """Discard the active grid without retaining its history."""
        self.active = False
        self.rectangle = None
        self.depth = 0

    def _projected_center(self) -> Point:
        rectangle = self.rectangle
        if rectangle is None:
            msg = "refinement rectangle is unavailable"
            raise ValueError(msg)
        return self.topology.to_global(self.topology.locate(rectangle.center))


class RefinementController:
    """Coordinate one process-local grid and manual-refinement hold."""

    def __init__(  # noqa: PLR0913
        self,
        topology: DisplayTopology,
        estimator: ConfidenceRegionEstimator,
        pointer: PointerController,
        status: StatusSink,
        surface_factory: Callable[[Sequence[DisplayRegion]], RefinementSurface] | None,
        *,
        config: RefinementConfig,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        hud: DebugHud | None = None,
        capture_authorizer: (
            Callable[
                [
                    Callable[[int, float, float], None],
                    Callable[[str], None],
                ],
                Awaitable[InputCaptureSession],
            ]
            | None
        ) = None,
    ) -> None:
        """Bind pure refinement state to pointer-only runtime resources."""
        self.topology = topology
        self.estimator = estimator
        self.pointer = pointer
        self.status = status
        self.surface_factory = surface_factory
        self.config = config
        self.sleep = sleep
        self.hud = hud
        self.capture_authorizer = capture_authorizer
        self.session = RefinementSession(topology, config)
        self.latest_rough: Point | None = None
        self.held = False
        self.manual = False
        self._surface: RefinementSurface | None = None
        self._recorder: SettledPositionRecorder | None = None
        self._capture: InputCaptureSession | None = None
        self._cleanup_tasks: set[asyncio.Task[None]] = set()

    def update_rough(self, point: Point) -> None:
        """Remember only the latest projected gaze estimate."""
        self.latest_rough = self.topology.to_global(self.topology.locate(point))

    async def handle(self, command: ControlCommand) -> bool:  # noqa: PLR0911
        """Apply one validated owner command, returning whether it was accepted."""
        try:
            if command.kind == "refine":
                return await self._start()
            if command.kind == "cell":
                point = self.session.select(command.label)
                self._move(point)
                self._show_grid()
                detail = f"depth {self.session.depth}; cell {command.label!r}"
                self.status.report(RuntimeStatus.REFINEMENT, detail)
                self._set_hud_context(detail, point)
                return True
            if command.kind == "accept":
                point = self.session.accept()
                await self._start_manual(point)
                return True
            if command.kind == "cancel":
                await self._cancel()
                return True
            if command.kind in {"move", "position"}:
                return self._report_motion(command)
            if command.kind == "capture":
                return await self._authorize_capture()
        except (IndexError, ValueError) as error:
            self.status.report(RuntimeStatus.INPUT_ERROR, str(error))
            return False
        return False

    async def close(self) -> None:
        """Release grid, recorder, timer, and native surfaces idempotently."""
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            await recorder.close()
        capture, self._capture = self._capture, None
        if capture is not None:
            await capture.close()
        surface, self._surface = self._surface, None
        if surface is not None:
            await surface.close()
        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            self._cleanup_tasks.clear()
        self.session.cancel()
        self.held = False
        self.manual = False

    async def _start(self) -> bool:
        point = self.latest_rough
        if point is None:
            self.status.report(RuntimeStatus.INPUT_ERROR, "rough-in has no gaze position yet")
            return False
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            await recorder.close()
        rectangle = self.estimator.rectangle(point)
        center = self.session.start(rectangle)
        self.held = True
        self.manual = False
        self._move(center)
        self._show_grid()
        detail = (
            f"pinned {center.x:.0f},{center.y:.0f}; "
            f"{self.config.row_count}x{self.config.column_count}; "
            f"{rectangle.width:.0f}x{rectangle.height:.0f}px; "
            f"{rectangle.source}; samples {rectangle.samples}"
        )
        self.status.report(RuntimeStatus.ROUGH_IN, detail)
        self._set_hud_context(detail, center)
        return True

    async def _start_manual(self, point: Point) -> None:
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            await recorder.close()
        self.held = True
        self.manual = True
        self._recorder = SettledPositionRecorder(
            self.topology,
            point,
            self._settled,
            delay=self.config.settle_seconds,
            sleep=self.sleep,
        )
        self._show_manual_rectangle(point)
        detail = f"manual rectangle; settling {self.config.settle_seconds * 1000.0:.0f}ms"
        self.status.report(RuntimeStatus.REFINEMENT, detail)
        self._set_hud_context(detail, point)

    def _report_motion(self, command: ControlCommand) -> bool:
        recorder = self._recorder
        if not self.manual or recorder is None:
            self.status.report(RuntimeStatus.INPUT_ERROR, "manual refinement is not active")
            return False
        if command.kind == "move":
            point = recorder.report_relative(*command.values)
        else:
            point = recorder.report_absolute(Point(*command.values))
        self._move(point)
        self._show_manual_rectangle(point)
        detail = f"manual rectangle; settling {self.config.settle_seconds * 1000.0:.0f}ms"
        self.status.report(RuntimeStatus.REFINEMENT, detail)
        self._set_hud_context(detail, point)
        return True

    async def _cancel(self) -> None:
        point = self._recorder.position if self._recorder is not None else self.latest_rough
        self.session.cancel()
        self._hide_grid()
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            await recorder.close()
        capture, self._capture = self._capture, None
        if capture is not None:
            await capture.close()
        self.held = False
        self.manual = False
        self.status.report(RuntimeStatus.ACTIVE, "refinement cancelled")
        if point is not None:
            self._set_hud_context("inactive", point)

    async def _authorize_capture(self) -> bool:
        if not self.manual or self._recorder is None:
            self.status.report(RuntimeStatus.INPUT_ERROR, "manual refinement is not active")
            return False
        if self.capture_authorizer is None:
            self.status.report(
                RuntimeStatus.INPUT_CAPTURE,
                "portal capture is unavailable; use socket motion reports",
            )
            return False
        capture, self._capture = self._capture, None
        if capture is not None:
            await capture.close()
        try:
            self._capture = await self.capture_authorizer(
                self._captured_motion,
                self._capture_disconnected,
            )
        except RuntimeError as error:
            self.status.report(
                RuntimeStatus.INPUT_CAPTURE,
                f"{error}; use socket motion reports",
            )
            return False
        self.status.report(
            RuntimeStatus.INPUT_CAPTURE,
            f"enabled with {self._capture.barrier_count} boundary barriers",
        )
        return True

    def _captured_motion(self, absolute: int, x: float, y: float) -> None:
        recorder = self._recorder
        if not self.manual or recorder is None:
            return
        point = (
            recorder.report_absolute(Point(x, y)) if absolute else recorder.report_relative(x, y)
        )
        self._move(point)
        self._show_manual_rectangle(point)
        detail = f"captured rectangle; settling {self.config.settle_seconds * 1000.0:.0f}ms"
        self.status.report(RuntimeStatus.REFINEMENT, detail)
        self._set_hud_context(detail, point)

    def _capture_disconnected(self, detail: str) -> None:
        self.status.report(
            RuntimeStatus.INPUT_CAPTURE,
            f"{detail}; use socket motion reports",
        )
        capture, self._capture = self._capture, None
        if capture is not None:
            task = asyncio.create_task(capture.close())
            self._cleanup_tasks.add(task)
            task.add_done_callback(self._cleanup_tasks.discard)

    def _settled(self, point: Point) -> None:
        self._hide_grid()
        detail = f"settled {point.x:.0f},{point.y:.0f}"
        self.status.report(RuntimeStatus.REFINEMENT_SETTLED, detail)
        self._set_hud_context(detail, point)

    def _move(self, point: Point) -> None:
        target = self.topology.locate(point)
        self.pointer.move(target.region_id, target.x, target.y)

    def _show_grid(self) -> None:
        rectangle = self.session.rectangle
        if rectangle is None:
            return
        if self._surface is None:
            if self.surface_factory is None:
                msg = "refinement surface is unavailable"
                raise ValueError(msg)
            self._surface = self.surface_factory(self.topology.regions)
        self._surface.show_refinement(
            rectangle.left,
            rectangle.top,
            rectangle.width,
            rectangle.height,
            self.session.depth,
            rectangle.source,
            self.config.rows,
        )

    def _show_manual_rectangle(self, point: Point) -> None:
        rectangle = self.session.rectangle
        if rectangle is None:
            return
        projected = self.topology.to_global(self.topology.locate(point))
        source = (
            rectangle.source
            if rectangle.source.endswith("+manual-follow")
            else f"{rectangle.source}+manual-follow"
        )
        self.session.rectangle = ConfidenceRectangle(
            projected.x - rectangle.width / 2.0,
            projected.y - rectangle.height / 2.0,
            rectangle.width,
            rectangle.height,
            source,
            rectangle.samples,
        )
        self._show_grid()

    def _set_hud_context(self, context: str, point: Point) -> None:
        if self.hud is None:
            return
        self.hud.set_refinement_context(context)
        target = self.topology.locate(point)
        global_point = self.topology.to_global(target)
        task = asyncio.create_task(
            self.hud.update(target.region_id, global_point.x, global_point.y)
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    def _hide_grid(self) -> None:
        if self._surface is not None:
            self._surface.hide_refinement()


class SettledPositionRecorder:
    """Keep only the latest authorized position and debounce its final report."""

    def __init__(
        self,
        topology: DisplayTopology,
        initial: Point,
        on_settled: Callable[[Point], None],
        *,
        delay: float = DEFAULT_SETTLE_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Start from a known point with an injected final-position callback."""
        if not math.isfinite(delay) or delay < 0.0:
            msg = "settle delay must be finite and non-negative"
            raise ValueError(msg)
        self.topology = topology
        self.position = self._project(initial)
        self.settled_position: Point | None = None
        self._on_settled = on_settled
        self._delay = delay
        self._sleep = sleep
        self._generation = 0
        self._task: asyncio.Task[None] | None = None
        self._restart()

    def report_absolute(self, point: Point) -> Point:
        """Replace the latest position and restart settling."""
        self.position = self._project(point)
        self._restart()
        return self.position

    def report_relative(self, delta_x: float, delta_y: float) -> Point:
        """Apply one finite delta to the latest authorized position."""
        if not math.isfinite(delta_x) or not math.isfinite(delta_y):
            msg = "relative refinement motion must be finite"
            raise ValueError(msg)
        return self.report_absolute(Point(self.position.x + delta_x, self.position.y + delta_y))

    async def close(self) -> None:
        """Cancel unsettled motion and release the timer idempotently."""
        task, self._task = self._task, None
        self._generation += 1
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _restart(self) -> None:
        self._generation += 1
        generation = self._generation
        if self._task is not None:
            self._task.cancel()
        self._task = asyncio.create_task(self._settle(generation))

    async def _settle(self, generation: int) -> None:
        try:
            await self._sleep(self._delay)
        except asyncio.CancelledError:
            return
        if generation != self._generation:
            return
        self.settled_position = self.position
        self._on_settled(self.position)

    def _project(self, point: Point) -> Point:
        return self.topology.to_global(self.topology.locate(point))


def _topology_bounds(topology: DisplayTopology) -> ConfidenceRectangle:
    left = min(region.x for region in topology.regions)
    top = min(region.y for region in topology.regions)
    right = max(region.right for region in topology.regions)
    bottom = max(region.bottom for region in topology.regions)
    return ConfidenceRectangle(left, top, right - left, bottom - top, "topology", 0)
