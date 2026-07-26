"""Bounded region-aware surprise scheduling for adaptive training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gazeebo.adaptation import map_stored_target
from gazeebo.geometry import PointerTarget

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from gazeebo.geometry import DisplayTopology
    from gazeebo.state import StoredTarget

REGION_GRID_SIZE = 3
MAXIMUM_SURPRISE_OUTPUTS = 16
MAXIMUM_REGION_SEQUENCE = 1 << 20
_DEFAULT_CONFIDENCE_Z = 1.6448536269514722
_DEFAULT_MAXIMUM_SURPRISE = 100.0
_DEFAULT_DECAY = 0.95
_DEFAULT_TAIL_FRACTION = 0.10
_MAXIMUM_TAIL_FRACTION = 0.50
_DEFAULT_HISTOGRAM_BINS = 1024
_MINIMUM_HISTOGRAM_BINS = 32
_MAXIMUM_HISTOGRAM_BINS = 4096
_MINIMUM_SURPRISE_VARIANCE = 0.25
_MAXIMUM_NOISE_WIDENING = 3.0
_CELL_ORDER = (
    (1, 1),
    (0, 0),
    (0, 2),
    (2, 0),
    (2, 2),
    (0, 1),
    (1, 0),
    (1, 2),
    (2, 1),
)
_CELL_RANK = {cell: rank for rank, cell in enumerate(_CELL_ORDER)}


@dataclass(frozen=True, order=True, slots=True)
class RegionKey:
    """One normalized 3x3 cell on an authorized output."""

    output_key: str
    row: int
    column: int

    def __post_init__(self) -> None:
        """Reject cells outside the fixed partition."""
        if (
            not self.output_key
            or not 0 <= self.row < REGION_GRID_SIZE
            or not 0 <= self.column < REGION_GRID_SIZE
        ):
            msg = "surprise region is invalid"
            raise ValueError(msg)

    @property
    def label(self) -> str:
        """Return a stable output-relative cell label."""
        return f"{self.output_key}:{self.row},{self.column}"

    @property
    def edge_or_corner(self) -> bool:
        """Return whether this cell touches an output boundary."""
        return self.row != 1 or self.column != 1


@dataclass(frozen=True, slots=True)
class RegionEstimate:
    """One finite posterior interval used for target selection."""

    key: RegionKey
    visits: int
    observations: int
    cvar90: float
    tail_variance: float
    effective_tail_count: float
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class RegionSelection:
    """One target selected before its eventual observation is available."""

    target: PointerTarget
    key: RegionKey
    mode: str
    estimate: RegionEstimate
    seeded_regions: int
    observed_regions: int
    total_regions: int

    @property
    def summary(self) -> str:
        """Describe selection and bounded region evidence without feature data."""
        return (
            f"region {self.key.label}; {self.mode}; CVaR90 "
            f"{self.estimate.cvar90:.2f}; surprise "
            f"{self.estimate.lower:.2f}..{self.estimate.upper:.2f}; "
            f"coverage {self.observed_regions}/{self.total_regions} observed, "
            f"{self.seeded_regions}/{self.total_regions} seeded"
        )


@dataclass(frozen=True, slots=True)
class SurpriseWork:
    """Structural work counters independent of historical target count."""

    updates: int
    region_scans: int
    selections: int
    target_probes: int
    tail_bin_updates: int
    tail_bin_scans: int


@dataclass(slots=True)
class _RegionMoments:
    visits: int = 0
    observations: int = 0
    weight: float = 0.0
    squared_weight: float = 0.0
    uncertainty_sum: float = 0.0
    noise_sum: float = 0.0
    histogram_mass: list[float] = field(default_factory=list)
    histogram_squared_mass: list[float] = field(default_factory=list)
    histogram_error_sum: list[float] = field(default_factory=list)
    histogram_error_square_sum: list[float] = field(default_factory=list)

    @classmethod
    def create(cls, histogram_bins: int) -> _RegionMoments:
        """Allocate one fixed-size regional tail summary."""
        return cls(
            histogram_mass=[0.0] * histogram_bins,
            histogram_squared_mass=[0.0] * histogram_bins,
            histogram_error_sum=[0.0] * histogram_bins,
            histogram_error_square_sum=[0.0] * histogram_bins,
        )

    def mark_visit(self) -> None:
        if self.visits >= MAXIMUM_REGION_SEQUENCE:
            msg = "surprise region exhausted its bounded target sequence"
            raise ValueError(msg)
        self.visits += 1

    def observe(
        self,
        normalized_error: float,
        normalized_uncertainty: float,
        normalized_noise: float,
        decay: float,
        maximum_surprise: float,
    ) -> None:
        self.observations += 1
        self.weight = decay * self.weight + 1.0
        self.squared_weight = decay * decay * self.squared_weight + 1.0
        self.uncertainty_sum = decay * self.uncertainty_sum + normalized_uncertainty
        self.noise_sum = decay * self.noise_sum + normalized_noise
        for index in range(len(self.histogram_mass)):
            self.histogram_mass[index] *= decay
            self.histogram_squared_mass[index] *= decay * decay
            self.histogram_error_sum[index] *= decay
            self.histogram_error_square_sum[index] *= decay
        bin_index = min(
            len(self.histogram_mass) - 1,
            int(normalized_error / maximum_surprise * len(self.histogram_mass)),
        )
        self.histogram_mass[bin_index] += 1.0
        self.histogram_squared_mass[bin_index] += 1.0
        self.histogram_error_sum[bin_index] += normalized_error
        self.histogram_error_square_sum[bin_index] += normalized_error**2


@dataclass(slots=True)
class RegionSurpriseScheduler:
    """Select high-surprise output cells with fixed bounded per-target work."""

    topology: DisplayTopology
    precision_threshold: float
    confidence_z: float = _DEFAULT_CONFIDENCE_Z
    maximum_surprise: float = _DEFAULT_MAXIMUM_SURPRISE
    decay: float = _DEFAULT_DECAY
    maximum_outputs: int = MAXIMUM_SURPRISE_OUTPUTS
    tail_fraction: float = _DEFAULT_TAIL_FRACTION
    histogram_bins: int = _DEFAULT_HISTOGRAM_BINS
    _moments: dict[RegionKey, _RegionMoments] = field(init=False, repr=False)
    _output_order: dict[str, int] = field(init=False, repr=False)
    _used_coordinates: dict[RegionKey, set[tuple[float, float]]] = field(
        init=False,
        repr=False,
    )
    _updates: int = field(default=0, init=False, repr=False)
    _region_scans: int = field(default=0, init=False, repr=False)
    _selections: int = field(default=0, init=False, repr=False)
    _target_probes: int = field(default=0, init=False, repr=False)
    _tail_bin_updates: int = field(default=0, init=False, repr=False)
    _tail_bin_scans: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create one fixed region table and reject unbounded policy."""
        values = (
            self.precision_threshold,
            self.confidence_z,
            self.maximum_surprise,
            self.decay,
            self.tail_fraction,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or self.precision_threshold <= 0.0
            or self.confidence_z <= 0.0
            or self.maximum_surprise <= 0.0
            or not 0.0 < self.decay <= 1.0
            or self.maximum_outputs <= 0
            or not 0.0 < self.tail_fraction <= _MAXIMUM_TAIL_FRACTION
            or not _MINIMUM_HISTOGRAM_BINS <= self.histogram_bins <= _MAXIMUM_HISTOGRAM_BINS
        ):
            msg = "surprise scheduling bounds are invalid"
            raise ValueError(msg)
        if len(self.topology.regions) > self.maximum_outputs:
            msg = "authorized output count exceeds the bounded surprise scheduler"
            raise ValueError(msg)
        self._output_order = {
            region.region_id: index for index, region in enumerate(self.topology.regions)
        }
        self._moments = {
            RegionKey(region.region_id, row, column): _RegionMoments.create(self.histogram_bins)
            for region in self.topology.regions
            for row in range(REGION_GRID_SIZE)
            for column in range(REGION_GRID_SIZE)
        }
        self._used_coordinates = {key: set() for key in self._moments}

    @classmethod
    def from_stored_targets(  # noqa: PLR0913
        cls,
        topology: DisplayTopology,
        precision_threshold: float,
        targets: Sequence[StoredTarget],
        *,
        camera_id: str,
        feature_schema: str,
        confidence_z: float = _DEFAULT_CONFIDENCE_Z,
        maximum_surprise: float = _DEFAULT_MAXIMUM_SURPRISE,
        decay: float = _DEFAULT_DECAY,
        maximum_outputs: int = MAXIMUM_SURPRISE_OUTPUTS,
        tail_fraction: float = _DEFAULT_TAIL_FRACTION,
        histogram_bins: int = _DEFAULT_HISTOGRAM_BINS,
    ) -> RegionSurpriseScheduler:
        """Rebuild bounded moments in one chronological corpus pass."""
        scheduler = cls(
            topology,
            precision_threshold,
            confidence_z,
            maximum_surprise,
            decay,
            maximum_outputs,
            tail_fraction,
            histogram_bins,
        )
        for target in sorted(targets, key=lambda item: item.sequence):
            if target.camera_id != camera_id or target.feature_schema != feature_schema:
                continue
            mapped = map_stored_target(target, topology)
            if mapped is None:
                continue
            local = topology.locate(mapped.point)
            scheduler.mark_seed(local)
            if target.unseen_error is None:
                continue
            source = next(output for output in target.outputs if output.key == target.output_key)
            current = topology.region(local.region_id)
            scale = math.hypot(current.width, current.height) / math.hypot(
                source.width,
                source.height,
            )
            scheduler.observe(
                local,
                target.unseen_error * scale,
                None
                if target.predictive_uncertainty is None
                else target.predictive_uncertainty * scale,
                None if target.noise is None else target.noise.p95_radial_spread * scale,
                mark_visit=False,
            )
        return scheduler

    @property
    def work(self) -> SurpriseWork:
        """Return structural counters for deterministic complexity tests."""
        return SurpriseWork(
            self._updates,
            self._region_scans,
            self._selections,
            self._target_probes,
            self._tail_bin_updates,
            self._tail_bin_scans,
        )

    @property
    def region_keys(self) -> tuple[RegionKey, ...]:
        """Return the fixed authorized region keys in deterministic order."""
        return tuple(self._moments)

    @property
    def total_regions(self) -> int:
        """Return the fixed region-table size."""
        return len(self._moments)

    @property
    def seeded_regions(self) -> int:
        """Return cells that have received at least one target."""
        return sum(moment.visits > 0 for moment in self._moments.values())

    @property
    def observed_regions(self) -> int:
        """Return cells with pre-incorporation error evidence."""
        return sum(moment.observations > 0 for moment in self._moments.values())

    @property
    def equalized(self) -> bool:
        """Return whether every cell is observed and all intervals overlap."""
        if self.observed_regions != self.total_regions:
            return False
        estimates = tuple(self._estimate(key) for key in self._moments)
        return max(item.lower for item in estimates) <= min(item.upper for item in estimates)

    @property
    def regions_precise(self) -> bool:
        """Require every regional CVaR90 upper bound to meet precision."""
        return self.observed_regions == self.total_regions and all(
            self._estimate(key).upper <= 1.0 for key in self._moments
        )

    def mark_seed(self, target: PointerTarget) -> RegionKey:
        """Record a target visit without inventing pre-incorporation error."""
        key = self.region_for_target(target)
        self._moments[key].mark_visit()
        self._remember_coordinate(key, target)
        return key

    def observe(
        self,
        target: PointerTarget,
        radial_error: float,
        predictive_uncertainty: float | None,
        noise_spread: float | None,
        *,
        mark_visit: bool = True,
    ) -> RegionEstimate:
        """Update exactly one cell after a target's unseen measurement."""
        values = (radial_error, predictive_uncertainty, noise_spread)
        if any(value is not None and (not math.isfinite(value) or value < 0.0) for value in values):
            msg = "surprise observations must be finite and non-negative"
            raise ValueError(msg)
        key = self.region_for_target(target)
        moments = self._moments[key]
        if mark_visit:
            moments.mark_visit()
            self._remember_coordinate(key, target)
        normalized_error = min(radial_error / self.precision_threshold, self.maximum_surprise)
        normalized_uncertainty = min(
            0.0
            if predictive_uncertainty is None
            else predictive_uncertainty / self.precision_threshold,
            self.maximum_surprise,
        )
        normalized_noise = min(
            0.0 if noise_spread is None else noise_spread / self.precision_threshold,
            _MAXIMUM_NOISE_WIDENING,
        )
        moments.observe(
            normalized_error,
            normalized_uncertainty,
            normalized_noise,
            self.decay,
            self.maximum_surprise,
        )
        self._updates += 1
        self._tail_bin_updates += self.histogram_bins
        return self._estimate(key)

    def select(self, diameters: Mapping[str, float]) -> RegionSelection:
        """Choose one target using only bounded pre-target region state."""
        if set(diameters) != set(self._output_order):
            msg = "surprise target sizes must cover every authorized output"
            raise ValueError(msg)
        for region in self.topology.regions:
            diameter = diameters[region.region_id]
            if (
                not math.isfinite(diameter)
                or diameter <= 0.0
                or diameter >= min(region.width, region.height)
            ):
                msg = "surprise targets must fit every authorized output"
                raise ValueError(msg)

        estimates = {key: self._estimate(key) for key in self._moments}
        self._region_scans += len(estimates)
        unseeded = [item for item in estimates.values() if item.visits == 0]
        if unseeded:
            chosen = min(unseeded, key=self._balanced_key)
            mode = "seeding"
        else:
            unobserved = [item for item in estimates.values() if item.observations == 0]
            if unobserved:
                chosen = min(unobserved, key=self._balanced_key)
                mode = "uncertainty-exploration"
            else:
                minimum_upper = min(item.upper for item in estimates.values())
                dominant = [item for item in estimates.values() if item.lower > minimum_upper]
                if dominant:
                    chosen = max(
                        dominant,
                        key=lambda item: (
                            item.upper,
                            item.lower,
                            -item.observations,
                            -_CELL_RANK[(item.key.row, item.key.column)],
                            -self._output_order[item.key.output_key],
                        ),
                    )
                    mode = "high-surprise"
                else:
                    chosen = min(estimates.values(), key=self._balanced_key)
                    mode = "balanced-equalized"
        moments = self._moments[chosen.key]
        target = self._unseen_target_for(
            chosen.key,
            moments.visits,
            diameters[chosen.key.output_key],
        )
        self._selections += 1
        return RegionSelection(
            target,
            chosen.key,
            mode,
            chosen,
            self.seeded_regions,
            self.observed_regions,
            self.total_regions,
        )

    def estimate(self, key: RegionKey) -> RegionEstimate:
        """Expose one bounded estimate for tests, reports, and the HUD."""
        if key not in self._moments:
            msg = "surprise region is not authorized"
            raise ValueError(msg)
        return self._estimate(key)

    def region_for_target(self, target: PointerTarget) -> RegionKey:
        """Map one authorized local target into its normalized 3x3 cell."""
        region = self.topology.region(target.region_id)
        if not 0.0 <= target.x < region.width or not 0.0 <= target.y < region.height:
            msg = "surprise target lies outside its authorized output"
            raise ValueError(msg)
        column = min(REGION_GRID_SIZE - 1, int(REGION_GRID_SIZE * target.x / region.width))
        row = min(REGION_GRID_SIZE - 1, int(REGION_GRID_SIZE * target.y / region.height))
        return RegionKey(target.region_id, row, column)

    def _estimate(self, key: RegionKey) -> RegionEstimate:
        moments = self._moments[key]
        if moments.observations == 0 or moments.weight <= 0.0:
            return RegionEstimate(
                key,
                moments.visits,
                moments.observations,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                self.maximum_surprise,
            )
        cvar90, tail_variance, effective_tail_count = self._tail_statistics(moments)
        uncertainty_mean = moments.uncertainty_sum / moments.weight
        mean = min(self.maximum_surprise, cvar90 + uncertainty_mean)
        sampling_radius = self.confidence_z * math.sqrt(
            max(tail_variance, _MINIMUM_SURPRISE_VARIANCE) / effective_tail_count
        )
        noise_radius = min(
            moments.noise_sum / moments.weight,
            _MAXIMUM_NOISE_WIDENING,
        )
        radius = sampling_radius + noise_radius
        return RegionEstimate(
            key,
            moments.visits,
            moments.observations,
            cvar90,
            tail_variance,
            effective_tail_count,
            mean,
            max(0.0, mean - radius),
            min(self.maximum_surprise, mean + radius),
        )

    def _tail_statistics(self, moments: _RegionMoments) -> tuple[float, float, float]:
        """Estimate the exact weighted worst-tail moments within fixed bins."""
        tail_weight = moments.weight * self.tail_fraction
        remaining = tail_weight
        error_sum = 0.0
        error_square_sum = 0.0
        squared_weight = 0.0
        for index in range(self.histogram_bins - 1, -1, -1):
            self._tail_bin_scans += 1
            mass = moments.histogram_mass[index]
            if mass <= 0.0:
                continue
            fraction = min(1.0, remaining / mass)
            selected_mass = fraction * mass
            error_sum += fraction * moments.histogram_error_sum[index]
            error_square_sum += fraction * moments.histogram_error_square_sum[index]
            squared_weight += fraction * moments.histogram_squared_mass[index]
            remaining -= selected_mass
            if remaining <= max(1e-15, tail_weight * 1e-12):
                break
        if tail_weight <= 0.0 or error_sum < 0.0:
            msg = "regional CVaR90 tail summary is invalid"
            raise ValueError(msg)
        cvar90 = error_sum / tail_weight
        variance = max(0.0, error_square_sum / tail_weight - cvar90**2)
        effective_count = max(
            1.0,
            tail_weight**2 / squared_weight if squared_weight > 0.0 else 1.0,
        )
        return cvar90, variance, effective_count

    def _balanced_key(self, estimate: RegionEstimate) -> tuple[int, int, int, int]:
        return (
            estimate.observations,
            estimate.visits,
            _CELL_RANK[(estimate.key.row, estimate.key.column)],
            self._output_order[estimate.key.output_key],
        )

    def _unseen_target_for(
        self,
        key: RegionKey,
        visit_index: int,
        diameter: float,
    ) -> PointerTarget:
        for sequence_index in range(visit_index, MAXIMUM_REGION_SEQUENCE):
            self._target_probes += 1
            target = self._target_for(key, sequence_index, diameter)
            if (target.x, target.y) not in self._used_coordinates[key]:
                return target
        msg = "surprise region exhausted its bounded target sequence"
        raise ValueError(msg)

    def _target_for(self, key: RegionKey, sequence_index: int, diameter: float) -> PointerTarget:
        region = self.topology.region(key.output_key)
        if sequence_index == 0:
            local_x = 0.5
            local_y = 0.5
        else:
            local_x = _fixed_radical_inverse(sequence_index, 2, 20)
            local_y = _fixed_radical_inverse(sequence_index, 3, 13)
        normalized_x = (key.column + local_x) / REGION_GRID_SIZE
        normalized_y = (key.row + local_y) / REGION_GRID_SIZE
        radius = diameter / 2.0
        x = min(region.width - radius, max(radius, normalized_x * region.width))
        y = min(region.height - radius, max(radius, normalized_y * region.height))
        return PointerTarget(region.region_id, x, y)

    def _remember_coordinate(self, key: RegionKey, target: PointerTarget) -> None:
        used = self._used_coordinates[key]
        if len(used) >= MAXIMUM_REGION_SEQUENCE:
            msg = "surprise region exhausted its bounded target sequence"
            raise ValueError(msg)
        used.add((target.x, target.y))


def _fixed_radical_inverse(index: int, base: int, digits: int) -> float:
    """Return a bounded-work radical inverse over one fixed finite sequence."""
    result = 0.0
    fraction = 1.0
    for _ in range(digits):
        fraction /= base
        result += fraction * (index % base)
        index //= base
    return result
