"""Persistent target labels and best-effort display-topology adaptation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from gazeebo.contracts import DisplayRegion
from gazeebo.geometry import DisplayTopology, Point, PointerTarget
from gazeebo.state import CursorNoiseSummary, OutputDescriptor, StoredTarget

if TYPE_CHECKING:
    from gazeebo.contracts import FeatureVector


class TopologyQuality(IntEnum):
    """How strongly stored labels correspond to current portal geometry."""

    WEAK = 0
    STRONG = 1
    EXACT = 2


@dataclass(frozen=True, slots=True)
class MappedTarget:
    """One stored target projected into current global logical coordinates."""

    point: Point
    quality: TopologyQuality


def describe_topology(topology: DisplayTopology) -> tuple[OutputDescriptor, ...]:
    """Convert authorized regions into generic persistent descriptors."""
    return tuple(
        OutputDescriptor(
            region.region_id,
            region.x,
            region.y,
            region.width,
            region.height,
        )
        for region in topology.regions
    )


def make_stored_target(  # noqa: PLR0913
    sequence: int,
    camera_id: str,
    feature_schema: str,
    features: FeatureVector,
    context: tuple[float, ...],
    topology: DisplayTopology,
    target: PointerTarget,
    zone: str,
    noise: CursorNoiseSummary | None = None,
    feature_dispersion: FeatureVector = (),
    unseen_error: float | None = None,
    predictive_uncertainty: float | None = None,
    horizontal_residual: float | None = None,
    vertical_residual: float | None = None,
) -> StoredTarget:
    """Represent one current target in output-local and topology-relative space."""
    region = topology.region(target.region_id)
    point = topology.to_global(target)
    left, top, width, height = _bounds(describe_topology(topology))
    return StoredTarget(
        sequence=sequence,
        camera_id=camera_id,
        feature_schema=feature_schema,
        features=features,
        context=context,
        outputs=describe_topology(topology),
        output_key=region.region_id,
        target_u=_normalize(target.x, region.width),
        target_v=_normalize(target.y, region.height),
        desktop_u=_normalize(point.x - left, width),
        desktop_v=_normalize(point.y - top, height),
        zone=zone,
        noise=noise,
        feature_dispersion=feature_dispersion,
        unseen_error=unseen_error,
        predictive_uncertainty=predictive_uncertainty,
        horizontal_residual=horizontal_residual,
        vertical_residual=vertical_residual,
    )


def map_stored_target(
    target: StoredTarget,
    topology: DisplayTopology,
) -> MappedTarget | None:
    """Map one stored label onto current geometry, excluding removed outputs."""
    current = describe_topology(topology)
    source_output = next(output for output in target.outputs if output.key == target.output_key)
    exact_topology = _same_topology(target.outputs, current)

    matched = _match_output(source_output, target.outputs, current)

    if matched is not None:
        point = Point(
            matched.x + _denormalize(target.target_u, matched.width),
            matched.y + _denormalize(target.target_v, matched.height),
        )
        quality = TopologyQuality.EXACT if exact_topology else TopologyQuality.STRONG
        if len(target.outputs) != len(current):
            quality = TopologyQuality.WEAK
        return MappedTarget(point, quality)

    if len(current) < len(target.outputs):
        return None

    left, top, width, height = _bounds(current)
    fallback = Point(
        left + _denormalize(target.desktop_u, width),
        top + _denormalize(target.desktop_v, height),
    )
    projected = topology.to_global(topology.locate(fallback))
    return MappedTarget(projected, TopologyQuality.WEAK)


def topology_quality(
    targets: tuple[StoredTarget, ...],
    topology: DisplayTopology,
) -> TopologyQuality:
    """Return the weakest usable mapping quality across stored targets."""
    mapped = [map_stored_target(target, topology) for target in targets]
    usable = [item for item in mapped if item is not None]
    if not usable:
        return TopologyQuality.WEAK
    return min(item.quality for item in usable)


def output_mapping_quality(
    source: tuple[OutputDescriptor, ...],
    topology: DisplayTopology,
) -> TopologyQuality:
    """Classify whether every source output maps unambiguously to current geometry."""
    current = describe_topology(topology)
    if _same_topology(source, current):
        return TopologyQuality.EXACT
    if len(source) != len(current):
        return TopologyQuality.WEAK
    if all(_match_output(output, source, current) is not None for output in source):
        return TopologyQuality.STRONG
    return TopologyQuality.WEAK


def model_mapping_supported(
    source: tuple[OutputDescriptor, ...],
    topology: DisplayTopology,
) -> bool:
    """Return whether every current output has one surviving source correspondence."""
    current = describe_topology(topology)
    if len(current) > len(source):
        return False
    matched = {
        output.key
        for source_output in source
        if (output := _match_output(source_output, source, current)) is not None
    }
    return matched == {output.key for output in current}


def map_model_point(
    point: Point,
    source: tuple[OutputDescriptor, ...],
    topology: DisplayTopology,
) -> Point:
    """Map a validated source-topology prediction into authorized current geometry."""
    current = describe_topology(topology)
    source_topology = topology_from_outputs(source)
    local = source_topology.locate(point)
    source_output = next(output for output in source if output.key == local.region_id)
    matched = _match_output(source_output, source, current)
    if matched is not None:
        mapped = Point(
            matched.x + _denormalize(_normalize(local.x, source_output.width), matched.width),
            matched.y + _denormalize(_normalize(local.y, source_output.height), matched.height),
        )
        return topology.to_global(topology.locate(mapped))

    source_left, source_top, source_width, source_height = _bounds(source)
    left, top, width, height = _bounds(current)
    fallback = Point(
        left + _denormalize(_normalize(point.x - source_left, source_width), width),
        top + _denormalize(_normalize(point.y - source_top, source_height), height),
    )
    return topology.to_global(topology.locate(fallback))


def topology_from_outputs(outputs: tuple[OutputDescriptor, ...]) -> DisplayTopology:
    """Reconstruct generic source geometry from persisted output descriptors."""
    return DisplayTopology(
        tuple(
            DisplayRegion(output.key, output.x, output.y, output.width, output.height)
            for output in outputs
        )
    )


def topology_id_for_outputs(outputs: tuple[OutputDescriptor, ...]) -> str:
    """Return the geometry-only identity for persisted output descriptors."""
    return topology_from_outputs(outputs).topology_id


def legacy_topology_id(outputs: tuple[OutputDescriptor, ...]) -> str:
    """Reproduce the old opaque-ID identity for existing validated model records."""
    identity = "\n".join(
        f"{output.key}:{output.x}:{output.y}:{output.width}:{output.height}"
        for output in sorted(outputs, key=lambda item: item.key)
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _match_output(
    source_output: OutputDescriptor,
    source: tuple[OutputDescriptor, ...],
    current: tuple[OutputDescriptor, ...],
) -> OutputDescriptor | None:
    matched = next((output for output in current if output.key == source_output.key), None)
    if matched is not None:
        return matched
    geometry_matches = [
        output
        for output in current
        if (output.x, output.y, output.width, output.height)
        == (source_output.x, source_output.y, source_output.width, source_output.height)
    ]
    if len(geometry_matches) == 1:
        return geometry_matches[0]
    source_size_matches = [
        output
        for output in source
        if (output.width, output.height) == (source_output.width, source_output.height)
    ]
    current_size_matches = [
        output
        for output in current
        if (output.width, output.height) == (source_output.width, source_output.height)
    ]
    if len(source_size_matches) == 1 and len(current_size_matches) == 1:
        return current_size_matches[0]
    return None


def _same_topology(
    source: tuple[OutputDescriptor, ...],
    current: tuple[OutputDescriptor, ...],
) -> bool:
    def key(item: OutputDescriptor) -> tuple[int, int, int, int]:
        return item.x, item.y, item.width, item.height

    return sorted(map(key, source)) == sorted(map(key, current))


def _bounds(outputs: tuple[OutputDescriptor, ...]) -> tuple[int, int, int, int]:
    left = min(output.x for output in outputs)
    top = min(output.y for output in outputs)
    right = max(output.x + output.width for output in outputs)
    bottom = max(output.y + output.height for output in outputs)
    return left, top, right - left, bottom - top


def _normalize(value: float, extent: int) -> float:
    return min(1.0, max(0.0, value / max(extent - 1.0, 1.0)))


def _denormalize(value: float, extent: int) -> float:
    return min(extent - 1.0, max(0.0, value * max(extent - 1.0, 1.0)))
