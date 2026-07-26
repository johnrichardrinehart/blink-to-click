"""Tests for reusing target labels across portal display topologies."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

from gazeebo.adaptation import (
    TopologyQuality,
    describe_topology,
    make_stored_target,
    map_model_point,
    map_stored_target,
    model_mapping_supported,
    output_mapping_quality,
)
from gazeebo.contracts import DisplayRegion
from gazeebo.geometry import DisplayTopology, Point, PointerTarget

if TYPE_CHECKING:
    from gazeebo.state import StoredTarget


def stored(
    topology: DisplayTopology,
    target: PointerTarget,
) -> StoredTarget:
    """Create one normalized persistent target."""
    return make_stored_target(
        0,
        "camera-a",
        "gaze-v1",
        (0.1, 0.2),
        (0.0, 0.1),
        topology,
        target,
        "center",
    )


class TopologyAdaptationTests(unittest.TestCase):
    """Lock exact, strong, weak, added, and removed-output behavior."""

    def test_exact_topology_retains_global_target(self) -> None:
        """An unchanged portal layout has measured-quality correspondence."""
        topology = DisplayTopology((DisplayRegion("stable", 100, 200, 1000, 700),))
        sample = stored(topology, PointerTarget("stable", 500.0, 350.0))
        mapped = map_stored_target(sample, topology)
        assert mapped is not None
        assert mapped.quality is TopologyQuality.EXACT
        assert abs(mapped.point.x - 600.0) < 1e-6
        assert abs(mapped.point.y - 550.0) < 1e-6

    def test_opaque_key_change_with_same_geometry_remains_exact(self) -> None:
        """Session-local portal IDs do not invalidate unchanged logical geometry."""
        source = DisplayTopology((DisplayRegion("old-stream", 100, 200, 1000, 700),))
        sample = stored(source, PointerTarget("old-stream", 500.0, 350.0))
        current = DisplayTopology((DisplayRegion("new-stream", 100, 200, 1000, 700),))
        mapped = map_stored_target(sample, current)
        assert mapped is not None
        assert mapped.quality is TopologyQuality.EXACT
        assert mapped.point == Point(600.0, 550.0)

    def test_model_mapping_preserves_reach_on_an_unchanged_output(self) -> None:
        """Moving another output cannot compress a validated display's range."""
        source = DisplayTopology(
            (
                DisplayRegion("old-left", 0, 0, 3840, 2160),
                DisplayRegion("old-laptop", 3840, 1800, 2256, 1504),
            )
        )
        current = DisplayTopology(
            (
                DisplayRegion("new-left", 0, 0, 3840, 2160),
                DisplayRegion("new-laptop", 4100, 1800, 2256, 1504),
            )
        )
        outputs = describe_topology(source)
        assert output_mapping_quality(outputs, current) is TopologyQuality.STRONG
        assert map_model_point(Point(0.0, 1000.0), outputs, current) == Point(0.0, 1000.0)
        assert map_model_point(Point(4000.0, 2000.0), outputs, current).x == 4260.0

    def test_resolution_and_position_change_remap_output_relative_label(self) -> None:
        """A stable output key survives logical movement and scaling."""
        source = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 500),))
        sample = stored(source, PointerTarget("stable", 750.0, 125.0))
        current = DisplayTopology((DisplayRegion("stable", 2000, 300, 2000, 1000),))
        mapped = map_stored_target(sample, current)
        assert mapped is not None
        assert mapped.quality is TopologyQuality.STRONG
        assert 3499.0 < mapped.point.x < 3502.0
        assert 549.0 < mapped.point.y < 551.0

    def test_unique_size_matches_when_portal_key_changes(self) -> None:
        """A unique geometry can retain an output despite an unstable stream ID."""
        source = DisplayTopology(
            (
                DisplayRegion("old-a", 0, 0, 800, 600),
                DisplayRegion("old-b", 800, 0, 1000, 700),
            )
        )
        sample = stored(source, PointerTarget("old-b", 500.0, 350.0))
        current = DisplayTopology(
            (
                DisplayRegion("new-a", 0, 100, 800, 600),
                DisplayRegion("new-b", 800, 0, 1000, 700),
            )
        )
        mapped = map_stored_target(sample, current)
        assert mapped is not None
        assert mapped.quality is TopologyQuality.STRONG
        assert mapped.point.x == 1300.0

    def test_removed_output_keeps_model_reach_on_surviving_geometry(self) -> None:
        """Removing a peer cannot compress a validated surviving output's range."""
        source = DisplayTopology(
            (
                DisplayRegion("old-left", 0, 0, 1000, 700),
                DisplayRegion("old-lower", 1000, 700, 800, 600),
            )
        )
        current = DisplayTopology((DisplayRegion("new-left", 0, 0, 1000, 700),))
        outputs = describe_topology(source)
        assert output_mapping_quality(outputs, current) is TopologyQuality.WEAK
        assert model_mapping_supported(outputs, current)
        assert map_model_point(Point(999.0, 699.0), outputs, current) == Point(999.0, 699.0)

    def test_removed_output_samples_are_excluded(self) -> None:
        """Targets belonging to an absent output do not train the current model."""
        source = DisplayTopology(
            (
                DisplayRegion("left", 0, 0, 800, 600),
                DisplayRegion("right", 800, 0, 1000, 700),
            )
        )
        sample = stored(source, PointerTarget("right", 500.0, 350.0))
        current = DisplayTopology((DisplayRegion("left", 0, 0, 800, 600),))
        assert map_stored_target(sample, current) is None

    def test_added_output_keeps_existing_sample_but_marks_topology_weak(self) -> None:
        """Existing evidence remains useful without claiming added-output precision."""
        source = DisplayTopology((DisplayRegion("left", 0, 0, 800, 600),))
        sample = stored(source, PointerTarget("left", 400.0, 300.0))
        current = DisplayTopology(
            (
                DisplayRegion("left", 0, 0, 800, 600),
                DisplayRegion("new", 800, 0, 1000, 700),
            )
        )
        mapped = map_stored_target(sample, current)
        assert mapped is not None
        assert mapped.quality is TopologyQuality.WEAK
        assert mapped.point.x == 400.0

    def test_ambiguous_output_uses_bounds_fallback_and_union_projection(self) -> None:
        """Duplicate geometry cannot force a guessed identity or a gap target."""
        source = DisplayTopology(
            (
                DisplayRegion("old-left", 0, 0, 800, 600),
                DisplayRegion("old-right", 1000, 0, 800, 600),
            )
        )
        sample = stored(source, PointerTarget("old-left", 799.0, 300.0))
        current = DisplayTopology(
            (
                DisplayRegion("new-left", 0, 100, 800, 600),
                DisplayRegion("new-right", 1200, 100, 800, 600),
            )
        )
        mapped = map_stored_target(sample, current)
        assert mapped is not None
        assert mapped.quality is TopologyQuality.WEAK
        assert current.region_containing(mapped.point) is not None


if __name__ == "__main__":
    unittest.main()
