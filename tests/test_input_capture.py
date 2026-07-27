"""Tests for optional portal InputCapture geometry and capability handling."""

from __future__ import annotations

import unittest

from dbus_next.signature import Variant

from gazeebo.input_capture import InputCaptureError, _parse_zones, _zone_barriers


class InputCaptureGeometryTests(unittest.TestCase):
    """Keep barrier creation deterministic and reject malformed portal data."""

    def test_adjacent_zones_propose_only_numbered_union_boundary_barriers(self) -> None:
        """Shared edges are removed while every exposed outer edge remains."""
        barriers = _zone_barriers(((100, 50, -100, 20), (80, 40, 0, 20)))
        assert len(barriers) == 7
        assert barriers[0]["barrier_id"].value == 1
        assert barriers[-1]["barrier_id"].value == 7
        assert barriers[0]["position"].value == [-100, 20, -1, 20]
        assert barriers[1]["position"].value == [-100, 70, -1, 70]
        partial = _zone_barriers(((100, 50, 0, 0), (100, 25, 100, 25)))
        assert [100, 0, 100, 24] in [item["position"].value for item in partial]

    def test_zones_are_finite_positive_integer_geometry(self) -> None:
        """The portal zone parser retains exact offsets and sizes."""
        assert _parse_zones(Variant("a(uuii)", [[100, 50, -10, 5]])) == ((100, 50, -10, 5),)
        with self.assertRaises(InputCaptureError):
            _parse_zones([[0, 50, 0, 0]])
        with self.assertRaises(InputCaptureError):
            _parse_zones([[100, 50, 0]])
