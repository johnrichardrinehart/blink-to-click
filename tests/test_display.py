"""Tests for generic live output monitoring and topology policy."""

from __future__ import annotations

import unittest

from gazeebo.display import DisplayMonitorError, parse_output_snapshot
from gazeebo.runtime import DISPLAY_REAUTHORIZATION_RESULT, _display_change_action


class DisplayTests(unittest.TestCase):
    """Lock canonical geometry parsing and authorization-safe change policy."""

    def test_snapshot_parser_canonicalizes_live_logical_geometry(self) -> None:
        """Registry order cannot masquerade as a topology change."""
        assert parse_output_snapshot("1920:0:1280:720;0:0:1920:1080") == (
            (0, 0, 1920, 1080),
            (1920, 0, 1280, 720),
        )
        with self.assertRaisesRegex(DisplayMonitorError, "duplicate"):
            parse_output_snapshot("0:0:100:100;0:0:100:100")
        with self.assertRaisesRegex(DisplayMonitorError, "dimensions"):
            parse_output_snapshot("0:0:0:100")

    def test_added_output_obeys_configured_pause_policy(self) -> None:
        """New displays pause by default but never expand authorization silently."""
        original = ((0, 0, 1920, 1080),)
        added = (*original, (1920, 0, 1280, 720))
        action, detail = _display_change_action(original, added, original, allow_pause=True)
        assert action == DISPLAY_REAUTHORIZATION_RESULT
        assert "pausing" in detail
        action, detail = _display_change_action(original, added, original, allow_pause=False)
        assert action is None
        assert "existing authorized union" in detail

    def test_removed_authorized_output_stops_without_pause(self) -> None:
        """The no-pause setting cannot retain geometry that no longer exists."""
        original = ((0, 0, 1920, 1080), (1920, 0, 1280, 720))
        current = (original[0],)
        action, _detail = _display_change_action(
            original,
            current,
            original,
            allow_pause=True,
        )
        assert action == DISPLAY_REAUTHORIZATION_RESULT
        action, detail = _display_change_action(
            original,
            current,
            original,
            allow_pause=False,
        )
        assert action == 3
        assert "stopping motion" in detail


if __name__ == "__main__":
    unittest.main()
