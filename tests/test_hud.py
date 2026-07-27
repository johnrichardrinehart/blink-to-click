"""Tests for the opt-in Wayland layer-shell debug HUD."""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field

from gazeebo.contracts import TrainingRegionStatus
from gazeebo.hud import LayerShellDebugHud


@dataclass(slots=True)
class FakeSurface:
    """Record HUD rendering without a graphical session."""

    updates: list[tuple[str, float, float]] = field(default_factory=list)
    closed: bool = False

    def show(self, region_id: str, x: float, y: float) -> None:
        """Record one rendered value."""
        self.updates.append((region_id, x, y))

    def close(self) -> None:
        """Record cleanup."""
        self.closed = True


class HudTests(unittest.TestCase):
    """Lock HUD content, rate limiting, and cleanup."""

    def test_content_lists_every_authorized_region(self) -> None:
        """Diagnostics retain the active region and full authorization set."""
        surface = FakeSurface()
        hud = LayerShellDebugHud(
            surface,
            clock=lambda: 10.0,
            authorized_regions=("display-a", "display-b", "display-c"),
        )
        hud.set_model_context("global+context-4", "weak", "inferred-weak")
        hud.set_noise_context("inferred-compatible", 0.2, 12.0)
        hud.set_evidence_context("head+face")
        hud.set_training_region(
            TrainingRegionStatus("display-c:2,2", 2.25, 1.25, 3.50, 17, 27, "high-surprise")
        )
        hud.set_refinement_context("3x4 depth 2 cell 'k' settling")
        asyncio.run(hud.update("display-b", 300.0, 400.0))
        assert surface.updates == [
            (
                "display-b; authorized: display-a, display-b, display-c; "
                "model: global+context-4; confidence: inferred-weak; topology: weak; "
                "evidence: head+face; training: display-c:2,2 high-surprise CVaR90 "
                "2.25 1.25..3.50 (17/27); refinement: 3x4 depth 2 cell 'k' "
                "settling; noise: inferred-compatible; alpha: 0.20; "
                "dead-zone: 12.0px",
                300.0,
                400.0,
            )
        ]

    def test_updates_at_most_once_per_second_and_removes_hud(self) -> None:
        """Rapid pointer updates render no more than once per second."""
        times = iter((10.0, 10.9, 11.0))
        surface = FakeSurface()
        hud = LayerShellDebugHud(surface, clock=lambda: next(times))

        async def exercise() -> None:
            await hud.update("display-a", 100.4, 200.6)
            await hud.update("display-b", 300.0, 400.0)
            await hud.update("display-b", 500.0, 600.0)
            await hud.close()
            await hud.close()

        asyncio.run(exercise())

        assert surface.updates == [
            ("display-a", 100.4, 200.6),
            ("display-b", 500.0, 600.0),
        ]
        assert surface.closed


if __name__ == "__main__":
    unittest.main()
