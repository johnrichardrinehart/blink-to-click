"""Tests for bounded region-aware adaptive target scheduling."""

from __future__ import annotations

import unittest
from dataclasses import replace

from gazeebo.adaptation import make_stored_target
from gazeebo.contracts import DisplayRegion
from gazeebo.geometry import DisplayTopology, PointerTarget
from gazeebo.state import CursorNoiseSummary
from gazeebo.surprise import RegionKey, RegionSurpriseScheduler


class RegionSurpriseTests(unittest.TestCase):
    """Lock seeding, dominance, equalization, persistence, and bounded work."""

    def setUp(self) -> None:
        """Create two differently sized authorized outputs."""
        self.topology = DisplayTopology(
            (
                DisplayRegion("left", 0, 0, 900, 600),
                DisplayRegion("right", 900, 120, 1200, 900),
            )
        )
        self.diameters = {"left": 30.0, "right": 40.0}

    def test_seeding_visits_every_output_relative_cell_deterministically(self) -> None:
        """A 3x3 phase crosses outputs before advancing to the next cell."""
        scheduler = RegionSurpriseScheduler(self.topology, 100.0)
        selected = []
        for _ in range(18):
            choice = scheduler.select(self.diameters)
            selected.append(choice)
            assert choice.mode == "seeding"
            scheduler.mark_seed(choice.target)

        assert [item.key for item in selected[:4]] == [
            RegionKey("left", 1, 1),
            RegionKey("right", 1, 1),
            RegionKey("left", 0, 0),
            RegionKey("right", 0, 0),
        ]
        assert {item.key for item in selected} == {
            RegionKey(output, row, column)
            for output in ("left", "right")
            for row in range(3)
            for column in range(3)
        }
        assert scheduler.seeded_regions == scheduler.total_regions == 18
        for item in selected:
            assert scheduler.region_for_target(item.target) == item.key

    def test_cvar90_is_the_interpolated_mean_of_the_worst_ten_percent(self) -> None:
        """The weighted tail includes exactly the mass above the p90 boundary."""
        topology = DisplayTopology((DisplayRegion("only", 0, 0, 900, 600),))
        scheduler = RegionSurpriseScheduler(topology, 100.0, decay=1.0)
        key = RegionKey("only", 1, 1)
        target = PointerTarget("only", 450.0, 300.0)
        for error in range(100, 2001, 100):
            scheduler.observe(target, float(error), 0.0, 0.0)
        estimate = scheduler.estimate(key)
        assert estimate.cvar90 == 19.5
        assert estimate.tail_variance == 0.25
        assert estimate.effective_tail_count == 2.0
        assert estimate.mean == estimate.cvar90

    def test_cvar90_tracks_high_variance_and_consistent_bias_without_raw_max_lock(self) -> None:
        """Upper-tail cost catches spread and bias while an old maximum decays."""
        topology = DisplayTopology((DisplayRegion("only", 0, 0, 900, 600),))
        variable = RegionSurpriseScheduler(topology, 100.0, decay=1.0)
        biased = RegionSurpriseScheduler(topology, 100.0, decay=1.0)
        target = PointerTarget("only", 450.0, 300.0)
        for error in (0.0, 400.0) * 10:
            variable.observe(target, error, 0.0, 0.0)
        for _ in range(20):
            biased.observe(target, 200.0, 0.0, 0.0)
        assert variable.estimate(RegionKey("only", 1, 1)).cvar90 == 4.0
        assert biased.estimate(RegionKey("only", 1, 1)).cvar90 == 2.0

        decayed = RegionSurpriseScheduler(topology, 100.0, decay=0.5)
        decayed.observe(target, 1000.0, 0.0, 0.0)
        initial = decayed.estimate(RegionKey("only", 1, 1)).cvar90
        for _ in range(20):
            decayed.observe(target, 100.0, 0.0, 0.0)
        assert initial == 10.0
        assert decayed.estimate(RegionKey("only", 1, 1)).cvar90 < 1.01

    def test_derived_overflow_clips_at_the_explicit_normalized_range(self) -> None:
        """Derived surprise stays finite at its explicit normalized range."""
        topology = DisplayTopology((DisplayRegion("only", 0, 0, 900, 600),))
        scheduler = RegionSurpriseScheduler(topology, 100.0)
        estimate = scheduler.observe(
            PointerTarget("only", 450.0, 300.0),
            20_000.0,
            20_000.0,
            20_000.0,
        )
        assert estimate.cvar90 == scheduler.maximum_surprise
        assert estimate.mean == scheduler.maximum_surprise
        assert 0.0 <= estimate.lower <= estimate.upper <= scheduler.maximum_surprise

    def test_equalized_regions_are_not_precise_until_every_tail_bound_passes(self) -> None:
        """Making all cells equally bad cannot satisfy the success gate."""
        bad = RegionSurpriseScheduler(self.topology, 100.0, decay=1.0)
        good = RegionSurpriseScheduler(self.topology, 100.0, decay=1.0)
        for output in self.topology.regions:
            for row in range(3):
                for column in range(3):
                    target = self._center(RegionKey(output.region_id, row, column))
                    for _ in range(100):
                        bad.observe(target, 200.0, 0.0, 0.0)
                        good.observe(target, 50.0, 0.0, 0.0)
        assert bad.equalized
        assert not bad.regions_precise
        assert good.equalized
        assert good.regions_precise

    def test_high_surprise_regions_block_statistically_lower_regions(self) -> None:
        """Disjoint confidence intervals always select the highest region first."""
        scheduler = RegionSurpriseScheduler(self.topology, 100.0)
        first_key = RegionKey("left", 0, 0)
        high_key = RegionKey("right", 2, 2)
        for output in self.topology.regions:
            for row in range(3):
                for column in range(3):
                    key = RegionKey(output.region_id, row, column)
                    target = self._center(key)
                    error = 900.0 if key == high_key else 40.0
                    scheduler.observe(target, error, 0.0, 0.0)
                    scheduler.observe(target, error, 0.0, 0.0)
        low_before = scheduler.estimate(first_key)
        high_before = scheduler.estimate(high_key)
        assert high_before.lower > low_before.upper

        choice = scheduler.select(self.diameters)
        assert choice.mode == "high-surprise"
        assert choice.key == high_key

    def test_posterior_uncertainty_raises_expected_surprise(self) -> None:
        """A poorly known cell is measured before statistically lower cells."""
        scheduler = RegionSurpriseScheduler(self.topology, 100.0)
        high_key = RegionKey("right", 0, 2)
        for output in self.topology.regions:
            for row in range(3):
                for column in range(3):
                    key = RegionKey(output.region_id, row, column)
                    target = self._center(key)
                    uncertainty = 500.0 if key == high_key else 0.0
                    scheduler.observe(target, 100.0, uncertainty, 0.0)
                    scheduler.observe(target, 100.0, uncertainty, 0.0)
        high = scheduler.estimate(high_key)
        low = scheduler.estimate(RegionKey("left", 1, 1))
        assert high.mean == 6.0
        assert high.lower > low.upper
        assert scheduler.select(self.diameters).key == high_key

    def test_overlapping_intervals_balance_tied_regions_without_starvation(self) -> None:
        """Equalized regions return to least-observed deterministic exploration."""
        scheduler = RegionSurpriseScheduler(self.topology, 100.0)
        for output in self.topology.regions:
            for row in range(3):
                for column in range(3):
                    target = self._center(RegionKey(output.region_id, row, column))
                    scheduler.observe(target, 100.0, 20.0, 10.0)
        assert scheduler.equalized

        first = scheduler.select(self.diameters)
        assert first.mode == "balanced-equalized"
        assert first.key == RegionKey("left", 1, 1)
        scheduler.observe(first.target, 100.0, 20.0, 10.0)
        second = scheduler.select(self.diameters)
        assert second.key == RegionKey("right", 1, 1)

    def test_unobserved_seeded_regions_receive_explicit_uncertainty_exploration(self) -> None:
        """A seed visit without unseen error cannot masquerade as low surprise."""
        scheduler = RegionSurpriseScheduler(self.topology, 100.0)
        keys = [
            RegionKey(output.region_id, row, column)
            for row in range(3)
            for column in range(3)
            for output in self.topology.regions
        ]
        for key in keys:
            scheduler.mark_seed(self._center(key))
        for key in keys[1:]:
            scheduler.observe(self._center(key), 50.0, 0.0, 0.0, mark_visit=False)

        choice = scheduler.select(self.diameters)
        assert choice.mode == "uncertainty-exploration"
        assert choice.key == keys[0]

    def test_low_discrepancy_positions_are_unseen_and_updates_have_constant_work(self) -> None:
        """Repeated selection changes position while each update scans no history."""
        scheduler = RegionSurpriseScheduler(
            DisplayTopology((DisplayRegion("only", 0, 0, 900, 600),)),
            100.0,
        )
        diameters = {"only": 30.0}
        positions: set[tuple[float, float]] = set()
        for _ in range(9):
            choice = scheduler.select(diameters)
            positions.add((choice.target.x, choice.target.y))
            scheduler.observe(choice.target, 100.0, 0.0, 0.0)
        for _ in range(30):
            choice = scheduler.select(diameters)
            positions.add((choice.target.x, choice.target.y))
            scheduler.observe(choice.target, 100.0, 0.0, 0.0)
        assert len(positions) == 39
        assert scheduler.work.updates == 39
        assert scheduler.work.selections == 39
        assert scheduler.work.region_scans == 39 * 9
        assert scheduler.work.target_probes == 39
        assert scheduler.work.tail_bin_updates == 39 * scheduler.histogram_bins
        assert scheduler.work.tail_bin_scans > 0

    def test_stored_surprise_rebuilds_across_safe_topology_remapping(self) -> None:
        """Restart reconstruction preserves cells and scales pixel evidence."""
        source = DisplayTopology((DisplayRegion("old", 0, 0, 900, 600),))
        target = make_stored_target(
            0,
            "camera",
            "schema",
            (0.1, 0.2),
            (0.0, 0.5),
            source,
            PointerTarget("old", 750.0, 500.0),
            "corner",
            CursorNoiseSummary(10, 1.0, 1.0, 0.0, 1.0, 2.0),
            unseen_error=200.0,
            predictive_uncertainty=20.0,
        )
        current = DisplayTopology((DisplayRegion("new", 100, 50, 1800, 1200),))
        scheduler = RegionSurpriseScheduler.from_stored_targets(
            current,
            100.0,
            (target,),
            camera_id="camera",
            feature_schema="schema",
        )
        estimate = scheduler.estimate(RegionKey("new", 2, 2))
        assert estimate.visits == 1
        assert estimate.observations == 1
        assert estimate.cvar90 == 4.0
        assert estimate.mean == 4.4

        legacy = replace(target, unseen_error=None, predictive_uncertainty=None)
        migrated = RegionSurpriseScheduler.from_stored_targets(
            current,
            100.0,
            (legacy,),
            camera_id="camera",
            feature_schema="schema",
        )
        legacy_estimate = migrated.estimate(RegionKey("new", 2, 2))
        assert legacy_estimate.visits == 1
        assert legacy_estimate.observations == 0

    def test_scheduler_reconstruction_work_scales_linearly_to_twenty_thousand(self) -> None:
        """Doubling retained surprise evidence exactly doubles bounded updates."""
        source = DisplayTopology((DisplayRegion("only", 0, 0, 900, 600),))
        template = make_stored_target(
            0,
            "camera",
            "schema",
            (0.1, 0.2),
            (0.0, 0.5),
            source,
            PointerTarget("only", 450.0, 300.0),
            "center",
            unseen_error=100.0,
            predictive_uncertainty=10.0,
        )
        first = tuple(replace(template, sequence=index) for index in range(10_000))
        second = tuple(replace(template, sequence=index) for index in range(20_000))
        first_work = RegionSurpriseScheduler.from_stored_targets(
            source,
            100.0,
            first,
            camera_id="camera",
            feature_schema="schema",
        ).work
        second_work = RegionSurpriseScheduler.from_stored_targets(
            source,
            100.0,
            second,
            camera_id="camera",
            feature_schema="schema",
        ).work
        assert first_work.updates == 10_000
        assert second_work.updates == 20_000
        assert second_work.region_scans == first_work.region_scans == 0
        assert second_work.tail_bin_updates == 2 * first_work.tail_bin_updates
        assert second_work.tail_bin_scans == 2 * first_work.tail_bin_scans
        assert first_work.tail_bin_updates // first_work.updates == 1024

    def test_persisted_partial_seeding_resumes_at_the_same_next_region(self) -> None:
        """An interrupted invocation reconstructs its exact deterministic frontier."""
        uninterrupted = RegionSurpriseScheduler(self.topology, 100.0)
        stored = []
        for sequence in range(5):
            choice = uninterrupted.select(self.diameters)
            uninterrupted.observe(choice.target, 100.0, 10.0, 5.0)
            stored.append(
                make_stored_target(
                    sequence,
                    "camera",
                    "schema",
                    (0.1, 0.2),
                    (0.0, 0.5),
                    self.topology,
                    choice.target,
                    "center",
                    unseen_error=100.0,
                    predictive_uncertainty=10.0,
                )
            )
        expected = uninterrupted.select(self.diameters)
        resumed = RegionSurpriseScheduler.from_stored_targets(
            self.topology,
            100.0,
            stored,
            camera_id="camera",
            feature_schema="schema",
        ).select(self.diameters)
        assert resumed.key == expected.key
        assert resumed.target == expected.target
        assert resumed.mode == expected.mode == "seeding"

    def test_noise_and_uncertainty_widen_but_never_remove_evidence(self) -> None:
        """Optional quality evidence enlarges finite intervals without gating."""
        target = self._center(RegionKey("left", 1, 1))
        quiet = RegionSurpriseScheduler(self.topology, 100.0)
        noisy = RegionSurpriseScheduler(self.topology, 100.0)
        quiet_estimate = quiet.observe(target, 100.0, 0.0, 0.0)
        noisy_estimate = noisy.observe(target, 100.0, 80.0, 200.0)
        assert noisy_estimate.upper > quiet_estimate.upper
        assert noisy_estimate.observations == quiet_estimate.observations == 1
        assert quiet_estimate.mean == 1.0
        assert noisy_estimate.mean == 1.8

    def _center(self, key: RegionKey) -> PointerTarget:
        region = self.topology.region(key.output_key)
        return PointerTarget(
            key.output_key,
            (key.column + 0.5) * region.width / 3.0,
            (key.row + 0.5) * region.height / 3.0,
        )


if __name__ == "__main__":
    unittest.main()
