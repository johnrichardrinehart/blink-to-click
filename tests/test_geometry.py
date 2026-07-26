"""Tests for authorized-display calibration, clipping, and smoothing."""

from __future__ import annotations

import math
import unittest

from gazeebo.calibration import (
    CalibrationModel,
    CalibrationSample,
    IncrementalCalibration,
    aggregate_feature_dispersion,
    aggregate_features,
    grouped_folds,
)
from gazeebo.contracts import DisplayRegion
from gazeebo.geometry import (
    DisplayTopology,
    Point,
    PointerSmoother,
    PointerTarget,
    calibration_targets,
    rolling_point_median,
)


class GeometryTests(unittest.TestCase):
    """Lock full-desktop mapping without a graphical session."""

    def setUp(self) -> None:
        """Create one representative authorized display."""
        self.topology = DisplayTopology((DisplayRegion("selected", 3840, 360, 2560, 1440),))

    def test_prediction_point_median_rejects_stationary_cursor_outliers(self) -> None:
        """One extreme estimate cannot move a stable navigation fixation."""
        history: list[Point] = []
        filtered = Point(0.0, 0.0)
        for point in (
            Point(100.0, 100.0),
            Point(102.0, 99.0),
            Point(900.0, 600.0),
            Point(101.0, 101.0),
            Point(99.0, 100.0),
        ):
            filtered = rolling_point_median(history, point)
        assert filtered == Point(101.0, 100.0)
        for value in range(20):
            rolling_point_median(history, Point(float(value), float(value)))
        assert len(history) == 15

    def test_feature_dispersion_reduces_noisy_input_sensitivity(self) -> None:
        """High-rate input noise regularizes coefficients without storing frames."""
        quiet = tuple(
            CalibrationSample((value,), Point(500.0 + 500.0 * value, 0.0))
            for value in (-1.0, 0.0, 1.0)
        )
        noisy = tuple(
            CalibrationSample(
                sample.features,
                sample.target,
                feature_dispersion=(1.0,),
            )
            for sample in quiet
        )
        quiet_model = CalibrationModel.fit(quiet, ridge=0.0)
        noisy_model = CalibrationModel.fit(noisy, ridge=0.0)
        quiet_shift = quiet_model.predict((0.1,)).x - quiet_model.predict((0.0,)).x
        noisy_shift = noisy_model.predict((0.1,)).x - noisy_model.predict((0.0,)).x
        assert 0.0 < noisy_shift < quiet_shift
        assert aggregate_feature_dispersion(((0.0,), (1.0,), (2.0,))) == (1.4826,)

        full_samples = tuple(
            CalibrationSample(
                (value, *([0.0] * 14)),
                Point(500.0 + 500.0 * value, 0.0),
                feature_dispersion=(0.1,) * 15,
            )
            for value in (-1.0, 0.0, 1.0)
        )
        incremental = IncrementalCalibration(full_samples)
        updated = incremental.add(
            CalibrationSample(
                (0.5, *([0.0] * 14)),
                Point(750.0, 0.0),
                feature_dispersion=(0.1,) * 15,
            )
        )
        assert updated.sample_count == 4

    def test_grouped_folds_train_and_validate_every_target(self) -> None:
        """Every aggregate is held out once and trains every other compatible fold."""
        folds = grouped_folds(13)
        held_out = [index for _training, validation in folds for index in validation]
        assert sorted(held_out) == list(range(13))
        assert all(set(training).isdisjoint(validation) for training, validation in folds)
        assert all(len(training) + len(validation) == 13 for training, validation in folds)
        assert all(
            sum(index in training for training, _validation in folds) == len(folds) - 1
            for index in range(13)
        )

    def test_sample_weights_are_positive_and_every_target_influences_fit(self) -> None:
        """A noisy target may have low weight but cannot have zero influence."""
        with self.assertRaisesRegex(ValueError, "positive"):
            CalibrationSample((0.0,), Point(0.0, 0.0), 0.0)
        common = [
            CalibrationSample((float(index),), Point(float(index), float(index)))
            for index in range(7)
        ]
        low_weight = CalibrationSample((8.0,), Point(100.0, 100.0), 1e-3)
        without = CalibrationModel.fit(common)
        with_target = CalibrationModel.fit((*common, low_weight))
        prediction_without = without.predict((4.0,))
        prediction_with = with_target.predict((4.0,))
        assert with_target.sample_count == 8
        assert prediction_with != prediction_without

    def test_targets_cover_the_selected_display(self) -> None:
        """Calibration contributes center and corner samples to one display."""
        targets = calibration_targets(self.topology)
        assert len(targets) == 5
        assert {target.region_id for target in targets} == {"selected"}
        for target in targets:
            global_point = self.topology.to_global(target)
            located = self.topology.locate(global_point)
            self.assertAlmostEqual(located.x, target.x)
            self.assertAlmostEqual(located.y, target.y)

    def test_calibration_targets_cover_every_authorized_display(self) -> None:
        """Combined calibration contributes five anchors per display."""
        topology = DisplayTopology(
            (
                DisplayRegion("first", 0, 0, 1000, 700),
                DisplayRegion("second", 1000, 100, 800, 600),
            )
        )
        targets = calibration_targets(topology)
        assert len(targets) == 10
        assert [target.region_id for target in targets].count("first") == 5
        assert [target.region_id for target in targets].count("second") == 5

    def test_predictions_clip_to_selected_display_edges(self) -> None:
        """No fitted point can roam onto another output."""
        upper_left = self.topology.locate(Point(-100.0, -100.0))
        lower_right = self.topology.locate(Point(10000.0, 10000.0))
        assert upper_left == PointerTarget("selected", 0.0, 0.0)
        assert lower_right.region_id == "selected"
        assert 2559.0 < lower_right.x < 2560.0
        assert 1439.0 < lower_right.y < 1440.0

    def test_global_and_local_mapping_retains_selected_origin(self) -> None:
        """Global diagnostics retain an authorized display's compositor origin."""
        local = PointerTarget("selected", 100.0, 200.0)
        assert self.topology.to_global(local) == Point(3940.0, 560.0)
        assert self.topology.locate(Point(3940.0, 560.0)) == local

    def test_topology_identity_changes_with_selected_geometry(self) -> None:
        """Stale calibration detects movement or resizing of authorized displays."""
        changed = DisplayTopology((DisplayRegion("selected", 0, 0, 1920, 1080),))
        same = DisplayTopology(self.topology.regions)
        renamed = DisplayTopology((DisplayRegion("opaque-new-id", 3840, 360, 2560, 1440),))
        assert same.topology_id == self.topology.topology_id
        assert renamed.topology_id == self.topology.topology_id
        assert changed.topology_id != self.topology.topology_id

    def test_multiple_regions_allow_roaming_and_clip_union_gaps(self) -> None:
        """Predictions cross displays but cannot remain in unauthorized gaps."""
        topology = DisplayTopology(
            (
                DisplayRegion("first", 0, 0, 1000, 700),
                DisplayRegion("second", 1200, 100, 500, 500),
            )
        )
        assert topology.locate(Point(1400.0, 300.0)) == PointerTarget(
            "second",
            200.0,
            200.0,
        )
        gap = topology.locate(Point(1080.0, 300.0))
        assert gap.region_id == "first"
        assert 999.0 < gap.x < 1000.0
        assert topology.to_global(gap).x < 1000.0

    def test_duplicate_region_ids_are_rejected(self) -> None:
        """Pointer stream lookup stays unambiguous across displays."""
        with self.assertRaisesRegex(ValueError, "unique"):
            DisplayTopology(
                (
                    DisplayRegion("same", 0, 0, 1000, 700),
                    DisplayRegion("same", 1000, 0, 1000, 700),
                )
            )

    def test_target_features_use_robust_component_medians(self) -> None:
        """Frame jitter cannot attenuate calibration response within one target."""
        assert aggregate_features(((0.1, 0.8), (0.2, 0.7), (9.0, 0.6), (0.3, 0.5), (0.4, 0.4))) == (
            0.3,
            0.6,
        )

    def test_head_only_features_fit_without_pupil_evidence(self) -> None:
        """Head pose and face geometry remain sufficient when pupils are unavailable."""
        samples = []
        for index in range(10):
            pitch = index / 20.0
            yaw = (9 - index) / 20.0
            center_x = 0.2 + index / 20.0
            center_y = 0.3 + index / 30.0
            features = (
                0.5,
                0.5,
                0.5,
                0.5,
                pitch,
                yaw,
                center_x,
                center_y,
                0.5,
                0.5,
                0.0,
                0.0,
                0.0,
                0.3,
                0.4,
            )
            samples.append(
                CalibrationSample(
                    features,
                    Point(100.0 + 800.0 * center_x, 50.0 + 600.0 * center_y),
                )
            )
        model = CalibrationModel.fit(tuple(samples))
        prediction = model.predict(samples[4].features)
        restored = CalibrationModel.from_record(model.to_record())
        restored_prediction = restored.predict(samples[4].features)
        self.assertAlmostEqual(prediction.x, samples[4].target.x, delta=2.0)
        self.assertAlmostEqual(prediction.y, samples[4].target.y, delta=2.0)
        self.assertAlmostEqual(restored_prediction.x, prediction.x)
        self.assertAlmostEqual(restored_prediction.y, prediction.y)
        assert "head" in model.kind or model.kind == "affine"

    def test_current_context_dominates_without_discarding_history(self) -> None:
        """Conflicting historical postures stay positive without overwhelming this run."""
        current = [
            CalibrationSample((float(index),), Point(float(index), 0.0), 1.0, (0.0, 0.0))
            for index in range(3)
        ]
        historical = [
            CalibrationSample(
                (float(index % 3),),
                Point(1000.0 + float(index % 3), 0.0),
                1.0,
                (1.0, 1.0),
            )
            for index in range(30)
        ]
        model = CalibrationModel.fit(
            (*historical, *current),
            routing_contexts=((0.0, 0.0),),
        )
        prediction = model.predict((1.0,))
        assert model.sample_count == 33
        assert 0.0 < prediction.x < 200.0

    def test_per_output_models_reduce_piecewise_display_bias_and_round_trip(self) -> None:
        """Bounded output experts improve a mapping that one desktop affine cannot express."""
        topology = DisplayTopology(
            (
                DisplayRegion("left", 0, 0, 1000, 700),
                DisplayRegion("right", 1200, 0, 1000, 700),
            )
        )
        samples: list[CalibrationSample] = []
        for index in range(10):
            fraction = index / 9.0
            samples.append(
                CalibrationSample(
                    (-2.0 + fraction, fraction),
                    Point(100.0 + 800.0 * fraction, 100.0 + 500.0 * fraction),
                )
            )
            samples.append(
                CalibrationSample(
                    (1.0 + fraction, fraction),
                    Point(1300.0 + 800.0 * fraction, 100.0 + 500.0 * fraction),
                )
            )
        global_model = CalibrationModel.fit(samples)
        output_model = CalibrationModel.fit(samples, topology=topology)
        probes = (
            ((-1.75, 0.25), Point(300.0, 225.0)),
            ((1.75, 0.75), Point(1900.0, 475.0)),
        )
        global_error = sum(
            math.hypot(
                global_model.predict(features).x - target.x,
                global_model.predict(features).y - target.y,
            )
            for features, target in probes
        )
        output_error = sum(
            math.hypot(
                output_model.predict(features).x - target.x,
                output_model.predict(features).y - target.y,
            )
            for features, target in probes
        )
        restored = CalibrationModel.from_record(output_model.to_record())
        assert output_model.kind.startswith("output-mixture/")
        assert output_model.output_expert_count == 2
        assert output_model.sample_count == len(samples)
        assert output_error < global_error
        assert restored.predict(probes[0][0]) == output_model.predict(probes[0][0])
        point, uncertainty = restored.predict_with_uncertainty(probes[1][0])
        assert point == output_model.predict(probes[1][0])
        assert uncertainty is not None
        assert uncertainty > 0.0

    def test_per_output_global_blend_preserves_all_target_influence(self) -> None:
        """A target on another output still changes prediction through the global component."""
        topology = DisplayTopology(
            (
                DisplayRegion("left", 0, 0, 1000, 700),
                DisplayRegion("right", 1000, 0, 1000, 700),
            )
        )
        samples = tuple(
            CalibrationSample((float(index),), Point(float(index * 100), 200.0))
            for index in range(3)
        ) + tuple(
            CalibrationSample((float(index + 10),), Point(float(1200 + index * 100), 200.0))
            for index in range(3)
        )
        baseline = CalibrationModel.fit(samples, topology=topology)
        changed = CalibrationModel.fit(
            (*samples[:-1], CalibrationSample(samples[-1].features, Point(1900.0, 200.0), 1e-3)),
            topology=topology,
        )
        assert baseline.predict((1.0,)) != changed.predict((1.0,))

    def test_sparse_output_uses_global_fallback_without_cross_display_drag(self) -> None:
        """An output without three targets keeps the global estimate instead of a wrong expert."""
        topology = DisplayTopology(
            (
                DisplayRegion("left", 0, 0, 1000, 700),
                DisplayRegion("right", 1000, 0, 1000, 700),
            )
        )
        samples = (
            *(
                CalibrationSample((float(index),), Point(float(100 + index * 200), 200.0))
                for index in range(4)
            ),
            CalibrationSample((10.0,), Point(1200.0, 200.0)),
            CalibrationSample((12.0,), Point(1800.0, 200.0)),
        )
        global_model = CalibrationModel.fit(samples)
        output_model = CalibrationModel.fit(samples, topology=topology)
        global_prediction = global_model.predict((11.0,))
        output_prediction = output_model.predict((11.0,))
        assert output_model.output_expert_count == 1
        self.assertAlmostEqual(output_prediction.x, global_prediction.x, places=9)
        self.assertAlmostEqual(output_prediction.y, global_prediction.y, places=9)
        restored = CalibrationModel.from_record(output_model.to_record())
        assert restored.predict((11.0,)) == output_prediction

    def test_per_output_head_fallback_remains_local_without_pupils(self) -> None:
        """Missing pupil evidence routes through bounded head-only output experts."""
        topology = DisplayTopology(
            (
                DisplayRegion("left", 0, 0, 1000, 700),
                DisplayRegion("right", 1000, 0, 1000, 700),
            )
        )

        def head_features(value: float) -> tuple[float, ...]:
            features = [0.0] * 15
            features[6] = value
            features[10] = 0.0
            return tuple(features)

        samples = tuple(
            CalibrationSample(head_features(float(index)), Point(float(100 + 300 * index), 300.0))
            for index in range(3)
        ) + tuple(
            CalibrationSample(
                head_features(float(index + 10)),
                Point(float(1100 + 300 * index), 300.0),
            )
            for index in range(3)
        )
        model = CalibrationModel.fit_head(samples, topology=topology)
        assert model.output_expert_count == 2
        left = model.predict(head_features(1.0))
        right = model.predict(head_features(11.0))
        assert 0.0 <= left.x < 1000.0
        assert 1000.0 <= right.x < 2000.0

    def test_per_output_incremental_update_matches_batch_and_has_constant_work(self) -> None:
        """One target updates the global and one bounded output state without a corpus scan."""
        topology = DisplayTopology(
            (
                DisplayRegion("left", 0, 0, 1000, 700),
                DisplayRegion("right", 1000, 0, 1000, 700),
            )
        )
        initial = tuple(
            CalibrationSample((float(index),), Point(float(100 + index * 200), 200.0))
            for index in range(3)
        ) + tuple(
            CalibrationSample((float(index + 10),), Point(float(1100 + index * 200), 200.0))
            for index in range(3)
        )
        addition = CalibrationSample((3.0,), Point(700.0, 200.0))
        incremental = IncrementalCalibration(initial, topology=topology)
        before = incremental.work
        updated = incremental.add(addition)
        after = incremental.work
        batch = CalibrationModel.fit((*initial, addition), topology=topology)
        assert after.statistic_updates - before.statistic_updates == 4
        assert updated.sample_count == 7
        for features in ((1.5,), (11.5,)):
            actual = updated.predict(features)
            expected = batch.predict(features)
            self.assertAlmostEqual(actual.x, expected.x, places=9)
            self.assertAlmostEqual(actual.y, expected.y, places=9)
        incremental.add(CalibrationSample((13.0,), Point(1700.0, 200.0)))
        final = incremental.work
        assert final.statistic_updates - after.statistic_updates == 4
        assert final.score_predictions - after.score_predictions == (
            after.score_predictions - before.score_predictions
        )

    def test_affine_calibration_recovers_known_mapping(self) -> None:
        """Ridge fitting maps gaze features into global logical coordinates."""
        samples = []
        for first, second in (
            (-1.0, -1.0),
            (-1.0, 0.0),
            (-1.0, 1.0),
            (0.0, -1.0),
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, -1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ):
            samples.append(
                CalibrationSample(
                    (first, second),
                    Point(100.0 + 500.0 * first, 200.0 + 300.0 * second),
                )
            )
        model = CalibrationModel.fit(samples, ridge=1e-10)
        prediction = model.predict((0.25, -0.5))
        assert model.kind == "affine"
        self.assertAlmostEqual(prediction.x, 225.0, places=5)
        self.assertAlmostEqual(prediction.y, 50.0, places=5)

    def test_bounded_neural_features_capture_nonlinear_head_mapping(self) -> None:
        """A frozen hidden layer adds nonlinearity without sample-sized state."""
        samples: list[CalibrationSample] = []
        head_indices = (4, 5, 6, 7, 12, 13, 14)
        for index in range(80):
            features = [0.0] * 15
            for dimension, feature_index in enumerate(head_indices):
                features[feature_index] = math.sin((index + 1) * (dimension + 2) * 0.17)
            hidden = (
                sum(
                    features[feature_index] * math.sin((dimension + 1) * 1.61803398875) * 1.5
                    for dimension, feature_index in enumerate(head_indices)
                )
                + math.cos(0.73) * 0.5
            )
            response = math.tanh(hidden)
            samples.append(
                CalibrationSample(
                    tuple(features),
                    Point(3000.0 + 2000.0 * response, 1500.0 + 800.0 * response**2),
                )
            )
        model = CalibrationModel.fit(samples)
        restored = CalibrationModel.from_record(model.to_record())
        assert model.kind == "affine/neural-head+face"
        assert restored.kind == model.kind
        assert restored.predict(samples[20].features) == model.predict(samples[20].features)

    def test_incremental_selection_tracks_new_unseen_candidate_evidence(self) -> None:
        """A sparse-run winner cannot remain frozen after later targets favor another model."""
        head_indices = (4, 5, 6, 7, 12, 13, 14)

        def sample(index: int, *, nonlinear: bool) -> CalibrationSample:
            features = [0.0] * 15
            for dimension, feature_index in enumerate(head_indices):
                features[feature_index] = math.sin((index + 1) * (dimension + 2) * 0.17)
            if nonlinear:
                hidden = (
                    sum(
                        features[feature_index] * math.sin((dimension + 1) * 1.61803398875) * 1.5
                        for dimension, feature_index in enumerate(head_indices)
                    )
                    + math.cos(0.73) * 0.5
                )
                response = math.tanh(hidden)
                target = Point(
                    3000.0 + 2000.0 * response,
                    1500.0 + 800.0 * response**2,
                )
            else:
                target = Point(
                    3000.0 + 1000.0 * features[4],
                    1500.0 + 500.0 * features[5],
                )
            return CalibrationSample(tuple(features), target)

        provisional = IncrementalCalibration(
            [sample(index, nonlinear=False) for index in range(10)]
        )
        assert provisional.model.kind != "affine/neural-head+face"
        for index in range(10, 70):
            model = provisional.add(sample(index, nonlinear=True))
        assert model.kind == "affine/neural-head+face"

    def test_model_selection_uses_bounded_affine_candidates(self) -> None:
        """New fits avoid sample-sized kernels while preserving extrapolation."""
        samples = [
            CalibrationSample(
                (value,),
                Point(1000.0 + 800.0 * value**3, 500.0 + 400.0 * value**2),
            )
            for value in (-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        ]
        model = CalibrationModel.fit(samples)
        assert model.kind.startswith("affine")
        assert model.supports_extrapolation
        assert model.to_record()["support"] is None

    def test_incremental_update_matches_batch_bayesian_posterior(self) -> None:
        """Conjugate natural-parameter updates match one-pass fitting exactly."""
        initial = (
            CalibrationSample((-1.0,), Point(100.0, 200.0)),
            CalibrationSample((0.0,), Point(500.0, 350.0)),
            CalibrationSample((1.0,), Point(900.0, 500.0)),
        )
        additions = (
            CalibrationSample((-0.5,), Point(300.0, 275.0), 0.5),
            CalibrationSample((0.5,), Point(700.0, 425.0), 0.75),
        )
        incremental = IncrementalCalibration(initial)
        incremental_model = incremental.model
        for sample in additions:
            incremental_model = incremental.add(sample)
        batch_model = CalibrationModel.fit((*initial, *additions))
        for features in ((-0.75,), (0.25,), (0.9,)):
            incremental_point, incremental_uncertainty = incremental_model.predict_with_uncertainty(
                features
            )
            batch_point, batch_uncertainty = batch_model.predict_with_uncertainty(features)
            self.assertAlmostEqual(incremental_point.x, batch_point.x, places=9)
            self.assertAlmostEqual(incremental_point.y, batch_point.y, places=9)
            assert incremental_uncertainty is not None
            assert batch_uncertainty is not None
            self.assertAlmostEqual(incremental_uncertainty, batch_uncertainty, places=9)

    def test_bayesian_uncertainty_falls_with_repeated_evidence(self) -> None:
        """Compatible target evidence contracts posterior uncertainty locally."""
        sparse = (
            CalibrationSample((-1.0,), Point(100.0, 200.0)),
            CalibrationSample((0.0,), Point(500.0, 350.0)),
            CalibrationSample((1.0,), Point(900.0, 500.0)),
        )
        dense = tuple(sample for _repeat in range(10) for sample in sparse)
        sparse_model = CalibrationModel.fit(sparse)
        dense_model = CalibrationModel.fit(dense)
        _sparse_point, sparse_uncertainty = sparse_model.predict_with_uncertainty((0.0,))
        _dense_point, dense_uncertainty = dense_model.predict_with_uncertainty((0.0,))
        assert sparse_uncertainty is not None
        assert dense_uncertainty is not None
        assert sparse_uncertainty > 0.0
        assert dense_uncertainty < sparse_uncertainty

    def test_bayesian_posterior_round_trip_preserves_uncertainty(self) -> None:
        """Accepted anchors retain posterior mean and uncertainty after restart."""
        samples = tuple(
            CalibrationSample(
                (float(index),),
                Point(float(index * 100), float(index * 50 + index % 2)),
            )
            for index in range(8)
        )
        model = CalibrationModel.fit(samples)
        restored = CalibrationModel.from_record(model.to_record())
        expected_point, expected_uncertainty = model.predict_with_uncertainty((3.5,))
        actual_point, actual_uncertainty = restored.predict_with_uncertainty((3.5,))
        assert actual_point == expected_point
        assert expected_uncertainty is not None
        assert actual_uncertainty is not None
        self.assertAlmostEqual(actual_uncertainty, expected_uncertainty, places=12)

    def test_grouped_selection_work_scales_linearly(self) -> None:
        """Doubling a large corpus doubles deterministic fit work, not quadruples it."""

        def samples(count: int) -> tuple[CalibrationSample, ...]:
            return tuple(
                CalibrationSample(
                    (float(index % 101) / 100.0,),
                    Point(float(index % 1000), float((index * 7) % 700)),
                )
                for index in range(count)
            )

        _first_model, first = CalibrationModel.fit_with_work(samples(10_000))
        _second_model, second = CalibrationModel.fit_with_work(samples(20_000))
        assert second.statistic_updates == first.statistic_updates * 2
        assert second.score_predictions == first.score_predictions * 2
        assert second.total == first.total * 2

    def test_per_output_grouped_selection_work_remains_linear(self) -> None:
        """A fixed output bound adds a constant pass rather than target-pair work."""
        topology = DisplayTopology(
            (
                DisplayRegion("left", 0, 0, 1000, 700),
                DisplayRegion("right", 1000, 0, 1000, 700),
            )
        )

        def samples(count: int) -> tuple[CalibrationSample, ...]:
            return tuple(
                CalibrationSample(
                    (float(index % 101) / 100.0,),
                    Point(
                        float(200 + 1000 * (index % 2)),
                        float((index * 7) % 700),
                    ),
                )
                for index in range(count)
            )

        _first_model, first = CalibrationModel.fit_with_work(
            samples(10_000),
            topology=topology,
        )
        _second_model, second = CalibrationModel.fit_with_work(
            samples(20_000),
            topology=topology,
        )
        assert second.statistic_updates == first.statistic_updates * 2
        assert second.score_predictions == first.score_predictions * 2
        assert second.total == first.total * 2

    def test_incremental_target_update_has_constant_work(self) -> None:
        """One new target updates fixed-dimensional state without scanning history."""
        initial = tuple(
            CalibrationSample((float(index),), Point(float(index), float(index)))
            for index in range(8)
        )
        provisional = IncrementalCalibration(initial)
        before = provisional.work
        first = provisional.add(
            CalibrationSample((8.0,), Point(8.0, 8.0)),
        )
        after_first = provisional.work
        second = provisional.add(
            CalibrationSample((9.0,), Point(9.0, 9.0)),
        )
        after_second = provisional.work
        assert after_first.statistic_updates - before.statistic_updates == 2
        assert after_second.statistic_updates - after_first.statistic_updates == 2
        assert first.sample_count == 9
        assert second.sample_count == 10

    def test_legacy_rbf_record_remains_readable_without_new_pairwise_fits(self) -> None:
        """Existing private anchors load even though new fits are bounded affine."""
        record: dict[str, object] = {
            "kind": "rbf",
            "coefficients": [[-10.0, 5.0], [10.0, -5.0]],
            "feature_mean": [0.5],
            "feature_scale": [0.5],
            "support": [[-1.0], [1.0]],
            "gamma": 0.5,
            "target_offset": [100.0, 50.0],
            "input_feature_count": 1,
            "feature_indices": [0],
            "feature_name": "all",
            "sample_count": 2,
            "head_fallback": None,
        }
        restored = CalibrationModel.from_record(record)
        prediction = restored.predict((0.5,))
        assert restored.kind == "rbf"
        assert prediction == Point(100.0, 50.0)

    def test_smoothing_suppresses_jitter_and_caps_repositioning(self) -> None:
        """Small idle changes stop while large gaze changes pan in bounded steps."""
        smoother = PointerSmoother(alpha=0.5, dead_zone=5.0, maximum_step=100.0)
        assert smoother.update(Point(10.0, 10.0)) == Point(10.0, 10.0)
        assert smoother.update(Point(13.0, 13.0)) == Point(10.0, 10.0)
        assert smoother.update(Point(1010.0, 10.0)) == Point(110.0, 10.0)
        assert smoother.update(Point(210.0, 10.0)) == Point(160.0, 10.0)
        smoother.reset()
        assert smoother.update(Point(-50.0, 20.0)) == Point(-50.0, 20.0)


if __name__ == "__main__":
    unittest.main()
