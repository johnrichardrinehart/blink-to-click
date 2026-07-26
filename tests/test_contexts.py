"""Tests for implicit context clustering and model routing."""

from __future__ import annotations

import unittest
from dataclasses import replace

from gazeebo.adaptation import (
    TopologyQuality,
    legacy_topology_id,
    topology_id_for_outputs,
)
from gazeebo.calibration import CalibrationModel, CalibrationSample
from gazeebo.contexts import (
    ContextConfig,
    ContextExpert,
    ModelRouter,
    SmoothingBounds,
    ValidationMetrics,
    add_target,
    build_router,
    candidate_is_acceptable,
    noise_smoothing_for,
)
from gazeebo.contracts import DisplayRegion
from gazeebo.geometry import DisplayTopology, Point
from gazeebo.state import (
    ContextCluster,
    CursorNoiseSummary,
    ModelAnchor,
    OutputDescriptor,
    StoredTarget,
    TrainingState,
    ValidationSummary,
)


def target(
    sequence: int,
    context: tuple[float, ...],
    *,
    output: str = "display",
    zone: str = "center",
) -> StoredTarget:
    """Create one target for clustering and retention tests."""
    return StoredTarget(
        sequence=sequence,
        camera_id="camera-a",
        feature_schema="gaze-v1",
        features=(sequence / 10.0, context[0]),
        context=context,
        outputs=(OutputDescriptor(output, 0, 0, 1000, 700),),
        output_key=output,
        target_u=0.5,
        target_v=0.5,
        desktop_u=0.5,
        desktop_v=0.5,
        zone=zone,
    )


def model(offset: float) -> CalibrationModel:
    """Fit a deterministic one-dimensional affine estimator."""
    return CalibrationModel.fit(
        (
            CalibrationSample((0.0,), Point(offset, 0.0)),
            CalibrationSample((0.5,), Point(offset + 50.0, 0.0)),
            CalibrationSample((1.0,), Point(offset + 100.0, 0.0)),
        )
    )


def cluster(name: str, center: float, error: float = 50.0) -> ContextCluster:
    """Create one validated routing context."""
    return ContextCluster(
        name,
        "camera-a",
        "gaze-v1",
        (center, 0.5),
        (0.01, 0.01),
        3,
        (0, 1, 2),
        median_error=error,
        edge_error=error,
    )


class ContextTests(unittest.TestCase):
    """Lock automatic clusters, all-target retention, routing, and acceptance."""

    def test_online_assignment_groups_near_contexts_and_splits_far_contexts(self) -> None:
        """Posture and illumination neighborhoods emerge without profile names."""
        state = TrainingState()
        config = ContextConfig(variance_floor=0.01)
        first = add_target(state, target(0, (0.0, 0.5)), config)
        nearby = add_target(state, target(1, (0.05, 0.52)), config)
        distant = add_target(state, target(2, (1.0, 0.1)), config)
        assert first == nearby
        assert distant != first
        assert len(state.clusters) == 2
        assert sorted(len(item.target_sequences) for item in state.clusters) == [1, 2]

    def test_cluster_metadata_is_bounded_without_evicting_targets(self) -> None:
        """Novel contexts merge or evict metadata while every target remains stored."""
        state = TrainingState()
        config = ContextConfig(
            maximum_clusters_per_partition=2,
            assignment_distance=0.5,
            merge_distance=0.1,
            variance_floor=0.01,
        )
        for index, center in enumerate((0.0, 1.0, 2.0, 2.1, 2.2, 2.3)):
            add_target(state, target(index, (center, center)), config)
        assert len(state.clusters) <= 2
        assert [item.sequence for item in state.targets] == list(range(6))
        assert all(item.target_sequences for item in state.clusters)

    def test_router_selects_posture_experts_and_falls_back_out_of_distribution(self) -> None:
        """Passive context chooses local experts without exposing profile controls."""
        router = ModelRouter(
            model(400.0),
            (
                ContextExpert(cluster("seated", 0.0), model(100.0)),
                ContextExpert(cluster("standing", 1.0), model(900.0)),
            ),
            camera_id="camera-a",
            feature_schema="gaze-v1",
            topology_quality=TopologyQuality.EXACT,
            config=ContextConfig(variance_floor=0.01, routing_smoothing=0.5),
        )
        seated_point, seated = router.predict_with_decision((0.5,), (0.0, 0.5))
        _bayesian_point, uncertainty = router.predict_with_uncertainty(
            (0.5,),
            (0.0, 0.5),
        )
        assert uncertainty is not None
        assert uncertainty > 0.0
        assert "seated" in seated.label
        assert seated.confidence_label == "inferred-compatible"
        assert seated_point.x < 500.0

        standing_point = seated_point
        standing = seated
        for _ in range(8):
            standing_point, standing = router.predict_with_decision((0.5,), (1.0, 0.5))
        assert "standing" in standing.label
        assert standing_point.x > 500.0

        fallback_point, fallback = router.predict_with_decision((0.5,), (10.0, 10.0))
        assert fallback.out_of_distribution
        assert fallback.confidence_label == "inferred-low"
        assert fallback.weights[0][0] == "global"
        assert 0.0 <= fallback_point.x <= 1000.0

    def test_topology_mapping_runs_once_after_expert_blending(self) -> None:
        """A changed output cannot put experts into different frames before blending."""
        global_model = model(0.0)
        local_model = model(1000.0)
        mapped: list[Point] = []

        def map_blend(point: Point) -> Point:
            mapped.append(point)
            return Point(point.x + 25.0, point.y + 10.0)

        router = ModelRouter(
            global_model,
            (ContextExpert(cluster("near", 0.0), local_model),),
            camera_id="camera-a",
            feature_schema="gaze-v1",
            topology_quality=TopologyQuality.STRONG,
            point_mapper=map_blend,
        )
        point, decision = router.predict_with_decision((0.5,), (0.0, 0.5))
        models = {"global": global_model, "near": local_model}
        expected_x = sum(
            models[name].predict((0.5,)).x * weight for name, weight in decision.weights
        )
        assert len(mapped) == 1
        assert abs(mapped[0].x - expected_x) < 1e-9
        assert abs(point.x - (expected_x + 25.0)) < 1e-9
        assert point.y == 10.0

    def test_weak_topology_confidence_remains_explicitly_inferred(self) -> None:
        """A mapped model cannot present topology compatibility as holdout accuracy."""
        router = ModelRouter(
            model(400.0),
            (ContextExpert(cluster("near", 0.0), model(100.0)),),
            camera_id="camera-a",
            feature_schema="gaze-v1",
            topology_quality=TopologyQuality.WEAK,
        )
        _point, decision = router.predict_with_decision((0.5,), (0.0, 0.5))
        assert decision.confidence_label == "inferred-weak"

    def test_compatible_noise_summaries_bound_adaptive_smoothing(self) -> None:
        """Stationary target spread reduces jitter without freezing navigation."""
        topology = DisplayTopology((DisplayRegion("display", 0, 0, 1000, 700),))
        noisy = replace(
            target(1, (0.0, 0.5)),
            noise=CursorNoiseSummary(120, 18.0, 16.0, 30.0, 20.0, 38.0),
        )
        settings = noise_smoothing_for(
            TrainingState(targets=[noisy]),
            topology,
            camera_id="camera-a",
            feature_schema="gaze-v1",
            context=(0.0, 0.5),
            defaults=(0.35, 6.0),
            bounds=SmoothingBounds(),
        )
        assert settings.alpha < 0.35
        assert 6.0 < settings.dead_zone <= SmoothingBounds().maximum_dead_zone
        assert settings.confidence == "inferred-compatible"

        fallback = noise_smoothing_for(
            TrainingState(targets=[replace(noisy, noise=None)]),
            topology,
            camera_id="camera-a",
            feature_schema="gaze-v1",
            context=(0.0, 0.5),
            defaults=(0.35, 6.0),
            bounds=SmoothingBounds(),
        )
        assert fallback.alpha == 0.35
        assert fallback.dead_zone == 6.0
        assert fallback.confidence == "default"

    def test_posture_precedes_illumination_in_context_routing(self) -> None:
        """Lighting refines routing without overriding a posture match."""
        posture_match = ContextCluster(
            "posture-match",
            "camera-a",
            "gaze-v1",
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.01,) * 9,
            5,
            (),
            50.0,
            60.0,
        )
        lighting_match = ContextCluster(
            "lighting-match",
            "camera-a",
            "gaze-v1",
            (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
            (0.01,) * 9,
            5,
            (),
            50.0,
            60.0,
        )
        router = ModelRouter(
            model(500.0),
            (
                ContextExpert(posture_match, model(100.0)),
                ContextExpert(lighting_match, model(900.0)),
            ),
            camera_id="camera-a",
            feature_schema="gaze-v1",
            topology_quality=TopologyQuality.EXACT,
            config=ContextConfig(minimum_global_weight=0.01),
        )
        _point, decision = router.predict_with_decision(
            (0.5,),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        )
        assert decision.weights[0][0] == "posture-match"

    def test_routing_weights_change_smoothly(self) -> None:
        """A small context boundary crossing cannot instantly replace all weight."""
        router = ModelRouter(
            model(400.0),
            (
                ContextExpert(cluster("left-context", 0.0), model(100.0)),
                ContextExpert(cluster("right-context", 0.2), model(900.0)),
            ),
            camera_id="camera-a",
            feature_schema="gaze-v1",
            topology_quality=TopologyQuality.STRONG,
            config=ContextConfig(variance_floor=0.01, routing_smoothing=0.1, switching_margin=0.2),
        )
        _point, before = router.predict_with_decision((0.5,), (0.0, 0.5))
        _point, after = router.predict_with_decision((0.5,), (0.2, 0.5))
        before_weights = dict(before.weights)
        after_weights = dict(after.weights)
        assert after_weights.get("left-context", 0.0) > 0.0
        assert (
            abs(after_weights.get("right-context", 0.0) - before_weights.get("right-context", 0.0))
            < 0.2
        )

    def test_candidate_must_not_regress_either_accuracy_gate(self) -> None:
        """Persistent updates preserve both median and edge/corner quality."""
        incumbent = ValidationMetrics(90.0, 95.0)
        assert candidate_is_acceptable(incumbent, ValidationMetrics(80.0, 95.0))
        assert not candidate_is_acceptable(incumbent, ValidationMetrics(80.0, 96.0))
        assert not candidate_is_acceptable(incumbent, ValidationMetrics(91.0, 90.0))
        assert candidate_is_acceptable(None, ValidationMetrics(500.0, 600.0))

    def test_legacy_validated_model_preserves_unchanged_output_reach(self) -> None:
        """A moved peer output cannot replace validated range with a compressed refit."""
        outputs = (
            OutputDescriptor("best-left", 0, 0, 1000, 700),
            OutputDescriptor("best-lower", 1000, 700, 800, 600),
        )
        inferior_outputs = (
            OutputDescriptor("older-left", 0, 0, 1000, 700),
            OutputDescriptor("older-lower", 1000, 700, 800, 600),
        )
        state = TrainingState(
            next_sequence=6,
            targets=[
                StoredTarget(
                    sequence=index,
                    camera_id="camera-a",
                    feature_schema="gaze-v1",
                    features=((index % 3) / 2.0,),
                    context=(0.0, 0.5),
                    outputs=outputs if index >= 3 else inferior_outputs,
                    output_key="best-left" if index >= 3 else "older-left",
                    target_u=0.5,
                    target_v=0.5,
                    desktop_u=0.25,
                    desktop_v=0.25,
                    zone="center",
                )
                for index in range(6)
            ],
        )
        topology_id = legacy_topology_id(outputs)
        inferior_id = legacy_topology_id(inferior_outputs)
        state.models[f"camera-a:{topology_id}:global"] = model(0.0).to_record()
        state.models[f"camera-a:{inferior_id}:global"] = model(400.0).to_record()
        state.validations.extend(
            (
                ValidationSummary(3, "camera-a", inferior_id, "global", 90.0, 90.0),
                ValidationSummary(6, "camera-a", topology_id, "global", 50.0, 60.0),
            )
        )
        current = DisplayTopology(
            (
                DisplayRegion("new-left", 0, 0, 1000, 700),
                DisplayRegion("new-lower", 1200, 700, 800, 600),
            )
        )

        router = build_router(
            state,
            current,
            camera_id="camera-a",
            feature_schema="gaze-v1",
        )
        prediction, decision = router.predict_with_decision((0.0,), (0.0, 0.5))
        assert prediction.x < 100.0
        assert decision.topology_quality is TopologyQuality.STRONG

        removed = DisplayTopology((DisplayRegion("new-left", 0, 0, 1000, 700),))
        removed_router = build_router(
            state,
            removed,
            camera_id="camera-a",
            feature_schema="gaze-v1",
        )
        removed_prediction, removed_decision = removed_router.predict_with_decision(
            (0.0,),
            (0.0, 0.5),
        )
        assert removed_prediction.x < 100.0
        assert removed_decision.topology_quality is TopologyQuality.WEAK

    def test_validated_training_anchors_interpolate_by_posture(self) -> None:
        """Runtime blends real training results without refitting anchor models."""
        outputs = (OutputDescriptor("display", 0, 0, 1000, 700),)
        topology = DisplayTopology((DisplayRegion("current", 0, 0, 1000, 700),))
        state = TrainingState(
            next_sequence=3,
            targets=[target(index, (index / 2.0, 0.5)) for index in range(3)],
            models={f"camera-a:{topology.topology_id}:global": model(400.0).to_record()},
            anchors=[
                ModelAnchor(
                    10,
                    "camera-a",
                    "gaze-v1",
                    topology.topology_id,
                    outputs,
                    (0.0, 0.5),
                    (0.01, 0.01),
                    model(100.0).to_record(),
                    50.0,
                    60.0,
                ),
                ModelAnchor(
                    20,
                    "camera-a",
                    "gaze-v1",
                    topology.topology_id,
                    outputs,
                    (0.062, 0.5),
                    (0.01, 0.01),
                    model(900.0).to_record(),
                    50.0,
                    60.0,
                ),
            ],
        )
        seated = build_router(
            state,
            topology,
            camera_id="camera-a",
            feature_schema="gaze-v1",
        )
        seated_point, seated_decision = seated.predict_with_decision((0.5,), (0.0, 0.5))
        assert "anchor-10" in seated_decision.label
        assert seated_point.x < 500.0

        standing = build_router(
            state,
            topology,
            camera_id="camera-a",
            feature_schema="gaze-v1",
        )
        standing_point, standing_decision = standing.predict_with_decision(
            (0.5,),
            (0.062, 0.5),
        )
        assert standing_decision.label == "anchor-20"
        assert standing_point.x > 800.0
        assert set(standing.records()) == {"global"}

        midpoint = build_router(
            state,
            topology,
            camera_id="camera-a",
            feature_schema="gaze-v1",
        )
        midpoint_point, midpoint_decision = midpoint.predict_with_decision(
            (0.5,),
            (0.031, 0.5),
        )
        assert "anchor-10" in midpoint_decision.label
        assert "anchor-20" in midpoint_decision.label
        assert 250.0 < midpoint_point.x < 750.0

        outside = build_router(
            state,
            topology,
            camera_id="camera-a",
            feature_schema="gaze-v1",
        )
        outside_point, outside_decision = outside.predict_with_decision(
            (0.5,),
            (0.8, 0.5),
        )
        assert outside_decision.label == "anchor-20"
        assert outside_decision.confidence_label == "inferred-low"
        assert outside_point.x > 800.0

    def test_camera_angle_anchors_preserve_the_same_intended_point(self) -> None:
        """Camera-relative feature shifts do not become cursor motion after routing."""
        outputs = (OutputDescriptor("display", 0, 0, 1000, 700),)
        topology = DisplayTopology((DisplayRegion("current", 0, 0, 1000, 700),))
        state = TrainingState(
            anchors=[
                ModelAnchor(
                    10,
                    "camera-a",
                    "gaze-v1",
                    topology.topology_id,
                    outputs,
                    (0.0, 0.5),
                    (0.01, 0.01),
                    model(0.0).to_record(),
                    50.0,
                    60.0,
                ),
                ModelAnchor(
                    20,
                    "camera-a",
                    "gaze-v1",
                    topology.topology_id,
                    outputs,
                    (0.062, 0.5),
                    (0.01, 0.01),
                    model(-30.0).to_record(),
                    50.0,
                    60.0,
                ),
            ]
        )
        first = build_router(
            state,
            topology,
            camera_id="camera-a",
            feature_schema="gaze-v1",
        ).predict((0.5,), (0.0, 0.5))
        rotated = build_router(
            state,
            topology,
            camera_id="camera-a",
            feature_schema="gaze-v1",
        ).predict((0.8,), (0.062, 0.5))
        assert abs(first.x - rotated.x) < 5.0
        assert abs(first.y - rotated.y) < 1.0

    def test_validated_global_is_not_silently_refit_from_old_targets(self) -> None:
        """Retained targets cannot masquerade as a newly validated model result."""
        values = (-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        outputs = (OutputDescriptor("display", 0, 0, 2000, 1000),)
        samples = [
            CalibrationSample(
                (value,),
                Point(1000.0 + 800.0 * value**3, 500.0 + 400.0 * value**2),
            )
            for value in values
        ]
        validated = CalibrationModel.fit(samples)
        assert validated.kind.startswith("affine")
        state = TrainingState(
            next_sequence=len(values),
            targets=[
                StoredTarget(
                    sequence=index,
                    camera_id="camera-a",
                    feature_schema="gaze-v1",
                    features=(value,),
                    context=(0.0, 0.5),
                    outputs=outputs,
                    output_key="display",
                    target_u=(1000.0 + 800.0 * value**3) / 1999.0,
                    target_v=(500.0 + 400.0 * value**2) / 999.0,
                    desktop_u=(1000.0 + 800.0 * value**3) / 1999.0,
                    desktop_v=(500.0 + 400.0 * value**2) / 999.0,
                    zone="center",
                )
                for index, value in enumerate(values)
            ],
        )
        state.clusters.append(
            ContextCluster(
                "new-context",
                "camera-a",
                "gaze-v1",
                (0.0, 0.5),
                (0.01, 0.01),
                len(values),
                tuple(range(len(values))),
            )
        )
        topology_id = topology_id_for_outputs(outputs)
        state.models[f"camera-a:{topology_id}:global"] = validated.to_record()
        state.validations.append(
            ValidationSummary(len(values), "camera-a", topology_id, "global", 50.0, 60.0)
        )
        router = build_router(
            state,
            DisplayTopology((DisplayRegion("new-id", 0, 0, 2000, 1000),)),
            camera_id="camera-a",
            feature_schema="gaze-v1",
        )
        records = router.records()
        assert records["global"]["kind"] == "affine"
        assert "new-context" not in records

    def test_all_invocation_anchor_supersedes_legacy_last_batch_metrics(self) -> None:
        """Old five-target scores cannot outrank complete invocation validation."""
        outputs = (OutputDescriptor("display", 0, 0, 1000, 700),)
        topology = DisplayTopology((DisplayRegion("current", 0, 0, 1000, 700),))
        scoped_record = model(800.0).to_record()
        scoped_record["validation_target_count"] = 10
        state = TrainingState(
            anchors=[
                ModelAnchor(
                    10,
                    "camera-a",
                    "gaze-v1",
                    topology.topology_id,
                    outputs,
                    (0.0, 0.5),
                    (0.01, 0.01),
                    model(0.0).to_record(),
                    1.0,
                    1.0,
                ),
                ModelAnchor(
                    20,
                    "camera-a",
                    "gaze-v1",
                    topology.topology_id,
                    outputs,
                    (0.0, 0.5),
                    (0.01, 0.01),
                    scoped_record,
                    100.0,
                    100.0,
                ),
                ModelAnchor(
                    30,
                    "camera-a",
                    "gaze-v1",
                    topology.topology_id,
                    outputs,
                    (1.0, 0.5),
                    (0.01, 0.01),
                    model(100.0).to_record(),
                    200.0,
                    200.0,
                ),
            ]
        )
        point = build_router(
            state,
            topology,
            camera_id="camera-a",
            feature_schema="gaze-v1",
        ).predict((0.5,), (0.0, 0.5))
        assert point.x > 800.0
        context_point = build_router(
            state,
            topology,
            camera_id="camera-a",
            feature_schema="gaze-v1",
        ).predict((0.5,), (1.0, 0.5))
        assert context_point.x < 500.0

    def test_model_coefficients_round_trip_without_training_samples(self) -> None:
        """The store can restore fitted coefficients after restart."""
        original = model(123.0)
        restored = CalibrationModel.from_record(original.to_record())
        expected = original.predict((0.25,))
        actual = restored.predict((0.25,))
        assert abs(actual.x - expected.x) < 1e-9
        assert abs(actual.y - expected.y) < 1e-9


if __name__ == "__main__":
    unittest.main()
