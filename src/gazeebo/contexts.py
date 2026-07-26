"""Bounded context clustering and global/local gaze-model routing."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from gazeebo.adaptation import (
    TopologyQuality,
    legacy_topology_id,
    map_model_point,
    map_stored_target,
    model_mapping_supported,
    output_mapping_quality,
    topology_id_for_outputs,
)
from gazeebo.calibration import CalibrationModel, CalibrationSample, target_fit_weight
from gazeebo.geometry import Point
from gazeebo.state import ContextCluster, ModelAnchor, OutputDescriptor, StoredTarget, TrainingState

MINIMUM_MODEL_TARGETS = 3
PUPIL_AVAILABILITY_INDEX = 10
PUPIL_AVAILABLE_THRESHOLD = 0.5
MINIMUM_ROUTING_WEIGHT = 1e-6
ROUTING_LABEL_WEIGHT = 0.05
ANCHOR_ROUTING_TEMPERATURE = 0.15
ANCHOR_MINIMUM_GLOBAL_WEIGHT = 0.01
GAZE_CONTEXT_DIMENSIONS = 9
POSTURE_CONTEXT_DIMENSIONS = 7
ILLUMINATION_DISTANCE_WEIGHT = 0.05

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from gazeebo.contracts import FeatureVector
    from gazeebo.geometry import DisplayTopology


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Finite clustering and routing policy."""

    maximum_clusters_per_partition: int = 8
    assignment_distance: float = 2.5
    merge_distance: float = 1.25
    variance_floor: float = 0.0025
    routing_smoothing: float = 0.25
    switching_margin: float = 0.15
    minimum_global_weight: float = 0.15

    def __post_init__(self) -> None:
        """Reject unbounded or numerically invalid context policy."""
        if self.maximum_clusters_per_partition <= 0:
            msg = "context limits must be positive"
            raise ValueError(msg)
        if (
            self.assignment_distance <= 0.0
            or self.merge_distance < 0.0
            or self.variance_floor <= 0.0
            or not 0.0 < self.routing_smoothing <= 1.0
            or self.switching_margin < 0.0
            or not 0.0 < self.minimum_global_weight <= 1.0
        ):
            msg = "context distances and routing settings are invalid"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SmoothingBounds:
    """Conservative limits for stationary-noise cursor adaptation."""

    minimum_alpha: float = 0.12
    maximum_alpha: float = 0.65
    minimum_dead_zone: float = 2.0
    maximum_dead_zone: float = 40.0
    minimum_samples: int = 20
    maximum_targets: int = 32

    def __post_init__(self) -> None:
        """Require finite ordered limits and bounded evidence counts."""
        values = (
            self.minimum_alpha,
            self.maximum_alpha,
            self.minimum_dead_zone,
            self.maximum_dead_zone,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or not 0.0 < self.minimum_alpha <= self.maximum_alpha <= 1.0
            or not 0.0 <= self.minimum_dead_zone <= self.maximum_dead_zone
            or self.minimum_samples <= 0
            or self.maximum_targets <= 0
        ):
            msg = "noise smoothing bounds are invalid"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SmoothingSettings:
    """One inferred context-compatible smoothing selection."""

    alpha: float
    dead_zone: float
    confidence: str


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """The two accuracy gates used for persistent candidate acceptance."""

    median_error: float
    edge_error: float

    def __post_init__(self) -> None:
        """Reject invalid validation errors."""
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.median_error, self.edge_error)
        ):
            msg = "validation metrics must be finite and non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ContextExpert:
    """One local estimator and its routing statistics."""

    cluster: ContextCluster
    model: CalibrationModel
    persistent: bool = True
    routing_temperature: float = 1.0

    def __post_init__(self) -> None:
        """Require a finite positive routing temperature."""
        if not math.isfinite(self.routing_temperature) or self.routing_temperature <= 0.0:
            msg = "expert routing temperature must be finite and positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Smoothed weights and quality exposed to runtime status and the HUD."""

    weights: tuple[tuple[str, float], ...]
    topology_quality: TopologyQuality
    out_of_distribution: bool

    @property
    def label(self) -> str:
        """Return a stable human-readable blend label."""
        return "+".join(name for name, weight in self.weights if weight >= ROUTING_LABEL_WEIGHT)

    @property
    def confidence_label(self) -> str:
        """Describe inferred compatibility without claiming measured accuracy."""
        if self.out_of_distribution:
            return "inferred-low"
        if self.topology_quality is TopologyQuality.WEAK:
            return "inferred-weak"
        return "inferred-compatible"


class ModelRouter:
    """Blend a global estimator with compatible context experts."""

    def __init__(  # noqa: PLR0913
        self,
        global_model: CalibrationModel,
        experts: Sequence[ContextExpert],
        *,
        camera_id: str,
        feature_schema: str,
        topology_quality: TopologyQuality,
        config: ContextConfig | None = None,
        point_mapper: Callable[[Point], Point] | None = None,
        nearest_expert_fallback: bool = False,
        head_fallback: CalibrationModel | None = None,
    ) -> None:
        """Bind compatible estimators to one implicit camera/schema partition."""
        self._global = global_model
        self._experts = tuple(experts)
        self._camera_id = camera_id
        self._feature_schema = feature_schema
        self._quality = topology_quality
        self._config = config or ContextConfig()
        self._point_mapper = point_mapper
        self._nearest_expert_fallback = nearest_expert_fallback
        self._head_fallback = head_fallback
        self._weights: dict[str, float] = {"global": 1.0}
        self._last_best = "global"
        self._initialized = False
        self._last_decision: RoutingDecision | None = None

    def decide(self, context: tuple[float, ...]) -> RoutingDecision:
        """Update context weights with a switching margin and EWMA smoothing."""
        compatible: list[tuple[ContextExpert, float, float]] = []
        for expert in self._experts:
            cluster = expert.cluster
            if (
                cluster.camera_id != self._camera_id
                or cluster.feature_schema != self._feature_schema
            ):
                continue
            distance = context_distance(context, cluster, self._config.variance_floor)
            quality = _validation_weight(cluster)
            scaled_distance = distance / expert.routing_temperature
            compatible.append((expert, distance, math.exp(-0.5 * scaled_distance**2) * quality))
        compatible.sort(key=lambda item: (-item[2], item[0].cluster.cluster_id))
        selected = compatible[:2]
        nearest = min((item[1] for item in selected), default=math.inf)
        out_of_distribution = nearest > self._config.assignment_distance
        if out_of_distribution and self._nearest_expert_fallback and compatible:
            nearest_expert = min(
                compatible,
                key=lambda item: (item[1], item[0].cluster.cluster_id),
            )
            raw = {nearest_expert[0].cluster.cluster_id: 1.0}
        else:
            raw = {
                "global": 1.0
                if out_of_distribution
                else max(
                    self._config.minimum_global_weight,
                    nearest / self._config.assignment_distance,
                )
            }
            for expert, _distance, score in selected:
                raw[expert.cluster.cluster_id] = score

        best = max(raw, key=raw.__getitem__)
        previous_score = raw.get(self._last_best, 0.0)
        if best != self._last_best and raw[best] < previous_score * (
            1.0 + self._config.switching_margin
        ):
            raw[self._last_best] = max(raw[best], previous_score)
            best = self._last_best
        self._last_best = best

        names = set(self._weights) | set(raw)
        alpha = self._config.routing_smoothing
        if self._initialized:
            smoothed = {
                name: (1.0 - alpha) * self._weights.get(name, 0.0) + alpha * raw.get(name, 0.0)
                for name in names
            }
        else:
            smoothed = raw
            self._initialized = True
        total = sum(smoothed.values())
        self._weights = {
            name: value / total
            for name, value in smoothed.items()
            if value > MINIMUM_ROUTING_WEIGHT
        }
        ordered = tuple(sorted(self._weights.items(), key=lambda item: (-item[1], item[0])))
        decision = RoutingDecision(ordered, self._quality, out_of_distribution)
        self._last_decision = decision
        return decision

    @property
    def kind(self) -> str:
        """Describe the adaptive estimator family."""
        return "context-mixture"

    @property
    def last_decision(self) -> RoutingDecision | None:
        """Expose the most recent routing result for status and diagnostics."""
        return self._last_decision

    def predict(
        self,
        features: FeatureVector,
        context: tuple[float, ...] | None = None,
    ) -> Point:
        """Blend current model predictions using smoothed context routing."""
        if context is None:
            msg = "context model requires a routing context"
            raise ValueError(msg)
        point, _decision = self.predict_with_decision(features, context)
        return point

    def predict_with_uncertainty(
        self,
        features: FeatureVector,
        context: tuple[float, ...] | None = None,
    ) -> tuple[Point, float | None]:
        """Blend posterior means and parameter uncertainty across routed experts."""
        if context is None:
            msg = "context model requires a routing context"
            raise ValueError(msg)
        point, decision = self.predict_with_decision(features, context)
        if (
            len(features) > PUPIL_AVAILABILITY_INDEX
            and features[PUPIL_AVAILABILITY_INDEX] < PUPIL_AVAILABLE_THRESHOLD
            and self._head_fallback is not None
        ):
            _head_point, uncertainty = self._head_fallback.predict_with_uncertainty(features)
            return point, uncertainty
        models = {expert.cluster.cluster_id: expert.model for expert in self._experts}
        models["global"] = self._global
        variance = 0.0
        used = 0.0
        for name, weight in decision.weights:
            model = models.get(name)
            if model is None:
                continue
            _model_point, uncertainty = model.predict_with_uncertainty(features)
            if uncertainty is None:
                return point, None
            variance += weight * uncertainty**2
            used += weight
        if used <= 0.0:
            return point, None
        return point, math.sqrt(variance / used)

    def predict_with_decision(
        self,
        features: FeatureVector,
        context: tuple[float, ...],
    ) -> tuple[Point, RoutingDecision]:
        """Return a blended prediction and its observable routing decision."""
        decision = self.decide(context)
        if (
            len(features) > PUPIL_AVAILABILITY_INDEX
            and features[PUPIL_AVAILABILITY_INDEX] < PUPIL_AVAILABLE_THRESHOLD
            and self._head_fallback is not None
        ):
            point = self._head_fallback.predict(features)
            if self._point_mapper is not None:
                point = self._point_mapper(point)
            head_decision = replace(decision, weights=(("head+face", 1.0),))
            self._last_decision = head_decision
            return point, head_decision
        models = {expert.cluster.cluster_id: expert.model for expert in self._experts}
        models["global"] = self._global
        x = 0.0
        y = 0.0
        used = 0.0
        for name, weight in decision.weights:
            model = models.get(name)
            if model is None:
                continue
            point = model.predict(features)
            x += point.x * weight
            y += point.y * weight
            used += weight
        if used <= 0.0:
            msg = "model router has no usable estimator"
            raise RuntimeError(msg)
        point = Point(x / used, y / used)
        if self._point_mapper is not None:
            point = self._point_mapper(point)
        self._last_decision = decision
        return point, decision

    def with_validated_model(
        self,
        model: CalibrationModel,
        cluster_id: str,
        *,
        replace_global: bool,
    ) -> ModelRouter:
        """Use the exact model measured by the terminal unseen batch."""
        experts = tuple(
            ContextExpert(expert.cluster, model, expert.persistent)
            if expert.cluster.cluster_id == cluster_id
            else expert
            for expert in self._experts
        )
        return ModelRouter(
            model if replace_global else self._global,
            experts,
            camera_id=self._camera_id,
            feature_schema=self._feature_schema,
            topology_quality=self._quality,
            config=self._config,
            point_mapper=self._point_mapper,
            nearest_expert_fallback=self._nearest_expert_fallback,
            head_fallback=self._head_fallback,
        )

    def with_head_fallback(self, model: CalibrationModel | None) -> ModelRouter:
        """Attach a transient head/face-only model for missing pupil evidence."""
        self._head_fallback = model
        return self

    def records(self) -> dict[str, dict[str, object]]:
        """Return serializable global and expert coefficients."""
        records = {"global": self._global.to_record()}
        records.update(
            {
                expert.cluster.cluster_id: expert.model.to_record()
                for expert in self._experts
                if expert.persistent
            }
        )
        return records


def context_distance(
    context: tuple[float, ...],
    cluster: ContextCluster,
    variance_floor: float,
) -> float:
    """Return diagonal-variance normalized context distance."""
    if len(context) != len(cluster.centroid):
        return math.inf
    normalized = _normalized_context_difference(
        context,
        cluster.centroid,
        cluster.variance,
        variance_floor,
    )
    if len(context) != GAZE_CONTEXT_DIMENSIONS:
        return float(np.sqrt(np.mean(normalized**2)))
    posture = float(np.sqrt(np.mean(normalized[:POSTURE_CONTEXT_DIMENSIONS] ** 2)))
    illumination = float(np.sqrt(np.mean(normalized[POSTURE_CONTEXT_DIMENSIONS:] ** 2)))
    return posture + ILLUMINATION_DISTANCE_WEIGHT * illumination


def _normalized_context_difference(
    context: tuple[float, ...],
    centroid: tuple[float, ...],
    variance: tuple[float, ...],
    variance_floor: float,
) -> np.ndarray:
    """Normalize one context difference with the persistent variance floor."""
    difference = np.asarray(context) - np.asarray(centroid)
    scale = np.sqrt(np.maximum(np.asarray(variance), variance_floor))
    normalized: np.ndarray = difference / scale
    return normalized


def noise_smoothing_for(  # noqa: PLR0913
    state: TrainingState,
    topology: DisplayTopology,
    *,
    camera_id: str,
    feature_schema: str,
    context: tuple[float, ...],
    defaults: tuple[float, float],
    bounds: SmoothingBounds,
    config: ContextConfig | None = None,
) -> SmoothingSettings:
    """Infer bounded smoothing from compatible stationary target summaries."""
    policy = config or ContextConfig()
    candidates: list[tuple[StoredTarget, float, TopologyQuality]] = []
    for target in state.targets:
        if (
            target.noise is None
            or target.camera_id != camera_id
            or target.feature_schema != feature_schema
            or len(target.context) != len(context)
        ):
            continue
        mapped = map_stored_target(target, topology)
        if mapped is None:
            continue
        temporary = ContextCluster(
            "noise-context",
            camera_id,
            feature_schema,
            target.context,
            tuple(policy.variance_floor for _ in target.context),
            1,
            (target.sequence,),
        )
        candidates.append(
            (
                target,
                context_distance(context, temporary, policy.variance_floor),
                mapped.quality,
            )
        )
    if not candidates:
        return SmoothingSettings(*defaults, "default")
    best_quality = max(item[2] for item in candidates)
    candidates = sorted(
        (item for item in candidates if item[2] is best_quality),
        key=lambda item: (item[1], item[0].sequence),
    )[: bounds.maximum_targets]
    total_samples = sum(item[0].noise.sample_count for item in candidates if item[0].noise)
    if total_samples < bounds.minimum_samples:
        return SmoothingSettings(*defaults, "default")
    weighted: list[tuple[float, float]] = []
    for target, distance, _quality in candidates:
        noise = target.noise
        if noise is None:
            continue
        weight = math.exp(-0.5 * (distance / policy.assignment_distance) ** 2)
        weight *= noise.sample_count
        weighted.append((noise.p95_radial_spread, weight))
    total_weight = sum(weight for _spread, weight in weighted)
    if total_weight <= MINIMUM_ROUTING_WEIGHT:
        return SmoothingSettings(*defaults, "default")
    spread = sum(value * weight for value, weight in weighted) / total_weight
    default_alpha, default_dead_zone = defaults
    alpha = min(
        bounds.maximum_alpha,
        max(bounds.minimum_alpha, default_alpha / (1.0 + spread / 25.0)),
    )
    dead_zone = min(
        bounds.maximum_dead_zone,
        max(bounds.minimum_dead_zone, default_dead_zone, spread * 0.75),
    )
    nearest = min(item[1] for item in candidates)
    confidence = (
        "inferred-low"
        if nearest > policy.assignment_distance
        else "inferred-weak"
        if best_quality is TopologyQuality.WEAK
        else "inferred-compatible"
    )
    return SmoothingSettings(alpha, dead_zone, confidence)


def add_target(
    state: TrainingState,
    target: StoredTarget,
    config: ContextConfig | None = None,
) -> str:
    """Add one target, update its online context cluster, and rebalance state."""
    policy = config or ContextConfig()
    if any(item.sequence == target.sequence for item in state.targets):
        msg = "training target sequence already exists"
        raise ValueError(msg)
    compatible = [
        cluster
        for cluster in state.clusters
        if cluster.camera_id == target.camera_id and cluster.feature_schema == target.feature_schema
    ]
    nearest = min(
        compatible,
        key=lambda cluster: (
            context_distance(target.context, cluster, policy.variance_floor),
            cluster.cluster_id,
        ),
        default=None,
    )
    if (
        nearest is None
        or context_distance(target.context, nearest, policy.variance_floor)
        > policy.assignment_distance
    ):
        if len(compatible) >= policy.maximum_clusters_per_partition:
            _make_cluster_room(state, compatible, policy)
        cluster_id = f"context-{target.sequence}"
        cluster = ContextCluster(
            cluster_id,
            target.camera_id,
            target.feature_schema,
            target.context,
            tuple(policy.variance_floor for _ in target.context),
            1,
            (target.sequence,),
        )
        state.clusters.append(cluster)
    else:
        cluster_id = nearest.cluster_id

    state.targets.append(target)
    state.next_sequence = max(state.next_sequence, target.sequence + 1)
    _rebalance(state, cluster_id, policy)
    return cluster_id


def calibration_samples_for(
    state: TrainingState,
    topology: DisplayTopology,
    *,
    camera_id: str,
    feature_schema: str,
) -> list[CalibrationSample]:
    """Return compatible stored targets remapped to current geometry."""
    compatible = [
        target
        for target in state.targets
        if target.camera_id == camera_id and target.feature_schema == feature_schema
    ]
    return [sample for _sequence, sample, _quality in _mapped_samples(compatible, topology)]


def build_router(  # noqa: PLR0913
    state: TrainingState,
    topology: DisplayTopology,
    *,
    camera_id: str,
    feature_schema: str,
    config: ContextConfig | None = None,
    reuse_validated: bool = True,
) -> ModelRouter:
    """Reuse validated anchors or explicitly fit a transient candidate."""
    policy = config or ContextConfig()
    compatible = [
        target
        for target in state.targets
        if target.camera_id == camera_id and target.feature_schema == feature_schema
    ]
    mapped = _mapped_samples(compatible, topology)
    head_fallback = _head_fallback_for(mapped, topology)
    anchor_router = (
        _validated_anchor_router(
            state.anchors,
            topology,
            camera_id=camera_id,
            feature_schema=feature_schema,
            config=policy,
        )
        if reuse_validated
        else None
    )
    if anchor_router is not None:
        return anchor_router.with_head_fallback(head_fallback)
    if len(mapped) < MINIMUM_MODEL_TARGETS:
        msg = "stored training data do not contain three compatible targets"
        raise ValueError(msg)
    quality = min(item[2] for item in mapped)
    prefix = f"{camera_id}:{topology.topology_id}:"
    stored_global = state.models.get(f"{prefix}global") if reuse_validated else None
    if stored_global is not None:
        try:
            global_model = _load_global_model(stored_global)
            exact_experts = _load_experts(
                state,
                prefix,
                state.clusters,
                camera_id=camera_id,
                feature_schema=feature_schema,
            )
            return ModelRouter(
                global_model,
                exact_experts,
                camera_id=camera_id,
                feature_schema=feature_schema,
                topology_quality=TopologyQuality.EXACT,
                config=policy,
                head_fallback=head_fallback,
            )
        except (KeyError, TypeError, ValueError):
            pass
    stored_router = (
        _compatible_stored_router(
            state,
            compatible,
            topology,
            camera_id=camera_id,
            feature_schema=feature_schema,
            config=policy,
            anchor_experts=(),
        )
        if reuse_validated
        else None
    )
    if stored_router is not None:
        return stored_router.with_head_fallback(head_fallback)
    global_model = CalibrationModel.fit(
        [sample for _sequence, sample, _quality in mapped],
        topology=topology,
    )
    by_sequence = {sequence: sample for sequence, sample, _item_quality in mapped}
    experts: list[ContextExpert] = []
    for cluster in state.clusters:
        if cluster.camera_id != camera_id or cluster.feature_schema != feature_schema:
            continue
        samples = [
            by_sequence[sequence]
            for sequence in cluster.target_sequences
            if sequence in by_sequence
        ]
        if len(samples) >= MINIMUM_MODEL_TARGETS:
            experts.append(
                ContextExpert(
                    cluster,
                    CalibrationModel.fit(samples, topology=topology),
                )
            )
    return ModelRouter(
        global_model,
        experts,
        camera_id=camera_id,
        feature_schema=feature_schema,
        topology_quality=quality,
        config=policy,
        head_fallback=head_fallback,
    )


def _head_fallback_for(
    mapped: Sequence[tuple[int, CalibrationSample, TopologyQuality]],
    topology: DisplayTopology,
) -> CalibrationModel | None:
    samples = [sample for _sequence, sample, _quality in mapped]
    try:
        return CalibrationModel.fit_head(samples, topology=topology)
    except ValueError:
        return None


def _anchor_validation_target_count(anchor: ModelAnchor) -> int:
    """Return the all-invocation validation scope recorded by new anchors."""
    value = anchor.model.get("validation_target_count", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return 0
    return value


def _anchor_context_distance(
    left: ModelAnchor,
    right: ModelAnchor,
    variance_floor: float,
) -> float:
    """Measure whether two accepted anchors represent distinct live contexts."""
    cluster = ContextCluster(
        "anchor-distance",
        right.camera_id,
        right.feature_schema,
        right.context_centroid,
        right.context_variance,
        1,
        (),
    )
    return context_distance(left.context_centroid, cluster, variance_floor)


def _validated_anchor_router(
    anchors: Sequence[ModelAnchor],
    topology: DisplayTopology,
    *,
    camera_id: str,
    feature_schema: str,
    config: ContextConfig,
) -> ModelRouter | None:
    """Route only among accepted all-data models from one source geometry."""
    groups: dict[tuple[OutputDescriptor, ...], list[ModelAnchor]] = {}
    for anchor in anchors:
        if anchor.camera_id == camera_id and anchor.feature_schema == feature_schema:
            groups.setdefault(anchor.outputs, []).append(anchor)
    if not groups:
        return None

    source, selected = min(
        groups.items(),
        key=lambda item: (
            -float(output_mapping_quality(item[0], topology)),
            0 if any(_anchor_validation_target_count(anchor) for anchor in item[1]) else 1,
            min(max(anchor.median_error, anchor.edge_error) for anchor in item[1]),
            min(anchor.median_error + anchor.edge_error for anchor in item[1]),
            -max(anchor.sequence for anchor in item[1]),
        ),
    )
    scoped = [anchor for anchor in selected if _anchor_validation_target_count(anchor) > 0]
    if scoped:
        selected = [
            *scoped,
            *(
                anchor
                for anchor in selected
                if _anchor_validation_target_count(anchor) == 0
                and all(
                    _anchor_context_distance(anchor, complete, config.variance_floor)
                    > ANCHOR_ROUTING_TEMPERATURE
                    for complete in scoped
                )
            ),
        ]
    experts: list[ContextExpert] = []
    models: dict[int, CalibrationModel] = {}
    for anchor in selected:
        try:
            model = CalibrationModel.from_record(anchor.model)
        except (TypeError, ValueError):
            continue
        cluster = ContextCluster(
            f"anchor-{anchor.sequence}",
            anchor.camera_id,
            anchor.feature_schema,
            anchor.context_centroid,
            anchor.context_variance,
            1,
            (),
            median_error=anchor.median_error,
            edge_error=anchor.edge_error,
        )
        models[anchor.sequence] = model
        experts.append(
            ContextExpert(
                cluster,
                model,
                persistent=False,
                routing_temperature=ANCHOR_ROUTING_TEMPERATURE,
            )
        )
    if not experts:
        return None
    fallback = min(
        (anchor for anchor in selected if anchor.sequence in models),
        key=lambda anchor: (
            0 if _anchor_validation_target_count(anchor) > 0 else 1,
            max(anchor.median_error, anchor.edge_error),
            anchor.median_error + anchor.edge_error,
            -anchor.sequence,
        ),
    )
    quality = output_mapping_quality(source, topology)
    return ModelRouter(
        models[fallback.sequence],
        experts,
        camera_id=camera_id,
        feature_schema=feature_schema,
        topology_quality=quality,
        config=replace(config, minimum_global_weight=ANCHOR_MINIMUM_GLOBAL_WEIGHT),
        point_mapper=(
            None if quality is TopologyQuality.EXACT else _topology_point_mapper(source, topology)
        ),
        nearest_expert_fallback=True,
    )


def _load_experts(
    state: TrainingState,
    prefix: str,
    clusters: Sequence[ContextCluster],
    *,
    camera_id: str,
    feature_schema: str,
) -> list[ContextExpert]:
    """Load only coefficient sets saved after folded no-regression acceptance."""
    experts: list[ContextExpert] = []
    for cluster in clusters:
        if cluster.camera_id != camera_id or cluster.feature_schema != feature_schema:
            continue
        record = state.models.get(f"{prefix}{cluster.cluster_id}")
        if record is not None:
            experts.append(ContextExpert(cluster, CalibrationModel.from_record(record)))
    return experts


def _compatible_stored_router(  # noqa: PLR0913
    state: TrainingState,
    targets: Sequence[StoredTarget],
    topology: DisplayTopology,
    *,
    camera_id: str,
    feature_schema: str,
    config: ContextConfig,
    anchor_experts: Sequence[ContextExpert],
) -> ModelRouter | None:
    """Load the best validated model with unambiguous source-output mapping."""
    candidates: list[
        tuple[
            tuple[float, ...],
            str,
            tuple[OutputDescriptor, ...],
            TopologyQuality,
        ]
    ] = []
    sources = {target.outputs for target in targets}
    for source in sources:
        quality = output_mapping_quality(source, topology)
        if not model_mapping_supported(source, topology):
            continue
        for topology_id in {topology_id_for_outputs(source), legacy_topology_id(source)}:
            prefix = f"{camera_id}:{topology_id}:"
            if f"{prefix}global" not in state.models:
                continue
            validations = [
                validation
                for validation in state.validations
                if validation.camera_id == camera_id and validation.topology_id == topology_id
            ]
            if validations:
                validation = min(
                    validations,
                    key=lambda item: (
                        max(item.median_error, item.edge_error),
                        item.median_error + item.edge_error,
                        -item.sequence,
                    ),
                )
                score = (
                    -float(quality),
                    0.0,
                    max(validation.median_error, validation.edge_error),
                    validation.median_error + validation.edge_error,
                    -float(validation.sequence),
                )
            else:
                score = (-float(quality), 1.0, math.inf, math.inf, 0.0)
            candidates.append((score, prefix, source, quality))
    for _score, prefix, source, quality in sorted(candidates, key=lambda item: item[0]):
        try:
            global_model = _load_global_model(state.models[f"{prefix}global"])
            experts = _load_experts(
                state,
                prefix,
                state.clusters,
                camera_id=camera_id,
                feature_schema=feature_schema,
            )
            routed_experts: Sequence[ContextExpert] = experts
            if quality is TopologyQuality.EXACT:
                routed_experts = (*experts, *anchor_experts)
            return ModelRouter(
                global_model,
                routed_experts,
                camera_id=camera_id,
                feature_schema=feature_schema,
                topology_quality=quality,
                config=config,
                point_mapper=(
                    None
                    if quality is TopologyQuality.EXACT
                    else _topology_point_mapper(source, topology)
                ),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _load_global_model(record: dict[str, object]) -> CalibrationModel:
    """Restore the coefficient set accepted by real unseen training."""
    return CalibrationModel.from_record(record)


def _topology_point_mapper(
    source: tuple[OutputDescriptor, ...],
    topology: DisplayTopology,
) -> Callable[[Point], Point]:
    """Bind source and current geometry for post-blend prediction mapping."""

    def mapper(point: Point) -> Point:
        return map_model_point(point, source, topology)

    return mapper


def candidate_is_acceptable(
    incumbent: ValidationMetrics | None,
    candidate: ValidationMetrics,
    *,
    tolerance: float = 0.0,
) -> bool:
    """Accept new contexts or candidates that do not regress either quality gate."""
    if tolerance < 0.0:
        msg = "acceptance tolerance must be non-negative"
        raise ValueError(msg)
    if incumbent is None:
        return True
    return (
        candidate.median_error <= incumbent.median_error + tolerance
        and candidate.edge_error <= incumbent.edge_error + tolerance
    )


def _mapped_samples(
    targets: Iterable[StoredTarget],
    topology: DisplayTopology,
) -> list[tuple[int, CalibrationSample, TopologyQuality]]:
    result: list[tuple[int, CalibrationSample, TopologyQuality]] = []
    for target in targets:
        mapped = map_stored_target(target, topology)
        if mapped is not None:
            result.append(
                (
                    target.sequence,
                    CalibrationSample(
                        target.features,
                        mapped.point,
                        target_fit_weight(target.noise),
                        target.context,
                        target.feature_dispersion,
                    ),
                    mapped.quality,
                )
            )
    return result


def _make_cluster_room(
    state: TrainingState,
    compatible: list[ContextCluster],
    config: ContextConfig,
) -> None:
    pairs = [
        (
            context_distance(left.centroid, right, config.variance_floor),
            left.cluster_id,
            right.cluster_id,
        )
        for index, left in enumerate(compatible)
        for right in compatible[index + 1 :]
    ]
    if pairs:
        distance, left_id, right_id = min(pairs)
        if distance <= config.merge_distance:
            sequences = next(
                cluster.target_sequences for cluster in compatible if cluster.cluster_id == left_id
            ) + next(
                cluster.target_sequences for cluster in compatible if cluster.cluster_id == right_id
            )
            state.clusters = [
                cluster
                for cluster in state.clusters
                if cluster.cluster_id not in {left_id, right_id}
            ]
            samples = [target for target in state.targets if target.sequence in sequences]
            if samples:
                state.clusters.append(_cluster_from_targets(left_id, samples, config))
            return
    evicted = min(
        compatible, key=lambda cluster: (len(cluster.target_sequences), cluster.cluster_id)
    )
    state.clusters = [
        cluster for cluster in state.clusters if cluster.cluster_id != evicted.cluster_id
    ]


def _rebalance(state: TrainingState, cluster_id: str, config: ContextConfig) -> None:
    """Refresh bounded cluster metadata without deleting target evidence."""
    available = {target.sequence: target for target in state.targets}
    newest = max(state.targets, key=lambda item: item.sequence)
    rebuilt: list[ContextCluster] = []
    for cluster in state.clusters:
        sequences = list(cluster.target_sequences)
        if cluster.cluster_id == cluster_id and newest.sequence not in sequences:
            sequences.append(newest.sequence)
        samples = [available[sequence] for sequence in sequences if sequence in available]
        if samples:
            rebuilt.append(_cluster_from_targets(cluster.cluster_id, samples, config, cluster))
    state.clusters = rebuilt


def _cluster_from_targets(
    cluster_id: str,
    targets: Sequence[StoredTarget],
    config: ContextConfig,
    prior: ContextCluster | None = None,
) -> ContextCluster:
    contexts = np.asarray([target.context for target in targets], dtype=np.float64)
    centroid = contexts.mean(axis=0)
    variance = np.maximum(contexts.var(axis=0), config.variance_floor)
    first = targets[0]
    return ContextCluster(
        cluster_id,
        first.camera_id,
        first.feature_schema,
        tuple(float(value) for value in centroid),
        tuple(float(value) for value in variance),
        (prior.sample_count if prior is not None else 0) + 1,
        tuple(target.sequence for target in targets),
        median_error=None if prior is None else prior.median_error,
        edge_error=None if prior is None else prior.edge_error,
    )


def _validation_weight(cluster: ContextCluster) -> float:
    errors = [value for value in (cluster.median_error, cluster.edge_error) if value is not None]
    if not errors:
        return 1.0
    return 1.0 / (1.0 + max(errors) / 100.0)
