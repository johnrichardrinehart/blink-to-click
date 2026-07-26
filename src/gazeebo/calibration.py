"""Deterministic linear-work gaze calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from gazeebo.contracts import DisplayRegion
from gazeebo.geometry import DisplayTopology, Point

MINIMUM_CALIBRATION_SAMPLES = 3
MINIMUM_MODEL_SELECTION_SAMPLES = 8
MAXIMUM_CROSS_VALIDATION_FOLDS = 5
BASE_GAZE_FEATURE_COUNT = 8
BINOCULAR_GAZE_FEATURE_COUNT = 10
PUPIL_OPTIONAL_FEATURE_COUNT = 15
FEATURE_MATRIX_DIMENSIONS = 2
PUPIL_AVAILABILITY_INDEX = 10
PUPIL_AVAILABLE_THRESHOLD = 0.5
HEAD_FEATURE_INDICES = (4, 5, 6, 7, 12, 13, 14)
STABLE_HEAD_FEATURE_INDICES = (4, 5, 6, 7, 13, 14)
HEAD_FEATURE_SETS = (
    (HEAD_FEATURE_INDICES, "head+face"),
    (STABLE_HEAD_FEATURE_INDICES, "head+face-no-roll"),
    ((4, 5, 6, 7), "head+position"),
    ((4, 5, 13, 14), "head+scale"),
)
MINIMUM_FEATURE_SCALE = 1e-6
MINIMUM_TARGET_WEIGHT = 0.25
MISSING_NOISE_WEIGHT = 0.5
NOISE_WEIGHT_SCALE = 2000.0
MINIMUM_CONTEXT_WEIGHT = 0.01
CONTEXT_DISTANCE_SCALE = 0.10
PROVISIONAL_SCORE_ALPHA = 0.20
MINIMUM_RESIDUAL_VARIANCE = 1.0
FIXED_NEURAL_HIDDEN_UNITS = 48
FIXED_NEURAL_WEIGHT_SCALE = 1.5
FEATURE_DISPERSION_PENALTY = 0.25
FIXED_NEURAL_WEIGHT_PHASE = 1.61803398875
FIXED_NEURAL_BIAS_PHASE = 0.73
FIXED_NEURAL_TRANSFORM = "fixed-neural-v1"
IDENTITY_TRANSFORM = "identity"
MAXIMUM_OUTPUT_EXPERTS = 32
OUTPUT_EXPERT_WEIGHT = 0.80
OUTPUT_ROUTING_SCALE = 0.03
OUTPUT_REGION_RECORD_LENGTH = 5


def _record_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError
    return int(value)


def _record_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError
    return float(value)


def _record_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError
    return value


def _decode_output_experts(
    value: object,
) -> tuple[DisplayTopology | None, dict[str, CalibrationModel]]:
    """Decode one bounded, flat set of persisted output experts."""
    if value is None:
        return None, {}
    entries = _record_list(value)
    if not entries or len(entries) > MAXIMUM_OUTPUT_EXPERTS:
        msg = "stored output expert count is invalid"
        raise ValueError(msg)
    regions: list[DisplayRegion] = []
    experts: dict[str, CalibrationModel] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError
        raw_region = _record_list(entry["region"])
        if len(raw_region) != OUTPUT_REGION_RECORD_LENGTH or not isinstance(raw_region[0], str):
            msg = "stored output expert region is invalid"
            raise ValueError(msg)
        region = DisplayRegion(
            raw_region[0],
            _record_integer(raw_region[1]),
            _record_integer(raw_region[2]),
            _record_integer(raw_region[3]),
            _record_integer(raw_region[4]),
        )
        raw_model = entry["model"]
        regions.append(region)
        if raw_model is None:
            continue
        if not isinstance(raw_model, dict):
            raise TypeError
        expert = CalibrationModel.from_record(raw_model)
        if expert.output_expert_count:
            msg = "stored output experts must not be nested"
            raise ValueError(msg)
        experts[region.region_id] = expert
    return DisplayTopology(tuple(regions)), experts


if TYPE_CHECKING:
    from collections.abc import Sequence

    from gazeebo.contracts import FeatureVector
    from gazeebo.state import CursorNoiseSummary


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One gaze feature vector paired with a visible global target."""

    features: FeatureVector
    target: Point
    weight: float = 1.0
    context: tuple[float, ...] = ()
    feature_dispersion: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Require every target to retain finite, strictly positive influence."""
        dispersion_valid = not self.feature_dispersion or (
            len(self.feature_dispersion) == len(self.features)
            and all(math.isfinite(value) and value >= 0.0 for value in self.feature_dispersion)
        )
        if (
            not math.isfinite(self.weight)
            or self.weight <= 0.0
            or not all(math.isfinite(value) for value in self.context)
            or not dispersion_valid
        ):
            msg = (
                "calibration sample weight must be finite and positive; "
                "feature dispersion must be valid"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CalibrationWork:
    """Deterministic target-proportional work performed by one fit state."""

    statistic_updates: int = 0
    score_predictions: int = 0

    @property
    def total(self) -> int:
        """Return the count used by non-timing complexity checks."""
        return self.statistic_updates + self.score_predictions


def grouped_folds(sample_count: int) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Assign every target to one deterministic held-out group."""
    if sample_count < MINIMUM_MODEL_SELECTION_SAMPLES:
        msg = "grouped folds require enough model-selection samples"
        raise ValueError(msg)
    fold_count = min(MAXIMUM_CROSS_VALIDATION_FOLDS, sample_count)
    indices = range(sample_count)
    return tuple(
        (
            tuple(index for index in indices if index % fold_count != fold),
            tuple(index for index in indices if index % fold_count == fold),
        )
        for fold in range(fold_count)
    )


def target_fit_weight(noise: CursorNoiseSummary | None) -> float:
    """Give every target bounded positive influence while downweighting noise."""
    if noise is None:
        return MISSING_NOISE_WEIGHT
    return max(
        MINIMUM_TARGET_WEIGHT,
        1.0 / (1.0 + noise.p95_radial_spread / NOISE_WEIGHT_SCALE),
    )


def _routing_context_reference(
    routing_contexts: Sequence[tuple[float, ...]] | None,
) -> tuple[float, ...] | None:
    """Reduce current-run contexts once so target weighting remains linear."""
    if not routing_contexts:
        return None
    dimensions = len(routing_contexts[0])
    compatible = tuple(context for context in routing_contexts if len(context) == dimensions)
    if not compatible:
        return None
    return tuple(
        math.fsum(context[index] for context in compatible) / len(compatible)
        for index in range(dimensions)
    )


def _weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Return one deterministic weighted percentile with positive influence."""
    if values.shape != weights.shape or values.size == 0 or not 0.0 <= quantile <= 1.0:
        msg = "weighted percentile inputs are invalid"
        raise ValueError(msg)
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    if cumulative[-1] <= 0.0:
        msg = "weighted percentile requires positive total weight"
        raise ValueError(msg)
    return float(np.interp(quantile * cumulative[-1], cumulative, ordered_values))


def _context_fit_weight(
    context: tuple[float, ...],
    routing_context: tuple[float, ...] | None,
) -> float:
    """Keep all contexts positive while emphasizing this run's camera geometry."""
    if not routing_context or not context or len(context) != len(routing_context):
        return 1.0
    difference = np.asarray(context, dtype=np.float64) - np.asarray(
        routing_context,
        dtype=np.float64,
    )
    posture_dimensions = min(7, len(difference))
    posture = float(np.sqrt(np.mean(difference[:posture_dimensions] ** 2)))
    illumination = (
        0.25 * float(np.sqrt(np.mean(difference[posture_dimensions:] ** 2)))
        if posture_dimensions < len(difference)
        else 0.0
    )
    distance = (posture + illumination) / CONTEXT_DISTANCE_SCALE
    return MINIMUM_CONTEXT_WEIGHT + (1.0 - MINIMUM_CONTEXT_WEIGHT) * math.exp(-0.5 * distance**2)


def aggregate_feature_dispersion(vectors: Sequence[FeatureVector]) -> FeatureVector:
    """Reduce frame noise to bounded per-feature robust standard deviations."""
    center = np.asarray(aggregate_features(vectors), dtype=np.float64)
    matrix = np.asarray(vectors, dtype=np.float64)
    absolute_deviation = np.abs(matrix - center)
    return tuple(float(value) for value in 1.4826 * np.median(absolute_deviation, axis=0))


def aggregate_features(vectors: Sequence[FeatureVector]) -> FeatureVector:
    """Reduce one target's frame samples to a robust feature vector."""
    if not vectors:
        msg = "at least one feature vector is required"
        raise ValueError(msg)
    feature_count = len(vectors[0])
    if feature_count == 0 or any(len(vector) != feature_count for vector in vectors):
        msg = "feature vectors must have one consistent non-zero length"
        raise ValueError(msg)
    matrix = np.asarray(vectors, dtype=np.float64)
    return tuple(float(value) for value in np.median(matrix, axis=0))


@dataclass(frozen=True, slots=True)
class _Candidate:
    ridge: float
    feature_indices: tuple[int, ...]
    feature_name: str


def _fixed_neural_features(features: np.ndarray) -> np.ndarray:
    """Append one deterministic frozen head/face layer for Bayesian output fitting."""
    if features.shape[-1] < PUPIL_OPTIONAL_FEATURE_COUNT:
        return features
    head = features[..., HEAD_FEATURE_INDICES]
    hidden = np.arange(1, FIXED_NEURAL_HIDDEN_UNITS + 1, dtype=np.float64)[:, np.newaxis]
    inputs = np.arange(1, len(HEAD_FEATURE_INDICES) + 1, dtype=np.float64)[np.newaxis, :]
    weights = np.sin(hidden * inputs * FIXED_NEURAL_WEIGHT_PHASE) * FIXED_NEURAL_WEIGHT_SCALE
    bias = (
        np.cos(
            np.arange(1, FIXED_NEURAL_HIDDEN_UNITS + 1, dtype=np.float64) * FIXED_NEURAL_BIAS_PHASE
        )
        * 0.5
    )
    projected = np.tanh(head @ weights.T + bias)
    return np.concatenate((features, projected), axis=-1)


def _fixed_neural_dispersion(
    features: np.ndarray,
    dispersion: np.ndarray,
) -> np.ndarray:
    """Propagate diagonal input dispersion through the frozen head layer."""
    if features.shape[-1] < PUPIL_OPTIONAL_FEATURE_COUNT:
        return dispersion
    head = features[..., HEAD_FEATURE_INDICES]
    head_dispersion = dispersion[..., HEAD_FEATURE_INDICES]
    hidden = np.arange(1, FIXED_NEURAL_HIDDEN_UNITS + 1, dtype=np.float64)[:, np.newaxis]
    inputs = np.arange(1, len(HEAD_FEATURE_INDICES) + 1, dtype=np.float64)[np.newaxis, :]
    weights = np.sin(hidden * inputs * FIXED_NEURAL_WEIGHT_PHASE) * FIXED_NEURAL_WEIGHT_SCALE
    bias = (
        np.cos(
            np.arange(1, FIXED_NEURAL_HIDDEN_UNITS + 1, dtype=np.float64) * FIXED_NEURAL_BIAS_PHASE
        )
        * 0.5
    )
    projected = np.tanh(head @ weights.T + bias)
    jacobian = (1.0 - projected**2)[:, np.newaxis] * weights
    hidden_variance = (jacobian**2) @ (head_dispersion**2)
    return np.concatenate((dispersion, np.sqrt(np.maximum(hidden_variance, 0.0))))


def _transform_features(features: np.ndarray, transform: str) -> np.ndarray:
    if transform == FIXED_NEURAL_TRANSFORM:
        return _fixed_neural_features(features)
    return features


@dataclass(slots=True)
class _AffineStatistics:
    """Fixed-dimensional weighted moments for exact affine ridge fitting."""

    weight_sum: float
    feature_sum: np.ndarray
    feature_cross: np.ndarray
    target_sum: np.ndarray
    target_square_sum: np.ndarray
    feature_target_cross: np.ndarray
    sample_count: int

    @classmethod
    def empty(cls, feature_count: int) -> _AffineStatistics:
        return cls(
            0.0,
            np.zeros(feature_count, dtype=np.float64),
            np.zeros((feature_count, feature_count), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros((feature_count, 2), dtype=np.float64),
            0,
        )

    def add(
        self,
        features: np.ndarray,
        target: np.ndarray,
        weight: float,
        dispersion: np.ndarray | None = None,
    ) -> None:
        """Add one positive target and its bounded input noise in constant work."""
        self.weight_sum += weight
        self.feature_sum += weight * features
        self.feature_cross += weight * np.outer(features, features)
        if dispersion is not None:
            if dispersion.shape != features.shape or np.any(dispersion < 0.0):
                msg = "feature dispersion must match the model feature vector"
                raise ValueError(msg)
            self.feature_cross += weight * FEATURE_DISPERSION_PENALTY * np.diag(dispersion**2)
        self.target_sum += weight * target
        self.target_square_sum += weight * target**2
        self.feature_target_cross += weight * np.outer(features, target)
        self.sample_count += 1

    def without(self, held_out: _AffineStatistics) -> _AffineStatistics:
        """Derive one grouped-fold training state without rescanning targets."""
        return _AffineStatistics(
            self.weight_sum - held_out.weight_sum,
            self.feature_sum - held_out.feature_sum,
            self.feature_cross - held_out.feature_cross,
            self.target_sum - held_out.target_sum,
            self.target_square_sum - held_out.target_square_sum,
            self.feature_target_cross - held_out.feature_target_cross,
            self.sample_count - held_out.sample_count,
        )

    def fit(
        self,
        candidate: _Candidate,
        input_feature_count: int,
    ) -> CalibrationModel:
        """Solve one fixed-size Bayesian output system from sufficient statistics."""
        if self.sample_count < MINIMUM_CALIBRATION_SAMPLES or self.weight_sum <= 0.0:
            msg = "at least three calibration samples are required"
            raise ValueError(msg)
        indices = np.asarray(candidate.feature_indices, dtype=np.intp)
        feature_sum = self.feature_sum[indices]
        feature_cross = self.feature_cross[np.ix_(indices, indices)]
        feature_target_cross = self.feature_target_cross[indices]
        feature_mean = feature_sum / self.weight_sum
        centered_cross = feature_cross - np.outer(feature_sum, feature_sum) / self.weight_sum
        variance = np.maximum(np.diag(centered_cross) / self.weight_sum, 0.0)
        feature_scale = np.sqrt(variance)
        feature_scale = np.where(feature_scale < MINIMUM_FEATURE_SCALE, 1.0, feature_scale)
        normalized_cross = centered_cross / np.outer(feature_scale, feature_scale)
        normalized_sum = (feature_sum - self.weight_sum * feature_mean) / feature_scale
        normalized_target_cross = (
            feature_target_cross - np.outer(feature_mean, self.target_sum)
        ) / feature_scale[:, np.newaxis]

        dimension = len(candidate.feature_indices) + 1
        system = np.zeros((dimension, dimension), dtype=np.float64)
        system[0, 0] = self.weight_sum
        system[0, 1:] = normalized_sum
        system[1:, 0] = normalized_sum
        system[1:, 1:] = normalized_cross
        penalty = np.eye(dimension, dtype=np.float64) * candidate.ridge
        penalty[0, 0] = 0.0
        right_hand = np.vstack((self.target_sum, normalized_target_cross))
        posterior_precision = system + penalty
        coefficients = np.linalg.solve(posterior_precision, right_hand)
        posterior_covariance = np.linalg.inv(posterior_precision)
        residual_sum = (
            self.target_square_sum
            - 2.0 * np.sum(coefficients * right_hand, axis=0)
            + np.sum(coefficients * (system @ coefficients), axis=0)
        )
        degrees_of_freedom = max(self.weight_sum - dimension, 1.0)
        residual_variance = np.maximum(
            residual_sum / degrees_of_freedom,
            MINIMUM_RESIDUAL_VARIANCE,
        )
        return CalibrationModel(
            coefficients,
            feature_mean,
            feature_scale,
            coefficient_covariance=posterior_covariance,
            residual_variance=residual_variance,
            input_feature_count=input_feature_count,
            feature_indices=candidate.feature_indices,
            feature_transform=(
                FIXED_NEURAL_TRANSFORM
                if len(self.feature_sum) > input_feature_count
                else IDENTITY_TRANSFORM
            ),
            feature_name=candidate.feature_name,
            sample_count=self.sample_count,
        )


@dataclass(frozen=True, slots=True)
class _PreparedSamples:
    features: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    statistics: _AffineStatistics
    held_out_statistics: tuple[_AffineStatistics, ...]
    held_out_indices: tuple[np.ndarray, ...]
    statistic_updates: int


class CalibrationModel:
    """Map gaze features to coordinates with linear-work model selection."""

    def __init__(  # noqa: PLR0913
        self,
        coefficients: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        *,
        kind: str = "affine",
        support: np.ndarray | None = None,
        gamma: float = 0.0,
        target_offset: np.ndarray | None = None,
        input_feature_count: int | None = None,
        feature_indices: tuple[int, ...] | None = None,
        feature_name: str = "all",
        feature_transform: str = IDENTITY_TRANSFORM,
        head_fallback: CalibrationModel | None = None,
        sample_count: int = 0,
        coefficient_covariance: np.ndarray | None = None,
        residual_variance: np.ndarray | None = None,
        output_topology: DisplayTopology | None = None,
        output_experts: dict[str, CalibrationModel] | None = None,
    ) -> None:
        """Store one fitted, process-local estimator."""
        self._coefficients = coefficients
        self._feature_mean = feature_mean
        self._feature_scale = feature_scale
        self._input_feature_count = (
            len(feature_mean) if input_feature_count is None else input_feature_count
        )
        self._feature_indices = (
            tuple(range(len(feature_mean))) if feature_indices is None else feature_indices
        )
        self._kind = kind
        self._feature_name = feature_name
        self._feature_transform = feature_transform
        self._support = support
        self._gamma = gamma
        self._target_offset = (
            np.zeros(2, dtype=np.float64) if target_offset is None else target_offset
        )
        self._head_fallback = head_fallback
        self._sample_count = sample_count
        self._coefficient_covariance = coefficient_covariance
        self._residual_variance = residual_variance
        self._output_topology = output_topology
        self._output_experts = {} if output_experts is None else dict(output_experts)
        if (output_topology is None) != (not self._output_experts):
            msg = "output topology and experts must be configured together"
            raise ValueError(msg)
        if output_topology is not None and (
            len(output_topology.regions) > MAXIMUM_OUTPUT_EXPERTS
            or not set(self._output_experts)
            <= {region.region_id for region in output_topology.regions}
        ):
            msg = "output experts must match their bounded topology"
            raise ValueError(msg)

    @property
    def kind(self) -> str:
        """Name the fixed-dimensional estimator and selected feature set."""
        base = self._kind if self._feature_name == "all" else f"{self._kind}/{self._feature_name}"
        return f"output-mixture/{base}" if self._output_experts else base

    @property
    def output_expert_count(self) -> int:
        """Return the bounded number of local display estimators."""
        return len(self._output_experts)

    @property
    def sample_count(self) -> int:
        """Return how many target aggregates influenced this final fit."""
        return self._sample_count

    @property
    def supports_extrapolation(self) -> bool:
        """Return whether predictions can extend beyond observed feature support."""
        return self._kind == "affine"

    @staticmethod
    def _feature_sets(feature_count: int) -> tuple[tuple[tuple[int, ...], str], ...]:
        all_features = tuple(range(feature_count))
        if feature_count >= PUPIL_OPTIONAL_FEATURE_COUNT:
            return (
                *HEAD_FEATURE_SETS,
                (all_features, "head+face+pupils"),
                ((0, 1, 2, 3, *STABLE_HEAD_FEATURE_INDICES), "pupils+head+face"),
                ((8, 9, *STABLE_HEAD_FEATURE_INDICES), "binocular+head+face"),
                (
                    (
                        *HEAD_FEATURE_INDICES,
                        *range(
                            feature_count,
                            feature_count + FIXED_NEURAL_HIDDEN_UNITS,
                        ),
                    ),
                    "neural-head+face",
                ),
            )
        if feature_count >= BINOCULAR_GAZE_FEATURE_COUNT:
            return (
                (all_features, "all"),
                ((8, 9), "binocular"),
                ((4, 5, 8, 9), "binocular+pose"),
                ((6, 7, 8, 9), "binocular+position"),
                ((4, 5, 6, 7, 8, 9), "binocular+context"),
                ((0, 1, 2, 3), "pupils"),
                ((0, 1, 2, 3, 4, 5), "pupils+pose"),
                ((0, 1, 2, 3, 6, 7), "pupils+position"),
            )
        if feature_count >= BASE_GAZE_FEATURE_COUNT:
            return (
                (all_features, "all"),
                ((0, 1, 2, 3), "pupils"),
                ((0, 1, 2, 3, 4, 5), "pupils+pose"),
                ((0, 1, 2, 3, 6, 7), "pupils+position"),
            )
        return ((all_features, "all"),)

    @classmethod
    def _candidates(
        cls,
        feature_count: int,
        sample_count: int,
        ridge: float,
    ) -> tuple[_Candidate, ...]:
        if sample_count < MINIMUM_MODEL_SELECTION_SAMPLES:
            return (_Candidate(ridge, tuple(range(feature_count)), "all"),)
        regularization = tuple(
            sorted({max(ridge * factor, 1e-10) for factor in (0.1, 1.0, 10.0, 100.0)})
        )
        return tuple(
            _Candidate(value, indices, name)
            for indices, name in cls._feature_sets(feature_count)
            for value in regularization
        )

    @classmethod
    def _prepare(
        cls,
        samples: Sequence[CalibrationSample],
        routing_contexts: Sequence[tuple[float, ...]] | None,
    ) -> _PreparedSamples:
        if len(samples) < MINIMUM_CALIBRATION_SAMPLES:
            msg = "at least three calibration samples are required"
            raise ValueError(msg)
        feature_count = len(samples[0].features)
        if feature_count == 0 or any(len(sample.features) != feature_count for sample in samples):
            msg = "calibration feature vectors must have one consistent non-zero length"
            raise ValueError(msg)
        features = np.empty((len(samples), feature_count), dtype=np.float64)
        targets = np.empty((len(samples), 2), dtype=np.float64)
        weights = np.empty(len(samples), dtype=np.float64)
        model_feature_count = len(_fixed_neural_features(np.zeros(feature_count, dtype=np.float64)))
        statistics = _AffineStatistics.empty(model_feature_count)
        routing_context = _routing_context_reference(routing_contexts)
        fold_count = min(MAXIMUM_CROSS_VALIDATION_FOLDS, len(samples))
        held_out_statistics = tuple(
            _AffineStatistics.empty(model_feature_count) for _ in range(fold_count)
        )
        held_out_lists: list[list[int]] = [[] for _ in range(fold_count)]
        for index, sample in enumerate(samples):
            feature_row = np.asarray(sample.features, dtype=np.float64)
            target_row = np.asarray((sample.target.x, sample.target.y), dtype=np.float64)
            weight = sample.weight * _context_fit_weight(sample.context, routing_context)
            features[index] = feature_row
            targets[index] = target_row
            weights[index] = weight
            model_features = _fixed_neural_features(feature_row)
            model_dispersion = (
                _fixed_neural_dispersion(
                    feature_row,
                    np.asarray(sample.feature_dispersion, dtype=np.float64),
                )
                if sample.feature_dispersion
                else None
            )
            statistics.add(model_features, target_row, weight, model_dispersion)
            fold = index % fold_count
            held_out_statistics[fold].add(
                model_features,
                target_row,
                weight,
                model_dispersion,
            )
            held_out_lists[fold].append(index)
        return _PreparedSamples(
            features,
            targets,
            weights,
            statistics,
            held_out_statistics,
            tuple(np.asarray(indices, dtype=np.intp) for indices in held_out_lists),
            len(samples) * 2,
        )

    @classmethod
    def _candidate_errors(
        cls,
        prepared: _PreparedSamples,
        candidates: Sequence[_Candidate],
    ) -> tuple[dict[_Candidate, float], int]:
        errors: dict[_Candidate, float] = {}
        predictions = 0
        for candidate in candidates:
            candidate_errors: list[np.ndarray] = []
            candidate_weights: list[np.ndarray] = []
            for held_statistics, held_indices in zip(
                prepared.held_out_statistics,
                prepared.held_out_indices,
                strict=True,
            ):
                training_statistics = prepared.statistics.without(held_statistics)
                model = training_statistics.fit(candidate, prepared.features.shape[1])
                held_features = prepared.features[held_indices]
                predicted = model.predict_many(held_features)
                residual = predicted - prepared.targets[held_indices]
                candidate_errors.append(np.hypot(residual[:, 0], residual[:, 1]))
                candidate_weights.append(prepared.weights[held_indices])
                predictions += len(held_indices)
            values = np.concatenate(candidate_errors)
            weights = np.concatenate(candidate_weights)
            errors[candidate] = float(
                _weighted_percentile(values, weights, 0.5)
                + 0.25 * _weighted_percentile(values, weights, 0.9)
            )
        return errors, predictions

    @classmethod
    def _fit_prepared(
        cls,
        prepared: _PreparedSamples,
        *,
        ridge: float,
    ) -> tuple[
        CalibrationModel,
        CalibrationWork,
        _Candidate,
        _Candidate | None,
        dict[_Candidate, float],
    ]:
        candidates = cls._candidates(
            prepared.features.shape[1],
            len(prepared.features),
            ridge,
        )
        score_predictions = 0
        if len(prepared.features) >= MINIMUM_MODEL_SELECTION_SAMPLES:
            errors, score_predictions = cls._candidate_errors(prepared, candidates)
            candidate = min(candidates, key=errors.__getitem__)
        else:
            errors = {}
            candidate = candidates[0]
        head_candidate: _Candidate | None = None
        if prepared.features.shape[1] >= PUPIL_OPTIONAL_FEATURE_COUNT:
            head_candidates = tuple(
                option
                for option in candidates
                if option.feature_indices in {indices for indices, _name in HEAD_FEATURE_SETS}
                or option.feature_name == "neural-head+face"
            )
            if head_candidates and errors:
                head_candidate = min(head_candidates, key=errors.__getitem__)
            else:
                head_candidate = _Candidate(ridge, HEAD_FEATURE_INDICES, "head+face")
        model = prepared.statistics.fit(candidate, prepared.features.shape[1])
        if head_candidate is not None:
            model.with_head_fallback(
                prepared.statistics.fit(head_candidate, prepared.features.shape[1])
            )
        return (
            model,
            CalibrationWork(prepared.statistic_updates, score_predictions),
            candidate,
            head_candidate,
            errors,
        )

    @classmethod
    def fit(
        cls,
        samples: Sequence[CalibrationSample],
        *,
        ridge: float = 0.1,
        routing_contexts: Sequence[tuple[float, ...]] | None = None,
        topology: DisplayTopology | None = None,
    ) -> CalibrationModel:
        """Select with grouped folds and fit all targets in O(n) total work."""
        model, _work = cls.fit_with_work(
            samples,
            ridge=ridge,
            routing_contexts=routing_contexts,
            topology=topology,
        )
        return model

    @classmethod
    def fit_with_work(
        cls,
        samples: Sequence[CalibrationSample],
        *,
        ridge: float = 0.1,
        routing_contexts: Sequence[tuple[float, ...]] | None = None,
        topology: DisplayTopology | None = None,
    ) -> tuple[CalibrationModel, CalibrationWork]:
        """Fit and expose deterministic work counts for complexity verification."""
        if ridge < 0.0:
            msg = "ridge regularization must be non-negative"
            raise ValueError(msg)
        prepared = cls._prepare(samples, routing_contexts)
        model, work, _candidate, _head_candidate, _errors = cls._fit_prepared(
            prepared,
            ridge=ridge,
        )
        if topology is None:
            return model, work
        grouped = cls._samples_by_output(samples, topology)
        experts: dict[str, CalibrationModel] = {}
        for region in topology.regions:
            local = grouped[region.region_id]
            if len(local) < MINIMUM_CALIBRATION_SAMPLES:
                continue
            expert, expert_work = cls.fit_with_work(
                local,
                ridge=ridge,
                routing_contexts=routing_contexts,
            )
            experts[region.region_id] = expert
            work = CalibrationWork(
                work.statistic_updates + expert_work.statistic_updates,
                work.score_predictions + expert_work.score_predictions,
            )
        if experts:
            model.with_output_experts(topology, experts)
        return model, work

    @staticmethod
    def _samples_by_output(
        samples: Sequence[CalibrationSample],
        topology: DisplayTopology,
    ) -> dict[str, list[CalibrationSample]]:
        """Partition labels once by their nearest authorized output."""
        if len(topology.regions) > MAXIMUM_OUTPUT_EXPERTS:
            msg = f"at most {MAXIMUM_OUTPUT_EXPERTS} output experts are supported"
            raise ValueError(msg)
        grouped: dict[str, list[CalibrationSample]] = {
            region.region_id: [] for region in topology.regions
        }
        for sample in samples:
            grouped[topology.locate(sample.target).region_id].append(sample)
        return grouped

    @classmethod
    def fit_head(
        cls,
        samples: Sequence[CalibrationSample],
        *,
        ridge: float = 0.1,
        topology: DisplayTopology | None = None,
    ) -> CalibrationModel:
        """Fit a head/face-only fallback from pupil-optional target records."""
        prepared = cls._prepare(samples, None)
        if prepared.features.shape[1] < PUPIL_OPTIONAL_FEATURE_COUNT:
            msg = "head fallback requires pupil-optional feature records"
            raise ValueError(msg)
        model = prepared.statistics.fit(
            _Candidate(ridge, HEAD_FEATURE_INDICES, "head+face"),
            prepared.features.shape[1],
        )
        if topology is not None:
            experts = {
                region_id: cls.fit_head(local, ridge=ridge)
                for region_id, local in cls._samples_by_output(samples, topology).items()
                if len(local) >= MINIMUM_CALIBRATION_SAMPLES
            }
            if experts:
                model.with_output_experts(topology, experts)
        return model

    def with_head_fallback(self, model: CalibrationModel) -> None:
        """Attach the head/face-only estimator paired with this model."""
        self._head_fallback = model

    def with_output_experts(
        self,
        topology: DisplayTopology,
        experts: dict[str, CalibrationModel],
    ) -> None:
        """Attach bounded local estimators while retaining this global model."""
        if (
            not experts
            or len(topology.regions) > MAXIMUM_OUTPUT_EXPERTS
            or not set(experts) <= {region.region_id for region in topology.regions}
            or any(expert.output_expert_count for expert in experts.values())
        ):
            msg = "output experts must be flat and match their bounded topology"
            raise ValueError(msg)
        self._output_topology = topology
        self._output_experts = dict(experts)

    def to_record(self) -> dict[str, object]:
        """Serialize fitted coefficients without retaining training observations."""
        return {
            "kind": self._kind,
            "coefficients": self._coefficients.tolist(),
            "feature_mean": self._feature_mean.tolist(),
            "feature_scale": self._feature_scale.tolist(),
            "support": None if self._support is None else self._support.tolist(),
            "gamma": self._gamma,
            "target_offset": self._target_offset.tolist(),
            "input_feature_count": self._input_feature_count,
            "feature_indices": list(self._feature_indices),
            "feature_name": self._feature_name,
            "feature_transform": self._feature_transform,
            "sample_count": self._sample_count,
            "posterior_covariance": (
                None
                if self._coefficient_covariance is None
                else self._coefficient_covariance.tolist()
            ),
            "residual_variance": (
                None if self._residual_variance is None else self._residual_variance.tolist()
            ),
            "head_fallback": (
                None if self._head_fallback is None else self._head_fallback.to_record()
            ),
            "output_experts": (
                None
                if self._output_topology is None
                else [
                    {
                        "region": [
                            region.region_id,
                            region.x,
                            region.y,
                            region.width,
                            region.height,
                        ],
                        "model": (
                            None
                            if region.region_id not in self._output_experts
                            else self._output_experts[region.region_id].to_record()
                        ),
                    }
                    for region in self._output_topology.regions
                ]
            ),
        }

    @classmethod
    def from_record(cls, record: dict[str, object]) -> CalibrationModel:
        """Restore validated affine or legacy RBF coefficients from the private store."""
        try:
            kind = str(record["kind"])
            coefficients = np.asarray(record["coefficients"], dtype=np.float64)
            feature_mean = np.asarray(record["feature_mean"], dtype=np.float64)
            feature_scale = np.asarray(record["feature_scale"], dtype=np.float64)
            raw_support = record.get("support")
            support = None if raw_support is None else np.asarray(raw_support, dtype=np.float64)
            gamma = _record_number(record["gamma"])
            target_offset = np.asarray(record["target_offset"], dtype=np.float64)
            input_feature_count = _record_integer(record["input_feature_count"])
            raw_indices = _record_list(record["feature_indices"])
            feature_indices = tuple(_record_integer(value) for value in raw_indices)
            feature_name = str(record["feature_name"])
            feature_transform = str(record.get("feature_transform", IDENTITY_TRANSFORM))
            sample_count = _record_integer(record.get("sample_count", 0))
            raw_covariance = record.get("posterior_covariance")
            coefficient_covariance = (
                None if raw_covariance is None else np.asarray(raw_covariance, dtype=np.float64)
            )
            raw_residual_variance = record.get("residual_variance")
            residual_variance = (
                None
                if raw_residual_variance is None
                else np.asarray(raw_residual_variance, dtype=np.float64)
            )
            raw_head_fallback = record.get("head_fallback")
            head_fallback = (
                None
                if raw_head_fallback is None
                else cls.from_record(cast("dict[str, object]", raw_head_fallback))
            )
            output_topology, output_experts = _decode_output_experts(record.get("output_experts"))
        except (KeyError, TypeError, ValueError) as error:
            msg = "stored calibration model is malformed"
            raise ValueError(msg) from error
        arrays = [coefficients, feature_mean, feature_scale, target_offset]
        if support is not None:
            arrays.append(support)
        if coefficient_covariance is not None:
            arrays.append(coefficient_covariance)
        if residual_variance is not None:
            arrays.append(residual_variance)
        if (
            kind not in {"affine", "rbf"}
            or not feature_indices
            or feature_transform not in {IDENTITY_TRANSFORM, FIXED_NEURAL_TRANSFORM}
            or input_feature_count <= 0
            or sample_count < 0
            or any(not np.all(np.isfinite(array)) for array in arrays)
            or np.any(feature_scale <= 0.0)
            or (kind == "rbf" and support is None)
            or (kind == "affine" and support is not None)
            or ((coefficient_covariance is None) != (residual_variance is None))
            or (
                coefficient_covariance is not None
                and coefficient_covariance.shape
                != (len(feature_indices) + 1, len(feature_indices) + 1)
            )
            or (residual_variance is not None and residual_variance.shape != (2,))
            or (residual_variance is not None and np.any(residual_variance <= 0.0))
        ):
            msg = "stored calibration model is invalid"
            raise ValueError(msg)
        return cls(
            coefficients,
            feature_mean,
            feature_scale,
            kind=kind,
            support=support,
            gamma=gamma,
            target_offset=target_offset,
            input_feature_count=input_feature_count,
            feature_indices=feature_indices,
            feature_name=feature_name,
            feature_transform=feature_transform,
            head_fallback=head_fallback,
            sample_count=sample_count,
            coefficient_covariance=coefficient_covariance,
            residual_variance=residual_variance,
            output_topology=output_topology,
            output_experts=output_experts,
        )

    def predict_many(self, features: np.ndarray) -> np.ndarray:
        """Predict a validated matrix for linear grouped-fold scoring."""
        if self._output_experts:
            if features.ndim != FEATURE_MATRIX_DIMENSIONS:
                msg = "gaze feature matrix does not match calibration"
                raise ValueError(msg)
            points = [self.predict(tuple(float(value) for value in row)) for row in features]
            return np.asarray(
                [(point.x, point.y) for point in points],
                dtype=np.float64,
            )
        if (
            features.ndim != FEATURE_MATRIX_DIMENSIONS
            or features.shape[1] < self._input_feature_count
        ):
            msg = "gaze feature matrix does not match calibration"
            raise ValueError(msg)
        model_features = _transform_features(features, self._feature_transform)
        selected = model_features[:, self._feature_indices]
        normalized = (selected - self._feature_mean) / self._feature_scale
        if self._kind == "rbf":
            if self._support is None:
                msg = "RBF calibration support is unavailable"
                raise RuntimeError(msg)
            distances = np.sum(
                (normalized[:, np.newaxis, :] - self._support[np.newaxis, :, :]) ** 2,
                axis=2,
            )
            return cast(
                "np.ndarray",
                self._target_offset + np.exp(-self._gamma * distances) @ self._coefficients,
            )
        design = np.column_stack((np.ones(len(features)), normalized))
        return cast("np.ndarray", design @ self._coefficients)

    def predict(
        self,
        features: FeatureVector,
        context: tuple[float, ...] | None = None,
    ) -> Point:
        """Predict one global point with smooth bounded output specialization."""
        del context
        rough = self._predict_single(features)
        weighted = self._output_predictions(features, rough)
        if not weighted:
            return rough
        local_weight = OUTPUT_EXPERT_WEIGHT * math.fsum(
            weight for _name, weight, _point in weighted
        )
        x = (1.0 - local_weight) * rough.x
        y = (1.0 - local_weight) * rough.y
        for _name, weight, point in weighted:
            x += OUTPUT_EXPERT_WEIGHT * weight * point.x
            y += OUTPUT_EXPERT_WEIGHT * weight * point.y
        return Point(x, y)

    def _predict_single(self, features: FeatureVector) -> Point:
        """Predict with this estimator without applying attached output experts."""
        if (
            len(features) >= PUPIL_OPTIONAL_FEATURE_COUNT
            and features[PUPIL_AVAILABILITY_INDEX] < PUPIL_AVAILABLE_THRESHOLD
            and self._head_fallback is not None
        ):
            return self._head_fallback._predict_single(features)  # noqa: SLF001
        if len(features) < self._input_feature_count:
            msg = "gaze feature vector does not match calibration"
            raise ValueError(msg)
        compatible = np.asarray(features[: self._input_feature_count], dtype=np.float64)
        model_features = _transform_features(compatible, self._feature_transform)
        selected = model_features[list(self._feature_indices)]
        normalized = (selected - self._feature_mean) / self._feature_scale
        if self._kind == "rbf":
            if self._support is None:
                msg = "RBF calibration support is unavailable"
                raise RuntimeError(msg)
            distances = np.sum((self._support - normalized) ** 2, axis=1)
            row = np.exp(-self._gamma * distances)
            prediction = self._target_offset + row @ self._coefficients
        else:
            row = np.concatenate(([1.0], normalized))
            prediction = row @ self._coefficients
        return Point(float(prediction[0]), float(prediction[1]))

    def _output_predictions(
        self,
        features: FeatureVector,
        rough: Point,
    ) -> tuple[tuple[str, float, Point], ...]:
        """Return continuous geometry weights and clipped local predictions."""
        if self._output_topology is None:
            return ()
        routed: list[tuple[DisplayRegion, float]] = []
        for region in self._output_topology.regions:
            horizontal = max(region.x - rough.x, 0.0, rough.x - region.right)
            vertical = max(region.y - rough.y, 0.0, rough.y - region.bottom)
            scale = OUTPUT_ROUTING_SCALE * min(region.width, region.height)
            score = -0.5 * (math.hypot(horizontal, vertical) / scale) ** 2
            routed.append((region, score))
        maximum = max(score for _region, score in routed)
        normalizer = math.fsum(math.exp(score - maximum) for _region, score in routed)
        predictions: list[tuple[str, float, Point]] = []
        for region, score in routed:
            expert = self._output_experts.get(region.region_id)
            if expert is None:
                continue
            prediction = expert._predict_single(features)  # noqa: SLF001
            clipped = Point(
                min(max(prediction.x, region.x), math.nextafter(float(region.right), region.x)),
                min(max(prediction.y, region.y), math.nextafter(float(region.bottom), region.y)),
            )
            predictions.append((region.region_id, math.exp(score - maximum) / normalizer, clipped))
        return tuple(predictions)

    def predict_with_uncertainty(
        self,
        features: FeatureVector,
        context: tuple[float, ...] | None = None,
    ) -> tuple[Point, float | None]:
        """Return the posterior mean and radial mixture uncertainty in pixels."""
        del context
        rough, global_uncertainty = self._predict_single_with_uncertainty(features)
        weighted = self._output_predictions(features, rough)
        if not weighted:
            return rough, global_uncertainty
        point = self.predict(features)
        if global_uncertainty is None:
            return point, None
        local_weight = OUTPUT_EXPERT_WEIGHT * math.fsum(
            weight for _name, weight, _point in weighted
        )
        components = [(1.0 - local_weight, rough, global_uncertainty)]
        for name, weight, prediction in weighted:
            expert = self._output_experts[name]
            _expert_point, uncertainty = expert._predict_single_with_uncertainty(  # noqa: SLF001
                features
            )
            if uncertainty is None:
                return point, None
            components.append((OUTPUT_EXPERT_WEIGHT * weight, prediction, uncertainty))
        variance = math.fsum(
            weight * (uncertainty**2 + (component.x - point.x) ** 2 + (component.y - point.y) ** 2)
            for weight, component, uncertainty in components
        )
        return point, math.sqrt(max(variance, 0.0))

    def _predict_single_with_uncertainty(
        self,
        features: FeatureVector,
    ) -> tuple[Point, float | None]:
        """Return uncertainty from this estimator without output routing."""
        if (
            len(features) >= PUPIL_OPTIONAL_FEATURE_COUNT
            and features[PUPIL_AVAILABILITY_INDEX] < PUPIL_AVAILABLE_THRESHOLD
            and self._head_fallback is not None
        ):
            return self._head_fallback._predict_single_with_uncertainty(  # noqa: SLF001
                features
            )
        point = self._predict_single(features)
        if self._coefficient_covariance is None or self._residual_variance is None:
            return point, None
        compatible = np.asarray(features[: self._input_feature_count], dtype=np.float64)
        model_features = _transform_features(compatible, self._feature_transform)
        selected = model_features[list(self._feature_indices)]
        normalized = (selected - self._feature_mean) / self._feature_scale
        row = np.concatenate(([1.0], normalized))
        leverage = float(row @ self._coefficient_covariance @ row)
        radial_variance = max(0.0, leverage * float(np.sum(self._residual_variance)))
        return point, math.sqrt(radial_variance)


class IncrementalCalibration:
    """Update one provisional all-target fit without rescanning prior targets."""

    def __init__(
        self,
        samples: Sequence[CalibrationSample],
        *,
        ridge: float = 0.1,
        topology: DisplayTopology | None = None,
    ) -> None:
        """Build shared candidate and fold statistics in one initial O(n) pass."""
        if ridge < 0.0:
            msg = "ridge regularization must be non-negative"
            raise ValueError(msg)
        prepared = CalibrationModel._prepare(samples, None)  # noqa: SLF001
        _model, work, candidate, head_candidate, errors = CalibrationModel._fit_prepared(  # noqa: SLF001
            prepared,
            ridge=ridge,
        )
        self._statistics = prepared.statistics
        self._held_out_statistics = list(prepared.held_out_statistics)
        self._feature_count = prepared.features.shape[1]
        self._candidates = CalibrationModel._candidates(  # noqa: SLF001
            self._feature_count,
            len(prepared.features),
            ridge,
        )
        self._candidate_scores = {
            option: errors.get(option, 0.0 if option == candidate else math.inf)
            for option in self._candidates
        }
        self._candidate = candidate
        self._head_candidate = head_candidate
        self._sample_count = len(samples)
        self._statistic_updates = work.statistic_updates
        self._score_predictions = work.score_predictions
        self._ridge = ridge
        self._topology = topology
        self._output_calibrations: dict[str, IncrementalCalibration] = {}
        self._pending_output_samples: dict[str, list[CalibrationSample]] = {}
        if topology is not None:
            for region_id, local in CalibrationModel._samples_by_output(  # noqa: SLF001
                samples,
                topology,
            ).items():
                if len(local) >= MINIMUM_CALIBRATION_SAMPLES:
                    self._output_calibrations[region_id] = IncrementalCalibration(
                        local,
                        ridge=ridge,
                    )
                else:
                    self._pending_output_samples[region_id] = list(local)

    @property
    def work(self) -> CalibrationWork:
        """Expose deterministic work counters without wall-clock assumptions."""
        children = tuple(calibration.work for calibration in self._output_calibrations.values())
        return CalibrationWork(
            self._statistic_updates + sum(item.statistic_updates for item in children),
            self._score_predictions + sum(item.score_predictions for item in children),
        )

    @property
    def model(self) -> CalibrationModel:
        """Fit the selected provisional candidate from fixed-dimensional state."""
        model = self._statistics.fit(self._candidate, self._feature_count)
        if self._head_candidate is not None:
            model.with_head_fallback(
                self._statistics.fit(self._head_candidate, self._feature_count)
            )
        if self._topology is not None and self._output_calibrations:
            experts = {
                region_id: calibration.model
                for region_id, calibration in self._output_calibrations.items()
            }
            model.with_output_experts(self._topology, experts)
        return model

    def add(self, sample: CalibrationSample) -> CalibrationModel:
        """Incorporate one target and return its successor's provisional model."""
        if len(sample.features) != self._feature_count:
            msg = "calibration feature vectors must have one consistent non-zero length"
            raise ValueError(msg)
        feature_row = np.asarray(sample.features, dtype=np.float64)
        target_row = np.asarray((sample.target.x, sample.target.y), dtype=np.float64)
        for option in self._candidates:
            prediction = self._statistics.fit(option, self._feature_count).predict(sample.features)
            error = math.hypot(prediction.x - sample.target.x, prediction.y - sample.target.y)
            previous = self._candidate_scores[option]
            self._candidate_scores[option] = (
                error
                if not math.isfinite(previous)
                else (1.0 - PROVISIONAL_SCORE_ALPHA) * previous + PROVISIONAL_SCORE_ALPHA * error
            )
        self._candidate = min(self._candidates, key=self._candidate_scores.__getitem__)
        head_options = tuple(
            option
            for option in self._candidates
            if option.feature_indices in {indices for indices, _name in HEAD_FEATURE_SETS}
            or option.feature_name == "neural-head+face"
        )
        if head_options:
            self._head_candidate = min(head_options, key=self._candidate_scores.__getitem__)
        self._score_predictions += len(self._candidates)
        model_features = _fixed_neural_features(feature_row)
        model_dispersion = (
            _fixed_neural_dispersion(
                feature_row,
                np.asarray(sample.feature_dispersion, dtype=np.float64),
            )
            if sample.feature_dispersion
            else None
        )
        self._statistics.add(
            model_features,
            target_row,
            sample.weight,
            model_dispersion,
        )
        fold = self._sample_count % len(self._held_out_statistics)
        self._held_out_statistics[fold].add(
            model_features,
            target_row,
            sample.weight,
            model_dispersion,
        )
        self._sample_count += 1
        self._statistic_updates += 2
        if self._topology is not None:
            region_id = self._topology.locate(sample.target).region_id
            output_calibration = self._output_calibrations.get(region_id)
            if output_calibration is not None:
                output_calibration.add(sample)
            else:
                pending = self._pending_output_samples.setdefault(region_id, [])
                pending.append(sample)
                if len(pending) == MINIMUM_CALIBRATION_SAMPLES:
                    self._output_calibrations[region_id] = IncrementalCalibration(
                        pending,
                        ridge=self._ridge,
                    )
                    del self._pending_output_samples[region_id]
        return self.model
