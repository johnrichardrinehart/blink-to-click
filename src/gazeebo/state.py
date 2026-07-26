"""Secure compact persistence for all target-level training data."""

from __future__ import annotations

import contextlib
import gzip
import json
import math
import os
import stat
import tempfile
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

STORE_VERSION = 8
PRE_CVAR_STORE_VERSION = 7
PRE_SURPRISE_STORE_VERSION = 6
PRE_FEATURE_DISPERSION_STORE_VERSION = 5
PRE_NOISE_STORE_VERSION = 2
NOISE_STORE_VERSION = 3
PRE_COMPACT_STORE_VERSION = 4
LEGACY_GAZE_FEATURE_COUNT = 10
HEAD_CONTEXT_FEATURE_COUNT = 7
MAXIMUM_STORED_CLUSTERS = 64
MAXIMUM_MODEL_ANCHORS = 64
MAXIMUM_VALIDATION_RECORDS = 64
MAXIMUM_NOISE_SAMPLES = 10000
MAXIMUM_ON_DISK_BYTES = 128 * 1024 * 1024
MAXIMUM_LOGICAL_BYTES = 512 * 1024 * 1024
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_STORE_MAGIC = b"GZB1"
_MAXIMUM_JSON_DEPTH = 64
_OUTPUT_RECORD_LENGTH = 5
_TARGET_RECORD_LENGTH = 16
_PRE_SURPRISE_TARGET_RECORD_LENGTH = 14
_PRE_FEATURE_DISPERSION_TARGET_RECORD_LENGTH = 13
_NOISE_RECORD_LENGTH = 6
_CLUSTER_RECORD_LENGTH = 9
_ANCHOR_RECORD_LENGTH = 10
_VALIDATION_RECORD_LENGTH = 9
_PRE_CVAR_VALIDATION_RECORD_LENGTH = 7
_PRE_SURPRISE_VALIDATION_RECORD_LENGTH = 6


class TrainingStoreError(RuntimeError):
    """The local training store is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class StoreStats:
    """Read-only compact-store size and schema information."""

    schema_version: int
    target_count: int
    logical_bytes: int
    on_disk_bytes: int
    compression_ratio: float
    bytes_per_target: float


@dataclass(frozen=True, slots=True)
class OutputDescriptor:
    """Generic logical identity and geometry for one authorized output."""

    key: str
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Reject unusable stored output geometry."""
        if not self.key or self.width <= 0 or self.height <= 0:
            msg = "stored output descriptor is invalid"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CursorNoiseSummary:
    """Bounded stationary cursor spread without a frame-level trajectory."""

    sample_count: int
    horizontal_dispersion: float
    vertical_dispersion: float
    covariance: float
    median_radial_spread: float
    p95_radial_spread: float

    def __post_init__(self) -> None:
        """Reject malformed, unbounded, or internally inconsistent summaries."""
        values = (
            self.horizontal_dispersion,
            self.vertical_dispersion,
            self.covariance,
            self.median_radial_spread,
            self.p95_radial_spread,
        )
        covariance_limit = self.horizontal_dispersion * self.vertical_dispersion
        if (
            self.sample_count <= 0
            or self.sample_count > MAXIMUM_NOISE_SAMPLES
            or not all(math.isfinite(value) for value in values)
            or self.horizontal_dispersion < 0.0
            or self.vertical_dispersion < 0.0
            or abs(self.covariance) > covariance_limit + 1e-9
            or self.median_radial_spread < 0.0
            or self.p95_radial_spread < self.median_radial_spread
        ):
            msg = "cursor noise summary is invalid"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StoredTarget:
    """One target-level aggregate without its source frame stream."""

    sequence: int
    camera_id: str
    feature_schema: str
    features: tuple[float, ...]
    context: tuple[float, ...]
    outputs: tuple[OutputDescriptor, ...]
    output_key: str
    target_u: float
    target_v: float
    desktop_u: float
    desktop_v: float
    zone: str
    noise: CursorNoiseSummary | None = None
    feature_dispersion: tuple[float, ...] = ()
    unseen_error: float | None = None
    predictive_uncertainty: float | None = None

    def __post_init__(self) -> None:
        """Reject records that cannot be routed or remapped safely."""
        values = (
            *self.features,
            *self.context,
            self.target_u,
            self.target_v,
            self.desktop_u,
            self.desktop_v,
            *self.feature_dispersion,
        )
        if self.sequence < 0 or not self.camera_id or not self.feature_schema:
            msg = "stored target identity is invalid"
            raise ValueError(msg)
        if (
            not self.features
            or not self.context
            or not all(math.isfinite(value) for value in values)
        ):
            msg = "stored target vectors must contain finite values"
            raise ValueError(msg)
        if self.feature_dispersion and (
            len(self.feature_dispersion) != len(self.features)
            or any(value < 0.0 for value in self.feature_dispersion)
        ):
            msg = "stored target feature dispersion is invalid"
            raise ValueError(msg)
        if any(
            value is not None and (not math.isfinite(value) or value < 0.0)
            for value in (self.unseen_error, self.predictive_uncertainty)
        ):
            msg = "stored target surprise evidence is invalid"
            raise ValueError(msg)
        if not self.outputs or self.output_key not in {output.key for output in self.outputs}:
            msg = "stored target output is unavailable"
            raise ValueError(msg)
        if not all(
            0.0 <= value <= 1.0
            for value in (self.target_u, self.target_v, self.desktop_u, self.desktop_v)
        ):
            msg = "stored target coordinates must be normalized"
            raise ValueError(msg)
        if self.zone not in {"center", "edge", "corner"}:
            msg = "stored target zone is invalid"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ContextCluster:
    """Bounded routing statistics and target membership for one context."""

    cluster_id: str
    camera_id: str
    feature_schema: str
    centroid: tuple[float, ...]
    variance: tuple[float, ...]
    sample_count: int
    target_sequences: tuple[int, ...]
    median_error: float | None = None
    edge_error: float | None = None

    def __post_init__(self) -> None:
        """Reject malformed bounded-cluster statistics."""
        if not self.cluster_id or not self.camera_id or not self.feature_schema:
            msg = "context cluster identity is invalid"
            raise ValueError(msg)
        if not self.centroid or len(self.centroid) != len(self.variance):
            msg = "context cluster dimensions are invalid"
            raise ValueError(msg)
        values = (*self.centroid, *self.variance)
        if not all(math.isfinite(value) for value in values) or any(
            value < 0.0 for value in self.variance
        ):
            msg = "context cluster statistics are invalid"
            raise ValueError(msg)
        if self.sample_count <= 0 or any(sequence < 0 for sequence in self.target_sequences):
            msg = "context cluster membership is invalid"
            raise ValueError(msg)
        errors = (self.median_error, self.edge_error)
        if any(value is not None and (not math.isfinite(value) or value < 0.0) for value in errors):
            msg = "context cluster validation is invalid"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ModelAnchor:
    """One accepted all-data model bound to aggregate training context."""

    sequence: int
    camera_id: str
    feature_schema: str
    topology_id: str
    outputs: tuple[OutputDescriptor, ...]
    context_centroid: tuple[float, ...]
    context_variance: tuple[float, ...]
    model: dict[str, object]
    median_error: float
    edge_error: float

    def __post_init__(self) -> None:
        """Reject anchors that cannot support honest context interpolation."""
        values = (
            *self.context_centroid,
            *self.context_variance,
            self.median_error,
            self.edge_error,
        )
        if (
            self.sequence < 0
            or not self.camera_id
            or not self.feature_schema
            or not self.topology_id
            or not self.outputs
            or not self.context_centroid
            or len(self.context_centroid) != len(self.context_variance)
            or not self.model
            or not all(math.isfinite(value) for value in values)
            or any(value < 0.0 for value in self.context_variance)
            or self.median_error < 0.0
            or self.edge_error < 0.0
        ):
            msg = "validated model anchor is invalid"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Aggregate unseen quality without retaining its observations."""

    sequence: int
    camera_id: str
    topology_id: str
    routing: str
    median_error: float
    edge_error: float
    maximum_region_error: float = 0.0
    maximum_region_cvar90: float | None = None
    maximum_region_upper: float | None = None

    def __post_init__(self) -> None:
        """Reject malformed aggregate holdout metrics."""
        if self.sequence < 0 or not self.camera_id or not self.topology_id or not self.routing:
            msg = "validation identity is invalid"
            raise ValueError(msg)
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.median_error,
                self.edge_error,
                self.maximum_region_error,
            )
        ) or any(
            value is not None and (not math.isfinite(value) or value < 0.0)
            for value in (
                self.maximum_region_cvar90,
                self.maximum_region_upper,
            )
        ):
            msg = "validation errors are invalid"
            raise ValueError(msg)


@dataclass(slots=True)
class TrainingState:
    """Complete versioned state replaced by one atomic transaction."""

    next_sequence: int = 0
    targets: list[StoredTarget] = field(default_factory=list)
    clusters: list[ContextCluster] = field(default_factory=list)
    models: dict[str, dict[str, object]] = field(default_factory=dict)
    anchors: list[ModelAnchor] = field(default_factory=list)
    validations: list[ValidationSummary] = field(default_factory=list)

    def validate(self) -> None:
        """Reject malformed or internally inconsistent state."""
        if self.next_sequence < 0:
            msg = "training sequence is invalid"
            raise TrainingStoreError(msg)
        if len(self.clusters) > MAXIMUM_STORED_CLUSTERS:
            msg = "training store exceeds its cluster limit"
            raise TrainingStoreError(msg)
        if len(self.anchors) > MAXIMUM_MODEL_ANCHORS:
            msg = "training store exceeds its validated model-anchor limit"
            raise TrainingStoreError(msg)
        if len(self.validations) > MAXIMUM_VALIDATION_RECORDS:
            msg = "training store exceeds its validation limit"
            raise TrainingStoreError(msg)
        sequences = [target.sequence for target in self.targets]
        if len(sequences) != len(set(sequences)):
            msg = "training target sequences must be unique"
            raise TrainingStoreError(msg)
        available = set(sequences)
        if any(not set(cluster.target_sequences) <= available for cluster in self.clusters):
            msg = "context cluster refers to a missing target"
            raise TrainingStoreError(msg)
        if any(
            not isinstance(key, str) or not isinstance(value, dict)
            for key, value in self.models.items()
        ):
            msg = "stored model map is malformed"
            raise TrainingStoreError(msg)


class TrainingStore:
    """Read and atomically replace one owner-only local training store."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        ephemeral: bool = False,
        maximum_logical_bytes: int = MAXIMUM_LOGICAL_BYTES,
    ) -> None:
        """Use the XDG store or an injected deterministic test path."""
        if maximum_logical_bytes <= 0:
            msg = "maximum logical store size must be positive"
            raise ValueError(msg)
        self.path = path or _default_path()
        self.ephemeral = ephemeral
        self.maximum_logical_bytes = maximum_logical_bytes

    def load(self) -> TrainingState:
        """Load validated compact or legacy state without mutating its file."""
        if self.ephemeral or not self.path.exists():
            return TrainingState()
        self._validate_directory(create=False)
        self._validate_file()
        try:
            payload, _compressed = self._read_payload()
            raw = json.loads(payload)
            _validate_json_shape(raw, self.maximum_logical_bytes)
            return _decode_state(raw)
        except TrainingStoreError:
            raise
        except (
            OSError,
            UnicodeDecodeError,
            ValueError,
            TypeError,
            KeyError,
            RecursionError,
            json.JSONDecodeError,
            zlib.error,
        ) as error:
            msg = "training store is malformed"
            raise TrainingStoreError(msg) from error

    def save(self, state: TrainingState) -> None:
        """Atomically save lossless compact state unless operation is ephemeral."""
        if self.ephemeral:
            return
        state.validate()
        parent = self._validate_directory(create=True)
        if self.path.exists():
            self._validate_file()
        logical = json.dumps(
            _encode_compact_state(state),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        if len(logical) > self.maximum_logical_bytes:
            msg = "training store exceeds its decompressed size limit"
            raise TrainingStoreError(msg)
        payload = _STORE_MAGIC + gzip.compress(logical, compresslevel=9, mtime=0)
        if len(payload) > MAXIMUM_ON_DISK_BYTES:
            msg = "training store exceeds its on-disk size limit"
            raise TrainingStoreError(msg)
        descriptor = -1
        temporary = ""
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".training-", dir=parent)
            os.fchmod(descriptor, _FILE_MODE)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            Path(temporary).replace(self.path)
            temporary = ""
            directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as error:
            msg = "could not atomically save training data"
            raise TrainingStoreError(msg) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary:
                with contextlib.suppress(FileNotFoundError):
                    Path(temporary).unlink()

    def dump_json(self) -> str:
        """Return stable expanded JSON without creating another plaintext file."""
        state = self.load()
        return (
            json.dumps(
                _encode_state(state),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def stats(self) -> StoreStats:
        """Report compact logical and physical size without mutating state."""
        state = self.load()
        if self.ephemeral or not self.path.exists():
            logical = json.dumps(
                _encode_compact_state(state),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            on_disk = 0
        else:
            logical, _compressed = self._read_payload()
            on_disk = self.path.stat().st_size
        target_count = len(state.targets)
        ratio = len(logical) / on_disk if on_disk else 0.0
        per_target = on_disk / target_count if target_count else 0.0
        return StoreStats(
            STORE_VERSION,
            target_count,
            len(logical),
            on_disk,
            ratio,
            per_target,
        )

    def _read_payload(self) -> tuple[bytes, bool]:
        metadata = self.path.stat()
        if metadata.st_size > MAXIMUM_ON_DISK_BYTES:
            msg = "training store exceeds its on-disk size limit"
            raise TrainingStoreError(msg)
        payload = self.path.read_bytes()
        if not payload.startswith(_STORE_MAGIC):
            if len(payload) > self.maximum_logical_bytes:
                msg = "training store exceeds its decompressed size limit"
                raise TrainingStoreError(msg)
            return payload, False
        decompressor = zlib.decompressobj(wbits=31)
        logical = decompressor.decompress(
            payload[len(_STORE_MAGIC) :],
            self.maximum_logical_bytes + 1,
        )
        if len(logical) > self.maximum_logical_bytes or decompressor.unconsumed_tail:
            msg = "training store exceeds its decompressed size limit"
            raise TrainingStoreError(msg)
        logical += decompressor.flush(self.maximum_logical_bytes + 1 - len(logical))
        if len(logical) > self.maximum_logical_bytes:
            msg = "training store exceeds its decompressed size limit"
            raise TrainingStoreError(msg)
        if not decompressor.eof or decompressor.unused_data:
            msg = "training store is malformed"
            raise TrainingStoreError(msg)
        return logical, True

    def reset(self) -> None:
        """Remove persisted training state without touching ephemeral paths."""
        if self.ephemeral or not self.path.exists():
            return
        self._validate_directory(create=False)
        self._validate_file()
        try:
            self.path.unlink()
        except OSError as error:
            msg = "could not reset training data"
            raise TrainingStoreError(msg) from error

    def _validate_directory(self, *, create: bool) -> Path:
        parent = self.path.parent
        if create and not parent.exists():
            try:
                parent.mkdir(mode=_DIRECTORY_MODE, parents=True)
                parent.chmod(_DIRECTORY_MODE)
            except OSError as error:
                msg = "could not create the training directory"
                raise TrainingStoreError(msg) from error
        try:
            metadata = parent.lstat()
        except OSError as error:
            msg = "training directory is unavailable"
            raise TrainingStoreError(msg) from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            msg = "training directory must not be a symlink"
            raise TrainingStoreError(msg)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            msg = "training directory must be owner-only"
            raise TrainingStoreError(msg)
        return parent

    def _validate_file(self) -> None:
        metadata = self.path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            msg = "training store must be a regular file"
            raise TrainingStoreError(msg)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            msg = "training store must be owner-only"
            raise TrainingStoreError(msg)


def _default_path() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return root / "gazeebo" / "training-v1.json"


class _StringTable:
    """Intern repeated compact-store strings in deterministic encounter order."""

    def __init__(self) -> None:
        self.values: list[str] = []
        self.indices: dict[str, int] = {}

    def index(self, value: str) -> int:
        """Return one stable string-table index."""
        existing = self.indices.get(value)
        if existing is not None:
            return existing
        result = len(self.values)
        self.values.append(value)
        self.indices[value] = result
        return result


def _encode_compact_state(state: TrainingState) -> dict[str, object]:
    """Encode state with interned descriptors and array-shaped records."""
    strings = _StringTable()
    outputs: list[list[object]] = []
    output_indices: dict[OutputDescriptor, int] = {}
    topologies: list[list[int]] = []
    topology_indices: dict[tuple[OutputDescriptor, ...], int] = {}

    def output_index(output: OutputDescriptor) -> int:
        existing = output_indices.get(output)
        if existing is not None:
            return existing
        result = len(outputs)
        outputs.append(
            [
                strings.index(output.key),
                output.x,
                output.y,
                output.width,
                output.height,
            ]
        )
        output_indices[output] = result
        return result

    def topology_index(value: tuple[OutputDescriptor, ...]) -> int:
        existing = topology_indices.get(value)
        if existing is not None:
            return existing
        result = len(topologies)
        topologies.append([output_index(output) for output in value])
        topology_indices[value] = result
        return result

    targets: list[list[object]] = []
    for target in state.targets:
        noise = target.noise
        targets.append(
            [
                target.sequence,
                strings.index(target.camera_id),
                strings.index(target.feature_schema),
                list(target.features),
                list(target.context),
                topology_index(target.outputs),
                strings.index(target.output_key),
                target.target_u,
                target.target_v,
                target.desktop_u,
                target.desktop_v,
                strings.index(target.zone),
                None
                if noise is None
                else [
                    noise.sample_count,
                    noise.horizontal_dispersion,
                    noise.vertical_dispersion,
                    noise.covariance,
                    noise.median_radial_spread,
                    noise.p95_radial_spread,
                ],
                list(target.feature_dispersion),
                target.unseen_error,
                target.predictive_uncertainty,
            ]
        )
    clusters = [
        [
            strings.index(cluster.cluster_id),
            strings.index(cluster.camera_id),
            strings.index(cluster.feature_schema),
            list(cluster.centroid),
            list(cluster.variance),
            cluster.sample_count,
            list(cluster.target_sequences),
            cluster.median_error,
            cluster.edge_error,
        ]
        for cluster in state.clusters
    ]
    anchors = [
        [
            anchor.sequence,
            strings.index(anchor.camera_id),
            strings.index(anchor.feature_schema),
            strings.index(anchor.topology_id),
            topology_index(anchor.outputs),
            list(anchor.context_centroid),
            list(anchor.context_variance),
            anchor.model,
            anchor.median_error,
            anchor.edge_error,
        ]
        for anchor in state.anchors
    ]
    validations = [
        [
            validation.sequence,
            strings.index(validation.camera_id),
            strings.index(validation.topology_id),
            strings.index(validation.routing),
            validation.median_error,
            validation.edge_error,
            validation.maximum_region_error,
            validation.maximum_region_cvar90,
            validation.maximum_region_upper,
        ]
        for validation in state.validations
    ]
    return {
        "v": STORE_VERSION,
        "n": state.next_sequence,
        "s": strings.values,
        "o": outputs,
        "p": topologies,
        "t": targets,
        "c": clusters,
        "m": state.models,
        "a": anchors,
        "r": validations,
    }


def _encode_state(state: TrainingState) -> dict[str, object]:
    """Expand state into stable schema-labelled human-readable records."""
    return {
        "version": STORE_VERSION,
        "next_sequence": state.next_sequence,
        "targets": [asdict(target) for target in state.targets],
        "clusters": [asdict(cluster) for cluster in state.clusters],
        "models": state.models,
        "anchors": [asdict(anchor) for anchor in state.anchors],
        "validations": [asdict(validation) for validation in state.validations],
    }


@dataclass(frozen=True, slots=True)
class _CompactReferences:
    """Validated string and topology tables for compact records."""

    strings: tuple[str, ...]
    topologies: tuple[tuple[OutputDescriptor, ...], ...]

    @classmethod
    def decode(cls, raw: dict[str, object]) -> _CompactReferences:
        """Decode shared descriptors before decoding records that reference them."""
        strings = tuple(str(value) for value in _list(raw.get("s", [])))

        def text(value: object) -> str:
            return _indexed(strings, value, "string")

        outputs: list[OutputDescriptor] = []
        for value in _list(raw.get("o", [])):
            item = _fixed_record(value, _OUTPUT_RECORD_LENGTH, "output")
            outputs.append(
                OutputDescriptor(
                    text(item[0]),
                    _integer(item[1]),
                    _integer(item[2]),
                    _integer(item[3]),
                    _integer(item[4]),
                )
            )
        topologies = tuple(
            tuple(_indexed(outputs, index, "output") for index in _list(value))
            for value in _list(raw.get("p", []))
        )
        return cls(strings, topologies)

    def text(self, value: object) -> str:
        """Resolve one compact string reference."""
        return _indexed(self.strings, value, "string")

    def topology(self, value: object) -> tuple[OutputDescriptor, ...]:
        """Resolve one compact topology reference."""
        return _indexed(self.topologies, value, "topology")


def _decode_compact_noise(value: object) -> CursorNoiseSummary | None:
    if value is None:
        return None
    item = _fixed_record(value, _NOISE_RECORD_LENGTH, "noise")
    return CursorNoiseSummary(
        _integer(item[0]),
        _number(item[1]),
        _number(item[2]),
        _number(item[3]),
        _number(item[4]),
        _number(item[5]),
    )


def _decode_compact_target(
    value: object,
    refs: _CompactReferences,
    version: int,
) -> StoredTarget:
    record_length = {
        PRE_FEATURE_DISPERSION_STORE_VERSION: _PRE_FEATURE_DISPERSION_TARGET_RECORD_LENGTH,
        PRE_SURPRISE_STORE_VERSION: _PRE_SURPRISE_TARGET_RECORD_LENGTH,
        PRE_CVAR_STORE_VERSION: _TARGET_RECORD_LENGTH,
        STORE_VERSION: _TARGET_RECORD_LENGTH,
    }[version]
    item = _fixed_record(value, record_length, "target")
    return StoredTarget(
        _integer(item[0]),
        refs.text(item[1]),
        refs.text(item[2]),
        _float_tuple(item[3]),
        _float_tuple(item[4]),
        refs.topology(item[5]),
        refs.text(item[6]),
        _number(item[7]),
        _number(item[8]),
        _number(item[9]),
        _number(item[10]),
        refs.text(item[11]),
        _decode_compact_noise(item[12]),
        () if version == PRE_FEATURE_DISPERSION_STORE_VERSION else _float_tuple(item[13]),
        None if version < PRE_CVAR_STORE_VERSION else _optional_float(item[14]),
        None if version < PRE_CVAR_STORE_VERSION else _optional_float(item[15]),
    )


def _decode_compact_cluster(value: object, refs: _CompactReferences) -> ContextCluster:
    item = _fixed_record(value, _CLUSTER_RECORD_LENGTH, "cluster")
    return ContextCluster(
        refs.text(item[0]),
        refs.text(item[1]),
        refs.text(item[2]),
        _float_tuple(item[3]),
        _float_tuple(item[4]),
        _integer(item[5]),
        tuple(_integer(sequence) for sequence in _list(item[6])),
        _optional_float(item[7]),
        _optional_float(item[8]),
    )


def _decode_compact_anchor(value: object, refs: _CompactReferences) -> ModelAnchor:
    item = _fixed_record(value, _ANCHOR_RECORD_LENGTH, "model-anchor")
    return ModelAnchor(
        _integer(item[0]),
        refs.text(item[1]),
        refs.text(item[2]),
        refs.text(item[3]),
        refs.topology(item[4]),
        _float_tuple(item[5]),
        _float_tuple(item[6]),
        _mapping(item[7]),
        _number(item[8]),
        _number(item[9]),
    )


def _decode_compact_validation(
    value: object,
    refs: _CompactReferences,
    version: int,
) -> ValidationSummary:
    record_length = {
        PRE_FEATURE_DISPERSION_STORE_VERSION: _PRE_SURPRISE_VALIDATION_RECORD_LENGTH,
        PRE_SURPRISE_STORE_VERSION: _PRE_SURPRISE_VALIDATION_RECORD_LENGTH,
        PRE_CVAR_STORE_VERSION: _PRE_CVAR_VALIDATION_RECORD_LENGTH,
        STORE_VERSION: _VALIDATION_RECORD_LENGTH,
    }[version]
    item = _fixed_record(value, record_length, "validation")
    return ValidationSummary(
        _integer(item[0]),
        refs.text(item[1]),
        refs.text(item[2]),
        refs.text(item[3]),
        _number(item[4]),
        _number(item[5]),
        0.0 if version < PRE_CVAR_STORE_VERSION else _number(item[6]),
        None if version < STORE_VERSION else _optional_float(item[7]),
        None if version < STORE_VERSION else _optional_float(item[8]),
    )


def _decode_compact_state(raw: dict[str, object]) -> TrainingState:
    """Decode the current or immediately preceding interned representation."""
    version = _integer(raw.get("v", -1))
    if version not in {
        PRE_FEATURE_DISPERSION_STORE_VERSION,
        PRE_SURPRISE_STORE_VERSION,
        PRE_CVAR_STORE_VERSION,
        STORE_VERSION,
    }:
        msg = "training store version is unsupported"
        raise TrainingStoreError(msg)
    refs = _CompactReferences.decode(raw)
    state = TrainingState(
        next_sequence=_integer(raw.get("n", 0)),
        targets=[_decode_compact_target(value, refs, version) for value in _list(raw.get("t", []))],
        clusters=[_decode_compact_cluster(value, refs) for value in _list(raw.get("c", []))],
        models=cast("dict[str, dict[str, object]]", _mapping(raw.get("m", {}))),
        anchors=[_decode_compact_anchor(value, refs) for value in _list(raw.get("a", []))],
        validations=[
            _decode_compact_validation(value, refs, version) for value in _list(raw.get("r", []))
        ],
    )
    state.validate()
    return state


def _decode_state(value: object) -> TrainingState:
    if not isinstance(value, dict):
        msg = "training store root must be an object"
        raise TrainingStoreError(msg)
    raw = cast("dict[str, object]", value)
    if "v" in raw:
        return _decode_compact_state(raw)
    version = raw.get("version")
    if version == 0:
        raw = {
            "version": STORE_VERSION,
            "next_sequence": raw.get("next_sequence", 0),
            "targets": raw.get("targets", []),
            "clusters": [],
            "models": {},
            "anchors": [],
            "validations": [],
        }
    elif version == 1:
        raw = {**raw, "version": STORE_VERSION, "anchors": []}
    elif version in (
        PRE_NOISE_STORE_VERSION,
        NOISE_STORE_VERSION,
        PRE_COMPACT_STORE_VERSION,
        PRE_FEATURE_DISPERSION_STORE_VERSION,
        PRE_SURPRISE_STORE_VERSION,
        PRE_CVAR_STORE_VERSION,
    ):
        raw = {**raw, "version": STORE_VERSION}
    elif version != STORE_VERSION:
        msg = "training store version is unsupported"
        raise TrainingStoreError(msg)

    state = TrainingState(
        next_sequence=_integer(raw.get("next_sequence", 0)),
        targets=[_decode_target(item) for item in _list(raw.get("targets", []))],
        clusters=[_decode_cluster(item) for item in _list(raw.get("clusters", []))],
        models=cast("dict[str, dict[str, object]]", raw.get("models", {})),
        anchors=[_decode_anchor(item) for item in _list(raw.get("anchors", []))],
        validations=[_decode_validation(item) for item in _list(raw.get("validations", []))],
    )
    state.validate()
    return state


def _decode_target(value: object) -> StoredTarget:
    raw = _mapping(value)
    outputs = _decode_outputs(raw["outputs"])
    features = _float_tuple(raw["features"])
    context = _float_tuple(raw["context"])
    if len(features) == LEGACY_GAZE_FEATURE_COUNT and len(context) >= HEAD_CONTEXT_FEATURE_COUNT:
        features = (*features, 1.0, 1.0, context[2], context[5], context[6])
    return StoredTarget(
        sequence=_integer(raw["sequence"]),
        camera_id=str(raw["camera_id"]),
        feature_schema=str(raw["feature_schema"]),
        features=features,
        context=context,
        outputs=outputs,
        output_key=str(raw["output_key"]),
        target_u=_number(raw["target_u"]),
        target_v=_number(raw["target_v"]),
        desktop_u=_number(raw["desktop_u"]),
        desktop_v=_number(raw["desktop_v"]),
        zone=str(raw["zone"]),
        noise=_decode_noise(raw.get("noise")),
        feature_dispersion=_float_tuple(raw.get("feature_dispersion", [])),
        unseen_error=_optional_float(raw.get("unseen_error")),
        predictive_uncertainty=_optional_float(raw.get("predictive_uncertainty")),
    )


def _decode_noise(value: object) -> CursorNoiseSummary | None:
    if value is None:
        return None
    raw = _mapping(value)
    return CursorNoiseSummary(
        sample_count=_integer(raw["sample_count"]),
        horizontal_dispersion=_number(raw["horizontal_dispersion"]),
        vertical_dispersion=_number(raw["vertical_dispersion"]),
        covariance=_number(raw["covariance"]),
        median_radial_spread=_number(raw["median_radial_spread"]),
        p95_radial_spread=_number(raw["p95_radial_spread"]),
    )


def _decode_outputs(value: object) -> tuple[OutputDescriptor, ...]:
    return tuple(
        OutputDescriptor(
            key=str(output["key"]),
            x=_integer(output["x"]),
            y=_integer(output["y"]),
            width=_integer(output["width"]),
            height=_integer(output["height"]),
        )
        for output in (_mapping(item) for item in _list(value))
    )


def _decode_cluster(value: object) -> ContextCluster:
    raw = _mapping(value)
    return ContextCluster(
        cluster_id=str(raw["cluster_id"]),
        camera_id=str(raw["camera_id"]),
        feature_schema=str(raw["feature_schema"]),
        centroid=_float_tuple(raw["centroid"]),
        variance=_float_tuple(raw["variance"]),
        sample_count=_integer(raw["sample_count"]),
        target_sequences=tuple(_integer(item) for item in _list(raw["target_sequences"])),
        median_error=_optional_float(raw.get("median_error")),
        edge_error=_optional_float(raw.get("edge_error")),
    )


def _decode_anchor(value: object) -> ModelAnchor:
    raw = _mapping(value)
    return ModelAnchor(
        sequence=_integer(raw["sequence"]),
        camera_id=str(raw["camera_id"]),
        feature_schema=str(raw["feature_schema"]),
        topology_id=str(raw["topology_id"]),
        outputs=_decode_outputs(raw["outputs"]),
        context_centroid=_float_tuple(raw["context_centroid"]),
        context_variance=_float_tuple(raw["context_variance"]),
        model=_mapping(raw["model"]),
        median_error=_number(raw["median_error"]),
        edge_error=_number(raw["edge_error"]),
    )


def _decode_validation(value: object) -> ValidationSummary:
    raw = _mapping(value)
    return ValidationSummary(
        sequence=_integer(raw["sequence"]),
        camera_id=str(raw["camera_id"]),
        topology_id=str(raw["topology_id"]),
        routing=str(raw["routing"]),
        median_error=_number(raw["median_error"]),
        edge_error=_number(raw["edge_error"]),
        maximum_region_error=_number(raw.get("maximum_region_error", 0.0)),
        maximum_region_cvar90=_optional_float(raw.get("maximum_region_cvar90")),
        maximum_region_upper=_optional_float(raw.get("maximum_region_upper")),
    )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = "training record must be an object"
        raise TrainingStoreError(msg)
    return cast("dict[str, object]", value)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        msg = "training record must contain arrays"
        raise TrainingStoreError(msg)
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        msg = "training record integer is malformed"
        raise TrainingStoreError(msg)
    return int(value)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        msg = "training record number is malformed"
        raise TrainingStoreError(msg)
    return float(value)


def _float_tuple(value: object) -> tuple[float, ...]:
    return tuple(_number(item) for item in _list(value))


def _optional_float(value: object) -> float | None:
    return None if value is None else _number(value)


def _fixed_record(value: object, length: int, label: str) -> list[object]:
    """Require one compact positional record with its exact schema length."""
    item = _list(value)
    if len(item) != length:
        msg = f"training store {label} record is malformed"
        raise TrainingStoreError(msg)
    return item


def _indexed[T](values: tuple[T, ...] | list[T], value: object, label: str) -> T:
    """Resolve one non-negative compact-table index."""
    index = _integer(value)
    if index < 0 or index >= len(values):
        msg = f"training store {label} reference is invalid"
        raise TrainingStoreError(msg)
    return values[index]


def _validate_json_shape(value: object, maximum_nodes: int) -> None:
    """Bound parser nesting and aggregate collection expansion."""
    remaining = maximum_nodes
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        remaining -= 1
        if remaining < 0:
            msg = "training store exceeds its collection limit"
            raise TrainingStoreError(msg)
        if depth > _MAXIMUM_JSON_DEPTH:
            msg = "training store exceeds its nesting limit"
            raise TrainingStoreError(msg)
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
