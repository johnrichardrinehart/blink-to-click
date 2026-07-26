"""Tests for secure target-level training persistence."""

from __future__ import annotations

import json
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from gazeebo.state import (
    MAXIMUM_MODEL_ANCHORS,
    ContextCluster,
    CursorNoiseSummary,
    ModelAnchor,
    OutputDescriptor,
    StoredTarget,
    TrainingState,
    TrainingStore,
    TrainingStoreError,
    ValidationSummary,
    _decode_state,
    _encode_compact_state,
)


def target(sequence: int = 0) -> StoredTarget:
    """Create one finite target-level aggregate."""
    return StoredTarget(
        sequence=sequence,
        camera_id="camera-a",
        feature_schema="gaze-v1",
        features=(0.1, 0.2, 0.3),
        context=(0.0, 0.1, 0.5, 0.4),
        outputs=(OutputDescriptor("left", 0, 0, 1000, 700),),
        output_key="left",
        target_u=0.5,
        target_v=0.5,
        desktop_u=0.5,
        desktop_v=0.5,
        zone="center",
        feature_dispersion=(0.01, 0.02, 0.03),
    )


def state() -> TrainingState:
    """Create a state containing every serialized record type."""
    return TrainingState(
        next_sequence=1,
        targets=[target()],
        clusters=[
            ContextCluster(
                "context-0",
                "camera-a",
                "gaze-v1",
                (0.0, 0.1, 0.5, 0.4),
                (0.01, 0.01, 0.01, 0.01),
                1,
                (0,),
                median_error=80.0,
                edge_error=90.0,
            )
        ],
        models={"global": {"kind": "affine", "coefficients": [[1.0, 2.0]]}},
        anchors=[
            ModelAnchor(
                1,
                "camera-a",
                "gaze-v1",
                "layout-a",
                (OutputDescriptor("left", 0, 0, 1000, 700),),
                (0.0, 0.1, 0.5, 0.4),
                (0.01, 0.01, 0.01, 0.01),
                {"kind": "affine"},
                80.0,
                90.0,
            )
        ],
        validations=[ValidationSummary(0, "camera-a", "layout-a", "global", 80.0, 90.0)],
    )


class TrainingStoreTests(unittest.TestCase):
    """Lock schema, permissions, atomicity, migration, and ephemeral behavior."""

    def test_round_trip_uses_owner_only_file_and_directory(self) -> None:
        """Persisted derived data are private and survive a complete reload."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gazeebo" / "training-v1.json"
            store = TrainingStore(path)
            expected = state()
            expected.targets[0] = replace(
                expected.targets[0],
                noise=CursorNoiseSummary(120, 4.0, 5.0, 3.0, 6.0, 12.0),
                unseen_error=123.5,
                predictive_uncertainty=45.25,
            )
            expected.validations[0] = replace(
                expected.validations[0],
                maximum_region_cvar90=180.0,
                maximum_region_upper=220.0,
            )
            store.save(expected)

            assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert store.load() == expected
            assert list(path.parent.glob(".training-*")) == []

    def test_v2_targets_migrate_without_inventing_noise_evidence(self) -> None:
        """Existing targets gain neither noise nor feature dispersion evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training-v1.json"
            store = TrainingStore(path)
            store.save(state())
            raw = json.loads(store.dump_json())
            raw["version"] = 2
            for item in raw["targets"]:
                item.pop("noise", None)
                item.pop("feature_dispersion", None)
                item["features"] = [value / 10.0 for value in range(10)]
                item["context"] = [value / 10.0 for value in range(9)]
            original = json.dumps(raw)
            path.write_text(original, encoding="utf-8")
            migrated = store.load()
            assert migrated.targets[0].noise is None
            assert migrated.targets[0].feature_dispersion == ()
            assert len(migrated.targets[0].features) == 15
            assert migrated.targets[0].features[10:] == (1.0, 1.0, 0.2, 0.5, 0.6)
            assert path.read_text(encoding="utf-8") == original

    def test_v5_compact_targets_migrate_without_feature_dispersion(self) -> None:
        """The prior compact store remains readable without invented input noise."""
        raw = _encode_compact_state(state())
        raw["v"] = 5
        records = raw["t"]
        assert isinstance(records, list)
        for record in records:
            assert isinstance(record, list)
            del record[-3:]
        validations = raw["r"]
        assert isinstance(validations, list)
        for validation in validations:
            assert isinstance(validation, list)
            del validation[-3:]
        migrated = _decode_state(raw)
        assert migrated.targets[0].feature_dispersion == ()
        assert migrated.validations[0].maximum_region_cvar90 is None

    def test_v6_compact_targets_migrate_without_inventing_surprise(self) -> None:
        """Pre-surprise compact targets remain exact without fabricated errors."""
        raw = _encode_compact_state(state())
        raw["v"] = 6
        records = raw["t"]
        assert isinstance(records, list)
        for record in records:
            assert isinstance(record, list)
            del record[-2:]
        validations = raw["r"]
        assert isinstance(validations, list)
        for validation in validations:
            assert isinstance(validation, list)
            del validation[-3:]
        migrated = _decode_state(raw)
        assert migrated.targets[0].feature_dispersion == (0.01, 0.02, 0.03)
        assert migrated.targets[0].unseen_error is None
        assert migrated.targets[0].predictive_uncertainty is None
        assert migrated.validations[0].maximum_region_error == 0.0
        assert migrated.validations[0].maximum_region_cvar90 is None

    def test_v7_compact_validations_migrate_without_inventing_cvar(self) -> None:
        """Schema-seven errors remain exact without fabricated tail metrics."""
        raw = _encode_compact_state(state())
        raw["v"] = 7
        validations = raw["r"]
        assert isinstance(validations, list)
        for validation in validations:
            assert isinstance(validation, list)
            del validation[-2:]
        migrated = _decode_state(raw)
        assert migrated.targets[0] == state().targets[0]
        assert migrated.validations[0].maximum_region_error == 0.0
        assert migrated.validations[0].maximum_region_cvar90 is None
        assert migrated.validations[0].maximum_region_upper is None

    def test_surprise_evidence_is_finite_non_negative_and_optional(self) -> None:
        """Only bounded target-level surprise values may enter persistence."""
        assert replace(target(), unseen_error=None, predictive_uncertainty=None)
        with self.assertRaisesRegex(ValueError, "surprise"):
            replace(target(), unseen_error=float("nan"))
        with self.assertRaisesRegex(ValueError, "surprise"):
            replace(target(), predictive_uncertainty=-1.0)

    def test_noise_summary_is_finite_bounded_and_ordered(self) -> None:
        """Persistence cannot retain malformed or frame-sized noise records."""
        with self.assertRaisesRegex(ValueError, "noise"):
            CursorNoiseSummary(0, 1.0, 1.0, 0.0, 1.0, 2.0)
        with self.assertRaisesRegex(ValueError, "noise"):
            CursorNoiseSummary(20, 1.0, 1.0, 0.0, 5.0, 4.0)
        with self.assertRaisesRegex(ValueError, "noise"):
            CursorNoiseSummary(20, float("nan"), 1.0, 0.0, 1.0, 2.0)

    def test_unsupported_corrupt_and_insecure_state_fails_closed(self) -> None:
        """Malformed or publicly readable state is never partly accepted."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training-v1.json"
            path.write_text('{"version":99}', encoding="utf-8")
            path.chmod(0o600)
            with self.assertRaisesRegex(TrainingStoreError, "version"):
                TrainingStore(path).load()

            path.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(TrainingStoreError, "malformed"):
                TrainingStore(path).load()

            path.write_text('{"version":1}', encoding="utf-8")
            path.chmod(0o644)
            with self.assertRaisesRegex(TrainingStoreError, "owner-only"):
                TrainingStore(path).load()

    def test_symlinked_store_is_rejected(self) -> None:
        """A store cannot redirect sensitive writes through a symbolic link."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "elsewhere"
            destination.write_text('{"version":1}', encoding="utf-8")
            destination.chmod(0o600)
            path = root / "training-v1.json"
            path.symlink_to(destination)
            with self.assertRaisesRegex(TrainingStoreError, "regular file"):
                TrainingStore(path).load()

    def test_old_empty_states_migrate_without_writing(self) -> None:
        """Pre-anchor schemas migrate deterministically in memory."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training-v1.json"
            original = json.dumps({"version": 0, "next_sequence": 7, "targets": []})
            path.write_text(original, encoding="utf-8")
            path.chmod(0o600)
            migrated = TrainingStore(path).load()
            assert migrated.next_sequence == 7
            assert migrated.targets == []
            assert migrated.anchors == []
            assert path.read_text(encoding="utf-8") == original

            version_one = json.dumps({"version": 1, "next_sequence": 8, "targets": []})
            path.write_text(version_one, encoding="utf-8")
            migrated = TrainingStore(path).load()
            assert migrated.next_sequence == 8
            assert migrated.anchors == []
            assert path.read_text(encoding="utf-8") == version_one

    def test_ephemeral_mode_does_not_read_write_or_reset_path(self) -> None:
        """Ephemeral operation ignores even malformed persistent state."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training-v1.json"
            path.write_text("private existing bytes", encoding="utf-8")
            path.chmod(0o644)
            store = TrainingStore(path, ephemeral=True)
            assert store.load() == TrainingState()
            store.save(state())
            store.reset()
            assert path.read_text(encoding="utf-8") == "private existing bytes"

    def test_reset_removes_only_a_valid_store(self) -> None:
        """The reset operation removes private state and remains idempotent."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training-v1.json"
            store = TrainingStore(path)
            store.save(state())
            store.reset()
            store.reset()
            assert not path.exists()

    def test_store_retains_more_than_the_legacy_target_cap(self) -> None:
        """Every historical target survives persistence without count-based eviction."""
        expected = TrainingState(
            next_sequence=400,
            targets=[target(index) for index in range(400)],
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "training-v1.store")
            store.save(expected)
            assert store.load() == expected

    def test_compact_store_dump_stats_and_size_budget_are_deterministic(self) -> None:
        """Ten thousand targets remain lossless, compact, and introspectable."""
        expected = TrainingState(
            next_sequence=10_000,
            targets=[
                replace(
                    target(index),
                    target_u=(index % 101) / 100.0,
                    target_v=(index % 97) / 96.0,
                    noise=CursorNoiseSummary(60, 4.0, 5.0, 3.0, 6.0, 12.0),
                    unseen_error=float(index % 1000),
                    predictive_uncertainty=float(index % 100),
                )
                for index in range(10_000)
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training-v1.store"
            store = TrainingStore(path)
            store.save(expected)
            actual = store.load()
            statistics = store.stats()
            dumped = json.loads(store.dump_json())

            assert actual == expected
            assert statistics.schema_version >= 5
            assert statistics.target_count == 10_000
            assert statistics.on_disk_bytes / statistics.target_count <= 1024
            assert statistics.logical_bytes > statistics.on_disk_bytes
            assert statistics.compression_ratio > 1.0
            assert dumped["version"] == statistics.schema_version
            assert len(dumped["targets"]) == 10_000
            assert store.dump_json() == store.dump_json()
            assert list(path.parent.glob("*.json")) == []

    def test_truncated_and_oversized_compressed_stores_fail_closed(self) -> None:
        """Compact decoding rejects damaged data and configured expansion limits."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training-v1.store"
            store = TrainingStore(path)
            store.save(state())
            original = path.read_bytes()
            path.write_bytes(original[:-3])
            with self.assertRaisesRegex(TrainingStoreError, "malformed"):
                store.load()

            path.write_bytes(original)
            constrained = TrainingStore(path, maximum_logical_bytes=16)
            with self.assertRaisesRegex(TrainingStoreError, "decompressed size"):
                constrained.load()

    def test_store_rejects_unbounded_validated_model_anchors(self) -> None:
        """Real training results remain bounded without becoming user profiles."""
        template = state().anchors[0]
        oversized = TrainingState(
            anchors=[
                ModelAnchor(
                    index,
                    template.camera_id,
                    template.feature_schema,
                    template.topology_id,
                    template.outputs,
                    template.context_centroid,
                    template.context_variance,
                    template.model,
                    template.median_error,
                    template.edge_error,
                )
                for index in range(MAXIMUM_MODEL_ANCHORS + 1)
            ]
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(TrainingStoreError, "model-anchor limit"),
        ):
            TrainingStore(Path(temporary) / "training-v1.json").save(oversized)

    def test_directory_permissions_are_checked_before_loading(self) -> None:
        """A shared training directory cannot expose target-level features."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "gazeebo"
            directory.mkdir(mode=0o700)
            path = directory / "training-v1.json"
            path.write_text('{"version":1}', encoding="utf-8")
            path.chmod(0o600)
            directory.chmod(0o755)
            with self.assertRaisesRegex(TrainingStoreError, "owner-only"):
                TrainingStore(path).load()
            directory.chmod(0o700)


if __name__ == "__main__":
    unittest.main()
