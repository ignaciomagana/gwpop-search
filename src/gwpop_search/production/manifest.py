"""Canonical production data manifests and checksum validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactEntry:
    role: str
    path: str
    sha256: str
    size_bytes: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in {"pe", "selection", "auxiliary"}:
            raise ValueError("artifact role must be pe, selection, or auxiliary")
        if not self.path:
            raise ValueError("artifact path cannot be empty")
        digest = self.sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("artifact sha256 must be a 64-character hex digest")
        object.__setattr__(self, "sha256", digest)
        if self.size_bytes < 0:
            raise ValueError("artifact size_bytes cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ArtifactEntry":
        return cls(
            role=str(payload["role"]),
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    coordinate_basis_identity: str
    event_names: tuple[str, ...]
    event_selection: Mapping[str, object]
    waveform_policy: Mapping[str, object]
    artifacts: tuple[ArtifactEntry, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    format_version: str = "gwpop-search-dataset-manifest-1.0"

    def __post_init__(self) -> None:
        if self.format_version != "gwpop-search-dataset-manifest-1.0":
            raise ValueError("unsupported dataset manifest format")
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")
        if not self.coordinate_basis_identity:
            raise ValueError("coordinate_basis_identity cannot be empty")
        if not self.event_names:
            raise ValueError("dataset manifest requires at least one event")
        if len(set(self.event_names)) != len(self.event_names):
            raise ValueError("event_names must be unique")
        roles = [item.role for item in self.artifacts]
        if "pe" not in roles or "selection" not in roles:
            raise ValueError("dataset manifest requires PE and selection artifacts")

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "dataset_id": self.dataset_id,
            "coordinate_basis_identity": self.coordinate_basis_identity,
            "event_names": list(self.event_names),
            "event_selection": dict(self.event_selection),
            "waveform_policy": dict(self.waveform_policy),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "metadata": dict(self.metadata),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def manifest_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DatasetManifest":
        return cls(
            dataset_id=str(payload["dataset_id"]),
            coordinate_basis_identity=str(payload["coordinate_basis_identity"]),
            event_names=tuple(str(x) for x in payload["event_names"]),
            event_selection=dict(payload["event_selection"]),
            waveform_policy=dict(payload["waveform_policy"]),
            artifacts=tuple(
                ArtifactEntry.from_dict(item)
                for item in payload["artifacts"]
            ),
            metadata=dict(payload.get("metadata", {})),
            format_version=str(
                payload.get(
                    "format_version",
                    "gwpop-search-dataset-manifest-1.0",
                )
            ),
        )


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    return DatasetManifest.from_dict(json.loads(Path(path).read_text()))


def save_dataset_manifest(path: str | Path, manifest: DatasetManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2))


def artifact_entry_from_file(
    path: str | Path,
    *,
    role: str,
    stored_path: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ArtifactEntry:
    path = Path(path)
    stat = path.stat()
    return ArtifactEntry(
        role=role,
        path=str(path if stored_path is None else stored_path),
        sha256=sha256_file(path),
        size_bytes=int(stat.st_size),
        metadata={} if metadata is None else dict(metadata),
    )


def validate_dataset_manifest_files(
    manifest: DatasetManifest,
    *,
    base_dir: str | Path = ".",
) -> dict[str, object]:
    """Verify every frozen artifact checksum and size against local files."""
    base_dir = Path(base_dir)
    results = []
    all_valid = True
    for entry in manifest.artifacts:
        path = Path(entry.path)
        if not path.is_absolute():
            path = base_dir / path
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_sha = sha256_file(path) if exists else None
        valid = bool(
            exists
            and actual_size == entry.size_bytes
            and actual_sha == entry.sha256
        )
        all_valid &= valid
        results.append(
            {
                "role": entry.role,
                "path": entry.path,
                "exists": exists,
                "expected_size_bytes": entry.size_bytes,
                "actual_size_bytes": actual_size,
                "expected_sha256": entry.sha256,
                "actual_sha256": actual_sha,
                "valid": valid,
            }
        )
    return {
        "dataset_id": manifest.dataset_id,
        "manifest_hash": manifest.manifest_hash,
        "valid": bool(all_valid),
        "artifacts": results,
    }
