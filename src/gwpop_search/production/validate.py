"""Cross-validation of frozen production data, graph, code, campaign and sampler pin."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from gwpop_search.grammar import HYPERPRIOR_PROFILES, baseline_model_spec
from gwpop_search.inference.numpyro import _code_identity

from .config import ProductionCampaignConfig


def root_hyperprior_profile_for_hash(root_hash: str | None) -> str | None:
    """Registered root hyperprior profile with this baseline hash, else None.

    The profile is fully determined by the graph root hash (priors are part of
    every ``ModelSpec`` hash), so a frozen campaign already carries it. Naming
    it makes the freeze report state which hyperpriors the graph was enumerated
    under (GWTC-5 production: ``gwtc5-v1``).
    """
    if not root_hash:
        return None
    for profile in HYPERPRIOR_PROFILES:
        if baseline_model_spec(profile).model_hash == root_hash:
            return profile
    return None


def _installed_version(package: str) -> str | None:
    try:
        return str(importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        return None


def sampler_backend_status(campaign: ProductionCampaignConfig) -> dict[str, object]:
    """Frozen versus installed sampler backend (and the recorded JAX stack)."""
    frozen = dict(campaign.sampler_backend)
    installed = _installed_version(frozen["name"])
    return {
        "frozen": frozen,
        "installed_version": installed,
        "matches": installed == frozen["version"],
        "jax_version": _installed_version("jax"),
        "jaxlib_version": _installed_version("jaxlib"),
    }
from .freeze import verify_graph_file
from .manifest import DatasetManifest, validate_dataset_manifest_files


def validate_production_freeze(
    manifest: DatasetManifest,
    graph_path: str | Path,
    campaign: ProductionCampaignConfig,
    *,
    data_base_dir: str | Path = ".",
    require_current_commit: bool = True,
) -> dict[str, object]:
    data = validate_dataset_manifest_files(
        manifest,
        base_dir=data_base_dir,
    )
    graph = verify_graph_file(graph_path)
    code = _code_identity()
    backend = sampler_backend_status(campaign)
    root_profile = root_hyperprior_profile_for_hash(graph.get("root_hash"))

    checks = {
        "dataset_files_valid": bool(data["valid"]),
        "dataset_manifest_hash_matches": (
            campaign.dataset_manifest_hash == manifest.manifest_hash
        ),
        "model_graph_hash_matches": (
            campaign.model_graph_hash == graph["graph_hash"]
        ),
        "model_graph_root_matches": (
            campaign.model_graph_root_hash == graph["root_hash"]
        ),
        # the whole file (descriptive metadata included) when the campaign
        # recorded it; campaigns frozen before the field existed carry None
        "model_graph_file_sha256_matches": (
            campaign.model_graph_file_sha256 is None
            or campaign.model_graph_file_sha256 == graph["file_sha256"]
        ),
        "model_graph_file_sha256_recorded_for_v2": (
            root_profile != "gwtc5-v2" or campaign.model_graph_file_sha256 is not None
        ),
        "git_commit_matches": (
            not require_current_commit
            or (
                code["git_commit"] != "unknown"
                and campaign.git_commit == code["git_commit"]
            )
        ),
        "sampler_backend_pin_matches": bool(backend["matches"]),
        # A production graph must be rooted at a registered baseline hyperprior
        # profile; an unregistered root means the enumeration drifted from the
        # declared grammar and the null calibration would have no defined null.
        "model_graph_root_is_registered_baseline": root_profile is not None,
    }
    return {
        "valid": bool(all(checks.values())),
        "checks": checks,
        "dataset": data,
        "graph": graph,
        "code": code,
        "sampler_backend": backend,
        "root_hyperprior_profile": root_profile,
        "campaign_id": campaign.campaign_id,
        "campaign_hash": campaign.campaign_hash,
    }
