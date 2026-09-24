"""Production freeze manifests and campaign configuration."""

from .builders import (
    build_dataset_manifest_from_canonical_files,
    build_production_campaign,
)
from .completion import complete_graph_evidence
from .config import (
    PRODUCTION_CAMPAIGN_FORMAT_VERSION,
    LegacyCampaignError,
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
    installed_sampler_backend,
    load_production_campaign,
    save_production_campaign,
    validate_sampler_backend,
)
from .freeze import canonical_graph_json, model_graph_hash, verify_graph_file
from .runner import (
    collect_best_available_evidence,
    evidence_from_evaluation,
    load_frozen_dataset,
    model_prior_from_config,
    run_production_search,
    write_scientific_scoring,
)
from .validate import sampler_backend_status, validate_production_freeze
from .spec_evaluation import (
    MODEL_SPEC_EVALUATION_FORMAT,
    model_spec_diff,
    model_spec_provenance_path,
    model_spec_run_dir,
    run_model_spec_evaluation,
    with_prior,
)
from .manifest import (
    ArtifactEntry,
    DatasetManifest,
    artifact_entry_from_file,
    load_dataset_manifest,
    save_dataset_manifest,
    sha256_file,
    validate_dataset_manifest_files,
)

__all__ = [
    "LegacyCampaignError",
    "MODEL_SPEC_EVALUATION_FORMAT",
    "model_spec_diff",
    "model_spec_provenance_path",
    "model_spec_run_dir",
    "run_model_spec_evaluation",
    "with_prior",
    "PRODUCTION_CAMPAIGN_FORMAT_VERSION",
    "evidence_from_evaluation",
    "installed_sampler_backend",
    "sampler_backend_status",
    "validate_sampler_backend",
    "ArtifactEntry",
    "build_dataset_manifest_from_canonical_files",
    "build_production_campaign",
    "DatasetManifest",
    "ProductionCampaignConfig",
    "SearchBudget",
    "SeedPolicy",
    "artifact_entry_from_file",
    "canonical_graph_json",
    "collect_best_available_evidence",
    "complete_graph_evidence",
    "load_dataset_manifest",
    "load_frozen_dataset",
    "load_production_campaign",
    "model_graph_hash",
    "model_prior_from_config",
    "run_production_search",
    "save_dataset_manifest",
    "save_production_campaign",
    "sha256_file",
    "validate_dataset_manifest_files",
    "validate_production_freeze",
    "verify_graph_file",
    "write_scientific_scoring",
]
