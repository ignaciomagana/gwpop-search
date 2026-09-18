"""Builders for explicit dataset and production campaign freezes."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from gwpop_search.data import PosteriorCatalog, SelectionCatalog
from gwpop_search.grammar import ModelGraph
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.search import SchedulerConfig

from .config import ProductionCampaignConfig, SearchBudget, SeedPolicy
from .freeze import model_graph_hash
from .manifest import (
    DatasetManifest,
    artifact_entry_from_file,
)


def build_dataset_manifest_from_canonical_files(
    pe_path: str | Path,
    selection_path: str | Path,
    *,
    dataset_id: str,
    event_selection: Mapping[str, object],
    waveform_policy: Mapping[str, object],
    stored_pe_path: str | None = None,
    stored_selection_path: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DatasetManifest:
    pe_path = Path(pe_path)
    selection_path = Path(selection_path)
    posterior = PosteriorCatalog.from_hdf5(pe_path)
    selection = SelectionCatalog.from_hdf5(selection_path)

    if posterior.basis.identity != selection.basis.identity:
        raise ValueError(
            "cannot freeze dataset: PE and selection basis identities differ"
        )

    return DatasetManifest(
        dataset_id=str(dataset_id),
        coordinate_basis_identity=posterior.basis.identity,
        event_names=posterior.event_names,
        event_selection=dict(event_selection),
        waveform_policy=dict(waveform_policy),
        artifacts=(
            artifact_entry_from_file(
                pe_path,
                role="pe",
                stored_path=stored_pe_path,
            ),
            artifact_entry_from_file(
                selection_path,
                role="selection",
                stored_path=stored_selection_path,
            ),
        ),
        metadata={} if metadata is None else dict(metadata),
    )


def build_production_campaign(
    manifest: DatasetManifest,
    graph: ModelGraph,
    *,
    campaign_id: str,
    git_commit: str,
    model_prior: Mapping[str, object],
    fidelity: FidelityRunConfig,
    scheduler: SchedulerConfig,
    seed_policy: SeedPolicy,
    budget: SearchBudget,
    artifact_root: str,
    state_database: str,
    agents_enabled: bool = False,
) -> ProductionCampaignConfig:
    return ProductionCampaignConfig(
        campaign_id=campaign_id,
        dataset_manifest_hash=manifest.manifest_hash,
        model_graph_hash=model_graph_hash(graph),
        model_graph_root_hash=graph.root_hash,
        git_commit=git_commit,
        model_prior=dict(model_prior),
        fidelity=fidelity,
        scheduler=scheduler,
        seed_policy=seed_policy,
        budget=budget,
        artifact_root=artifact_root,
        state_database=state_database,
        agents_enabled=agents_enabled,
    )
