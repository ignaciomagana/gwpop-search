"""Builders for explicit dataset and production campaign freezes."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from gwpop_search.data import PosteriorCatalog, SelectionCatalog
from gwpop_search.grammar import ModelGraph, baseline_hyperprior_profile
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.search import SchedulerConfig

from .config import (
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
    installed_sampler_backend,
)
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
    sampler_backend: Mapping[str, str] | None = None,
    require_root_profile: str | None = None,
    graph_file_sha256: str | None = None,
) -> ProductionCampaignConfig:
    """Freeze a v2 campaign; the sampler pin defaults to the installed dynesty.

    ``require_root_profile`` asserts which registered root hyperprior profile
    the graph was enumerated under (GWTC-5 production: ``"gwtc5-v1"``). The
    profile is implicit in ``graph.root_hash`` because priors are part of every
    model hash; naming it here refuses a graph frozen under the wrong
    hyperpriors before any compute is spent.

    ``graph_file_sha256`` (``verify_graph_file(path)["file_sha256"]``) records
    the whole graph file, including its descriptive metadata; it is required
    for a v2 graph, whose metadata (atom labels, DRAFT priors, G12 record)
    feeds the claim table.
    """
    if require_root_profile is not None:
        profile = baseline_hyperprior_profile(graph.by_hash[graph.root_hash])
        if profile != require_root_profile:
            found = "no registered profile" if profile is None else repr(profile)
            raise ValueError(
                f"model graph root is {found}, not the required "
                f"{require_root_profile!r} hyperprior profile; re-enumerate the "
                f"graph with --hyperprior-profile {require_root_profile}"
            )
    root_spec = graph.by_hash[graph.root_hash]
    from gwpop_search.grammar.v2_structure import is_v2_model

    if is_v2_model(root_spec) and fidelity.hbi.variance_taper is None:
        # v2 spec (plan 2026-09-30, Numerics): the sigma^2_lnL taper lives inside
        # the likelihood; a v2 campaign without it would score a different model
        raise ValueError(
            "a v2 model graph needs a fidelity configuration whose HBI likelihood "
            "carries the variance taper (inference.v2_numerics.v2_fidelity_run_config)"
        )
    if is_v2_model(root_spec) and graph_file_sha256 is None:
        raise ValueError(
            "a v2 campaign must record the graph file's sha256 (graph_file_sha256 = "
            "verify_graph_file(path)['file_sha256']): its metadata feeds the claim table"
        )
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
        sampler_backend=(
            installed_sampler_backend() if sampler_backend is None else dict(sampler_backend)
        ),
        model_graph_file_sha256=graph_file_sha256,
    )
