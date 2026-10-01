"""Nearby-baseline reruns for deterministic search robustness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

from gwpop_search.grammar import (
    DEFAULT_MUTATIONS,
    FOLLOWUP_MUTATIONS,
    ModelGraph,
    ModelSpec,
    enumerate_model_graph,
)
from gwpop_search.grammar.enumerate import ModelEdge
from gwpop_search.grammar.mutations import (
    InapplicableMutation,
    MutationSpec,
    apply_mutation,
)
from gwpop_search.grammar.paths import graph_edge_keys, path_edge_key
from gwpop_search.inference.fidelity import DeterministicHBIEvaluator
from gwpop_search.inference.numpyro import _code_identity
from gwpop_search.production import ProductionCampaignConfig
from gwpop_search.production.runner import (
    collect_best_available_evidence,
    model_prior_from_config,
    write_scientific_scoring,
)
from gwpop_search.search import (
    Fidelity,
    ModelEvidence,
    SearchExecutionConfig,
    evaluation_seed,
    execute_search,
)

from .stress import edge_log_bayes_factors


_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
SUITE_FORMAT = "gwpop-search-nearby-baseline-suite-1.1"
#: written when any scenario is restricted to declared mutation paths (a 1.1
#: reader would silently enumerate the full neighbourhood instead)
RESTRICTED_SUITE_FORMAT = "gwpop-search-nearby-baseline-suite-1.2"
_READABLE_SUITE_FORMATS = (
    "gwpop-search-nearby-baseline-suite-1.0",
    SUITE_FORMAT,
    RESTRICTED_SUITE_FORMAT,
)

#: Named mutation catalogues a restricted scenario resolves its mutation ids
#: against. Grammar extensions (e.g. the v2 atoms) register theirs with
#: :func:`register_mutation_catalogue`; the name is stored in the scenario so a
#: saved suite is replayable.
MUTATION_CATALOGUES: dict[str, tuple[MutationSpec, ...]] = {
    "default": tuple(DEFAULT_MUTATIONS),
    "default+followup": tuple(DEFAULT_MUTATIONS) + tuple(FOLLOWUP_MUTATIONS),
}

#: the v2 chi_eff atoms every D5 alternative root must run (plan 2026-09-30)
V2_CHI_EFF_ATOMS = ("C1", "C2", "C3", "C4", "C5", "C6", "S1", "S2", "S3", "S4")


def _register_v2_catalogue() -> None:
    from gwpop_search.grammar.v2 import V2_MUTATIONS, V2_PROFILE

    MUTATION_CATALOGUES[V2_PROFILE] = tuple(V2_MUTATIONS)


_register_v2_catalogue()


def v2_chi_eff_atom_mutations() -> dict[str, str]:
    """Plan label -> mutation id of the ten v2 chi_eff atoms (each one mutation of R0)."""
    from gwpop_search.grammar.v2 import V2_ATOM_IDS

    return {label: V2_ATOM_IDS[label] for label in V2_CHI_EFF_ATOMS}


def register_mutation_catalogue(name: str, mutations) -> None:
    """Register (or identically re-register) a named mutation catalogue."""
    name = str(name)
    if not name or not _ID.match(name.replace("+", "_")):
        raise ValueError("catalogue name must contain only letters, numbers, _, ., -, +")
    items = tuple(mutations)
    ids = [item.mutation_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"catalogue {name!r} has duplicate mutation ids")
    existing = MUTATION_CATALOGUES.get(name)
    if existing is not None and existing != items:
        raise ValueError(f"mutation catalogue {name!r} is already registered with different mutations")
    MUTATION_CATALOGUES[name] = items


def mutation_catalogue(name: str) -> dict[str, MutationSpec]:
    try:
        items = MUTATION_CATALOGUES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown mutation catalogue {name!r}; registered: {sorted(MUTATION_CATALOGUES)}"
        ) from exc
    return {item.mutation_id: item for item in items}
PLAN_FORMAT = "gwpop-search-nearby-baseline-plan-1.1"
SUMMARY_FORMAT = "gwpop-search-nearby-baseline-summary-1.1"
MODEL_EVALUATION_FORMAT = "gwpop-search-nearby-model-evaluation-1.0"
MODEL_LIST_FORMAT = "gwpop-search-nearby-scenario-models-1.0"


@dataclass(frozen=True)
class NearbyBaselineScenario:
    scenario_id: str
    root_spec: ModelSpec
    max_depth: int = 1
    max_models: int = 20
    note: str = ""
    #: ``None``: the full depth-``max_depth`` neighbourhood of the root (the
    #: v1 behaviour). Otherwise the scenario runs exactly the root plus the
    #: nodes reached by these ordered mutation paths (and their prefixes),
    #: e.g. the v2 D5 restriction "root + candidates + all 10 chi_eff atoms".
    mutation_paths: tuple[tuple[str, ...], ...] | None = None
    mutation_catalogue: str = "default"

    def __post_init__(self) -> None:
        if not self.scenario_id or not _ID.match(self.scenario_id):
            raise ValueError(
                "scenario_id must contain only letters, numbers, _, ., or -"
            )
        if self.max_depth < 0:
            raise ValueError("max_depth cannot be negative")
        if self.max_models <= 0:
            raise ValueError("max_models must be positive")
        if self.mutation_paths is not None:
            paths = []
            for path in self.mutation_paths:
                path = (path,) if isinstance(path, str) else tuple(str(x) for x in path)
                if not path or len(set(path)) != len(path):
                    raise ValueError(f"mutation path {path!r} must be non-empty without repeats")
                if len(path) > self.max_depth:
                    raise ValueError(
                        f"mutation path {path!r} is deeper than max_depth={self.max_depth}"
                    )
                paths.append(path)
            if not paths:
                raise ValueError("a restricted scenario needs at least one mutation path")
            if len(set(paths)) != len(paths):
                raise ValueError("mutation paths must be unique")
            object.__setattr__(self, "mutation_paths", tuple(sorted(paths)))
            mutation_catalogue(self.mutation_catalogue)  # must be registered

    @property
    def restricted(self) -> bool:
        return self.mutation_paths is not None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "scenario_id": self.scenario_id,
            "root_spec": self.root_spec.to_dict(),
            "max_depth": int(self.max_depth),
            "max_models": int(self.max_models),
            "note": self.note,
        }
        if self.mutation_paths is not None:
            # written only for restricted scenarios, so unrestricted suites keep
            # their 1.1 serialisation (and plan) byte for byte
            payload["mutation_paths"] = [list(path) for path in self.mutation_paths]
            payload["mutation_catalogue"] = self.mutation_catalogue
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "NearbyBaselineScenario":
        paths = payload.get("mutation_paths")
        return cls(
            scenario_id=str(payload["scenario_id"]),
            root_spec=ModelSpec.from_dict(dict(payload["root_spec"])),
            max_depth=int(payload.get("max_depth", 1)),
            max_models=int(payload.get("max_models", 20)),
            note=str(payload.get("note", "")),
            mutation_paths=(
                None if paths is None else tuple(tuple(str(x) for x in p) for p in paths)
            ),
            mutation_catalogue=str(payload.get("mutation_catalogue", "default")),
        )


@dataclass(frozen=True)
class NearbyBaselineConfig:
    stop_fidelity: Fidelity = Fidelity.F3_EVIDENCE
    max_gpu_hours_per_scenario: float = 250.0
    max_f3_models: int = 20
    max_f4_models: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stop_fidelity",
            Fidelity(self.stop_fidelity),
        )
        if self.stop_fidelity not in (Fidelity.F3_EVIDENCE, Fidelity.F4_PRODUCTION):
            raise ValueError(
                "nearby-baseline suite must stop at an evidence rung of fidelity ladder v2 "
                f"(F3 or F4); got {self.stop_fidelity.value} (F2 is not in the ladder)"
            )
        if (
            not math.isfinite(self.max_gpu_hours_per_scenario)
            or self.max_gpu_hours_per_scenario <= 0.0
        ):
            raise ValueError("max_gpu_hours_per_scenario must be positive")
        if self.max_f3_models <= 0 or self.max_f4_models <= 0:
            raise ValueError("nearby-baseline model limits must be positive")


@dataclass(frozen=True)
class NearbyBaselineSuiteSpec:
    scenarios: tuple[NearbyBaselineScenario, ...]
    config: NearbyBaselineConfig
    format_version: str = SUITE_FORMAT

    def __post_init__(self) -> None:
        if self.format_version not in _READABLE_SUITE_FORMATS:
            raise ValueError("unsupported nearby-baseline suite format")
        scenarios = tuple(self.scenarios)
        # 1.0 specs are read (their stop fidelity is re-validated) and written as
        # 1.1; a suite with a restricted scenario is written as 1.2
        object.__setattr__(
            self,
            "format_version",
            RESTRICTED_SUITE_FORMAT
            if any(item.restricted for item in scenarios)
            else SUITE_FORMAT,
        )
        if not scenarios:
            raise ValueError("nearby-baseline suite requires scenarios")
        ids = [item.scenario_id for item in scenarios]
        if len(set(ids)) != len(ids):
            raise ValueError("nearby-baseline scenario IDs must be unique")
        object.__setattr__(self, "scenarios", scenarios)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "config": {
                "stop_fidelity": self.config.stop_fidelity.value,
                "max_gpu_hours_per_scenario": (
                    self.config.max_gpu_hours_per_scenario
                ),
                "max_f3_models": self.config.max_f3_models,
                "max_f4_models": self.config.max_f4_models,
            },
            "scenarios": [item.to_dict() for item in self.scenarios],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "NearbyBaselineSuiteSpec":
        config = dict(payload["config"])
        return cls(
            scenarios=tuple(
                NearbyBaselineScenario.from_dict(item)
                for item in payload["scenarios"]
            ),
            config=NearbyBaselineConfig(
                stop_fidelity=Fidelity(str(config["stop_fidelity"])),
                max_gpu_hours_per_scenario=float(
                    config["max_gpu_hours_per_scenario"]
                ),
                max_f3_models=int(config["max_f3_models"]),
                max_f4_models=int(config["max_f4_models"]),
            ),
            format_version=str(
                payload.get(
                    "format_version",
                    "gwpop-search-nearby-baseline-suite-1.0",
                )
            ),
        )


def save_nearby_baseline_suite_spec(
    path: str | Path,
    spec: NearbyBaselineSuiteSpec,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), sort_keys=True, indent=2))


def load_nearby_baseline_suite_spec(
    path: str | Path,
) -> NearbyBaselineSuiteSpec:
    return NearbyBaselineSuiteSpec.from_dict(json.loads(Path(path).read_text()))


def nearby_baseline_seed(root_seed: int, scenario_id: str) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:{scenario_id}:nearby-baseline-v1".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def nearby_dataset_identity(
    base_dataset_identity: str,
    scenario: NearbyBaselineScenario,
) -> str:
    payload = {
        "base_dataset_identity": str(base_dataset_identity),
        "scenario_id": scenario.scenario_id,
        "root_model_hash": scenario.root_spec.model_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def mutation_log_bayes_factors(
    graph: ModelGraph,
    evidences: Mapping[str, ModelEvidence],
) -> dict[str, float]:
    """Maximum evaluated edge BF for each registered mutation."""
    by_mutation: dict[str, list[float]] = {}
    for row in edge_log_bayes_factors(graph, evidences).values():
        by_mutation.setdefault(
            str(row["mutation_id"]),
            [],
        ).append(float(row["log_bayes_factor"]))
    return {
        mutation_id: float(max(values))
        for mutation_id, values in sorted(by_mutation.items())
    }


def compare_mutation_support(
    reference: Mapping[str, float],
    scenario: Mapping[str, float],
) -> dict[str, object]:
    mutations = sorted(set(reference) | set(scenario))
    rows = []
    for mutation in mutations:
        ref = reference.get(mutation)
        alt = scenario.get(mutation)
        rows.append(
            {
                "mutation_id": mutation,
                "reference_max_log_bayes_factor": ref,
                "scenario_max_log_bayes_factor": alt,
                "delta_log_bayes_factor": (
                    None
                    if ref is None or alt is None
                    else float(alt - ref)
                ),
            }
        )
    comparable = [
        row
        for row in rows
        if row["delta_log_bayes_factor"] is not None
    ]
    return {
        "n_common_mutations": len(comparable),
        "max_abs_delta_log_bayes_factor": (
            None
            if not comparable
            else float(
                max(
                    abs(row["delta_log_bayes_factor"])
                    for row in comparable
                )
            )
        ),
        "mutations": rows,
    }


def build_nearby_baseline_plan(
    *,
    campaign: ProductionCampaignConfig,
    base_dataset_identity: str,
    reference_graph: ModelGraph,
    suite: NearbyBaselineSuiteSpec,
) -> dict[str, object]:
    ladder = getattr(campaign.scheduler, "ladder", None)
    return {
        "format_version": PLAN_FORMAT,
        "code": _code_identity(),
        "sampler_backend": "dynesty",
        "ladder": None if ladder is None else [Fidelity(item).value for item in ladder],
        "campaign_hash": campaign.campaign_hash,
        "base_dataset_identity": str(base_dataset_identity),
        "reference_graph_root_hash": reference_graph.root_hash,
        "suite": suite.to_dict(),
    }


def _write_plan_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(
                "existing nearby-baseline plan does not match request"
            )
    else:
        path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def restricted_model_graph(
    root: ModelSpec,
    paths,
    mutations: Mapping[str, MutationSpec],
    *,
    max_models: int | None = None,
) -> tuple[ModelGraph, tuple[str, ...]]:
    """Root plus the nodes reached by ordered mutation ``paths`` (and their prefixes).

    Returns the graph and the edge keys (:mod:`gwpop_search.grammar.paths`) of
    the paths whose mutations are inapplicable on this root (for example the
    beta-per-component atom on a root that already has it); those paths are
    skipped, not failed. Unknown mutation ids raise.
    """
    nodes: list[ModelSpec] = [root]
    by_hash: dict[str, ModelSpec] = {root.model_hash: root}
    depths: dict[str, int] = {root.model_hash: 0}
    edges: list[ModelEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()
    inapplicable: list[str] = []
    ordered = sorted({tuple(p) for p in paths}, key=lambda p: (len(p), p))
    for path in ordered:
        unknown = [m for m in path if m not in mutations]
        if unknown:
            raise ValueError(f"mutation path {path} uses unknown mutation id(s) {unknown}")
    for path in ordered:
        parent = root
        for depth, mutation_id in enumerate(path, start=1):
            try:
                child = apply_mutation(parent, mutations[mutation_id])
            except InapplicableMutation:
                inapplicable.append(path_edge_key(path[:depth]))
                break
            except ValueError as exc:
                # a grammar defect (the child is not a valid model), not an
                # inapplicable atom: fail loudly at planning time, naming the path
                raise ValueError(
                    f"mutation path {path}: {mutation_id} produces an invalid model on "
                    f"root {root.model_hash[:12]}: {exc}"
                ) from exc
            if child.model_hash not in by_hash:
                if max_models is not None and len(nodes) >= int(max_models):
                    raise ValueError(
                        f"restricted graph needs more than max_models={max_models} models"
                    )
                by_hash[child.model_hash] = child
                depths[child.model_hash] = depth
                nodes.append(child)
            key = (parent.model_hash, child.model_hash, mutation_id)
            if key not in edge_keys:
                edge_keys.add(key)
                edges.append(
                    ModelEdge(
                        parent_hash=parent.model_hash,
                        child_hash=child.model_hash,
                        mutation_id=mutation_id,
                        depth=depth,
                    )
                )
            parent = child
    graph = ModelGraph(
        root_hash=root.model_hash,
        nodes=tuple(nodes),
        edges=tuple(edges),
        depths=depths,
    )
    return graph, tuple(sorted(set(inapplicable)))


def nearby_scenario_graph(scenario: NearbyBaselineScenario) -> ModelGraph:
    """The alternative-root graph neighbourhood one scenario searches.

    The suite and the per-model evaluation both enumerate it here, so they
    cannot disagree on which models (and which model hashes) a scenario holds.
    A restricted scenario (``mutation_paths``) holds exactly the root and the
    nodes of its declared paths.
    """
    if scenario.restricted:
        graph, _ = restricted_model_graph(
            scenario.root_spec,
            scenario.mutation_paths,
            mutation_catalogue(scenario.mutation_catalogue),
            max_models=scenario.max_models,
        )
        return graph
    return enumerate_model_graph(
        scenario.root_spec,
        max_depth=scenario.max_depth,
        max_models=scenario.max_models,
    )


def nearby_scenario_inapplicable_paths(scenario: NearbyBaselineScenario) -> tuple[str, ...]:
    """Edge keys of a restricted scenario's paths that do not apply to its root."""
    if not scenario.restricted:
        return ()
    _, inapplicable = restricted_model_graph(
        scenario.root_spec,
        scenario.mutation_paths,
        mutation_catalogue(scenario.mutation_catalogue),
        max_models=scenario.max_models,
    )
    return inapplicable


def v2_alt_root_scenario(
    scenario_id: str,
    root_spec: ModelSpec,
    *,
    candidate_paths,
    chi_eff_mutation_ids: Mapping[str, object],
    mutation_catalogue_name: str = "default",
    note: str = "",
) -> NearbyBaselineScenario:
    """The v2 D5 scenario: an alternative root + the candidates + all 10 chi_eff atoms.

    ``chi_eff_mutation_ids`` maps every plan atom label of
    :data:`V2_CHI_EFF_ATOMS` (C1-C6, S1-S4) to its mutation id, or to an
    ordered mutation path when the grammar builds the atom in more than one
    step (e.g. a mixture followed by its fraction law); all ten are required
    (plan 2026-09-30, D5). ``candidate_paths`` are the ordered
    mutation paths of the candidate edges from the searched graph
    (``"<id>"`` for a depth-1 atom, ``(a, b)`` for the depth-2 edge applying
    ``b`` to the root + ``a`` node). Nothing else of the alternative root's
    neighbourhood is run.
    """
    labels = sorted(chi_eff_mutation_ids)
    if tuple(sorted(V2_CHI_EFF_ATOMS)) != tuple(labels):
        raise ValueError(
            f"D5 needs exactly the chi_eff atoms {V2_CHI_EFF_ATOMS}; got {labels}"
        )
    def _as_path(value) -> tuple[str, ...]:
        return (str(value),) if isinstance(value, str) else tuple(str(x) for x in value)

    atom_paths = [_as_path(chi_eff_mutation_ids[label]) for label in V2_CHI_EFF_ATOMS]
    if len(set(atom_paths)) != len(atom_paths):
        raise ValueError("the chi_eff atoms must map to distinct mutation paths")
    paths: set[tuple[str, ...]] = set()
    for path in [*atom_paths, *(_as_path(p) for p in candidate_paths)]:
        if not path:
            raise ValueError("empty mutation path")
        paths.add(path)
        paths.update(path[:k] for k in range(1, len(path)))
    depth = max(len(p) for p in paths)
    return NearbyBaselineScenario(
        scenario_id=scenario_id,
        root_spec=root_spec,
        max_depth=depth,
        max_models=1 + len(paths),
        note=note or "v2 D5: alternative root + candidates + all 10 chi_eff atoms",
        mutation_paths=tuple(sorted(paths)),
        mutation_catalogue=mutation_catalogue_name,
    )


def root_edge_log_bayes_factors(
    graph: ModelGraph,
    evidences: Mapping[str, ModelEvidence],
) -> dict[str, float]:
    """Signed ``ln BF`` of every evaluated edge, keyed by its root-independent edge key."""
    keys = graph_edge_keys(graph)
    out: dict[str, float] = {}
    for edge in graph.edges:
        if edge.parent_hash not in evidences or edge.child_hash not in evidences:
            continue
        key = keys.get((edge.parent_hash, edge.child_hash, edge.mutation_id))
        if key is None:
            continue
        out[key] = float(
            evidences[edge.child_hash].log_evidence
            - evidences[edge.parent_hash].log_evidence
        )
    return dict(sorted(out.items()))


def nearby_scenario_model_run_dir(
    root: str | Path,
    scenario_id: str,
    model_hash: str,
    fidelity: Fidelity,
) -> Path:
    """The run directory the suite uses for one scenario/model evaluation.

    ``root`` is the SUITE root; the suite writes each scenario under
    ``<root>/<scenario_id>/artifacts/<fidelity>/<model hash>``.
    """
    return (
        Path(root)
        / str(scenario_id)
        / "artifacts"
        / Fidelity(fidelity).value
        / str(model_hash)
    )


def nearby_scenario_models(
    scenario: NearbyBaselineScenario,
    config: NearbyBaselineConfig | None = None,
) -> dict[str, object]:
    """Which models one scenario evaluates, in the order the suite does.

    The suite's search evaluates every graph model at F0, then every model
    that passes F0 at F3, each rung in sorted model-hash order (F0 promotes
    every passing model; the beam only applies above F0). So the F3 candidates
    are all graph models, and which of them actually reach F3 is decided by
    F0 inside the suite. With ``stop_fidelity=F3`` nothing is evaluated above
    F3. If more than ``max_f3_models`` models pass F0 the suite raises
    :class:`~gwpop_search.search.SearchBudgetExceeded` before any F3 run.
    """
    graph = nearby_scenario_graph(scenario)
    mutation_by_child: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.parent_hash == graph.root_hash:
            mutation_by_child.setdefault(edge.child_hash, []).append(
                edge.mutation_id
            )
    order = sorted(model.model_hash for model in graph.nodes)
    models = [
        {
            "model_hash": model_hash,
            "depth": int(graph.depths[model_hash]),
            "is_root": model_hash == graph.root_hash,
            "root_mutation_ids": sorted(mutation_by_child.get(model_hash, [])),
        }
        for model_hash in order
    ]
    payload: dict[str, object] = {
        "format_version": MODEL_LIST_FORMAT,
        "scenario_id": scenario.scenario_id,
        "root_model_hash": graph.root_hash,
        "max_depth": int(scenario.max_depth),
        "max_models": int(scenario.max_models),
        "n_graph_models": len(graph.nodes),
        "evaluation_order": "sorted_model_hash",
        "models": models,
    }
    if config is not None:
        payload["stop_fidelity"] = config.stop_fidelity.value
        payload["max_f3_models"] = int(config.max_f3_models)
        payload["f3_budget_fits_all_models"] = (
            len(graph.nodes) <= int(config.max_f3_models)
        )
    return payload


def _nearby_evaluator(
    posterior,
    selection,
    campaign: ProductionCampaignConfig,
    *,
    base_dataset_identity: str,
    scenario: NearbyBaselineScenario,
) -> DeterministicHBIEvaluator:
    return DeterministicHBIEvaluator(
        posterior,
        selection,
        config=campaign.fidelity,
        dataset_identity=nearby_dataset_identity(
            base_dataset_identity,
            scenario,
        ),
    )


def run_nearby_model_evaluation(
    root: str | Path,
    posterior,
    selection,
    campaign: ProductionCampaignConfig,
    *,
    base_dataset_identity: str,
    scenario: NearbyBaselineScenario,
    model_hash: str,
) -> dict[str, object]:
    """Pre-compute one model's F3 evidence inside one nearby scenario's tree.

    Execution-only plumbing, the alternative-root counterpart of
    :func:`gwpop_search.validation.stress.run_stress_model_evaluation`:
    :func:`run_nearby_baseline_suite` evaluates every model of the scenario's
    graph serially in one process, and this runs exactly one of those F3
    evaluations with the scenario's own evaluator (full frozen catalog,
    scenario dataset identity), evaluation seed and run directory, so the
    later suite reloads it instead of recomputing it.

    Nothing is written to any state database and no
    ``nearby_baseline_plan.json`` is written (the plan freezes the whole suite
    and the reference graph, which only the suite run pins). F0 stays with the
    suite: its outcome is read from the suite's state database, so it cannot
    be pre-computed here. A model that later fails F0 inside the suite is not
    promoted, and its pre-computed F3 run is then simply never read.
    """
    graph = nearby_scenario_graph(scenario)
    model_hash = str(model_hash)
    if model_hash not in graph.by_hash:
        raise ValueError(
            f"model hash {model_hash!r} is not in the graph of nearby scenario "
            f"{scenario.scenario_id!r}"
        )

    evaluator = _nearby_evaluator(
        posterior,
        selection,
        campaign,
        base_dataset_identity=base_dataset_identity,
        scenario=scenario,
    )
    fidelity = Fidelity.F3_EVIDENCE
    seed = evaluation_seed(
        nearby_baseline_seed(
            campaign.seed_policy.root_seed,
            scenario.scenario_id,
        ),
        model_hash,
        fidelity,
    )
    run_dir = nearby_scenario_model_run_dir(
        root,
        scenario.scenario_id,
        model_hash,
        fidelity,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    record = evaluator.evaluate(
        graph.by_hash[model_hash],
        fidelity,
        seed=seed,
        run_dir=run_dir,
    )
    return {
        "format_version": MODEL_EVALUATION_FORMAT,
        "state_database_written": False,
        "scenario_id": scenario.scenario_id,
        "root_model_hash": graph.root_hash,
        "model_hash": model_hash,
        "fidelity": fidelity.value,
        "seed": int(seed),
        "dataset_identity": evaluator.dataset_identity,
        "fidelity_config_sha256": evaluator.fidelity_config_sha256,
        "diagnostics_pass": record.diagnostics_pass,
        "screen_value": record.screen_value,
        "compute_cost_hours": record.compute_cost,
        "run_dir": str(run_dir),
        "evaluation": str(run_dir / "evaluation.json"),
    }


def run_nearby_baseline_suite(
    root: str | Path,
    posterior,
    selection,
    reference_graph: ModelGraph,
    campaign: ProductionCampaignConfig,
    *,
    base_dataset_identity: str,
    suite: NearbyBaselineSuiteSpec,
    reference_state_database: str | Path | None = None,
) -> dict[str, object]:
    """Run/resume the search from explicitly declared nearby root models."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    plan = build_nearby_baseline_plan(
        campaign=campaign,
        base_dataset_identity=base_dataset_identity,
        reference_graph=reference_graph,
        suite=suite,
    )
    _write_plan_once(root / "nearby_baseline_plan.json", plan)

    reference_support = {}
    if reference_state_database is not None:
        reference_support = mutation_log_bayes_factors(
            reference_graph,
            collect_best_available_evidence(reference_state_database),
        )

    results = []
    for scenario in suite.scenarios:
        graph = nearby_scenario_graph(scenario)
        scenario_root = root / scenario.scenario_id
        artifacts = scenario_root / "artifacts"
        database = scenario_root / "state.sqlite"

        evaluator = _nearby_evaluator(
            posterior,
            selection,
            campaign,
            base_dataset_identity=base_dataset_identity,
            scenario=scenario,
        )
        execution = execute_search(
            graph,
            evaluator,
            state_database=database,
            artifact_root=artifacts,
            config=SearchExecutionConfig(
                root_seed=nearby_baseline_seed(
                    campaign.seed_policy.root_seed,
                    scenario.scenario_id,
                ),
                scheduler=campaign.scheduler,
                stop_fidelity=suite.config.stop_fidelity,
                max_models_by_fidelity={
                    "F3": suite.config.max_f3_models,
                    "F4": suite.config.max_f4_models,
                },
                max_total_compute_cost=(
                    suite.config.max_gpu_hours_per_scenario
                ),
            ),
        )
        evidence = collect_best_available_evidence(database)
        support = mutation_log_bayes_factors(graph, evidence)
        scoring = write_scientific_scoring(
            graph,
            state_database=database,
            artifact_root=artifacts,
            model_prior=model_prior_from_config(campaign.model_prior),
        )
        comparison = (
            None
            if not reference_support
            else compare_mutation_support(reference_support, support)
        )
        row = {
            "scenario": scenario.to_dict(),
            "graph_root_hash": graph.root_hash,
            "n_graph_models": len(graph.nodes),
            "execution": execution.to_dict(),
            "n_models_with_evidence": len(evidence),
            "mutation_max_log_bayes_factors": support,
            "reference_comparison": comparison,
            "scientific_scoring": scoring,
        }
        if scenario.restricted:
            # the signed per-edge ln BF the v2 claim table's D5 reads
            row["root_edge_log_bayes_factors"] = root_edge_log_bayes_factors(
                graph,
                evidence,
            )
            row["inapplicable_paths"] = list(
                nearby_scenario_inapplicable_paths(scenario)
            )
        (scenario_root / "nearby_baseline_summary.json").write_text(
            json.dumps(row, sort_keys=True, indent=2)
        )
        results.append(row)

    shifts = [
        row["reference_comparison"]["max_abs_delta_log_bayes_factor"]
        for row in results
        if (
            row["reference_comparison"] is not None
            and row["reference_comparison"][
                "max_abs_delta_log_bayes_factor"
            ]
            is not None
        )
    ]
    summary = {
        "format_version": SUMMARY_FORMAT,
        "n_scenarios": len(results),
        "reference_evidence_available": bool(reference_support),
        "max_abs_delta_log_bayes_factor_across_scenarios": (
            None if not shifts else float(max(shifts))
        ),
        "scenarios": results,
    }
    (root / "nearby_baseline_suite_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return summary
