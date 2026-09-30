"""Frozen multi-null replay campaigns using the exact deterministic search.

Calibrated statistic (``statistic="f3_completion"``, decision D4)
    The maximum edge ``ln BF`` and the maximum edge log posterior odds over
    the declared graph, both floored at zero, computed from F3 evidence after
    full-graph evidence completion. Nulls therefore run the F0 -> F3 search
    (stop at F3) followed by F3 completion of every node; the observed
    statistic uses the identical F3 procedure on the production state
    (F4 is precision reporting only and never enters the statistic). F3
    seeds are ``evaluation_seed(root, model, "F3")`` on every path, so the
    complete F3 evidence set does not depend on the search beam.

Formats: null configuration ``gwpop-search-exact-null-campaign-1.5``
(frozen-selection PE uses the declared noisy-observation approximation with a
PE precision tied to the frozen catalog, a catalog-scaled resampling-ESS gate
and an explicit per-null compute ceiling), plan
``gwpop-search-exact-null-plan-1.3`` and summary
``gwpop-search-exact-null-summary-1.3`` (both record the statistic mode,
ladder, sampler backend, PE-scale policy and fidelity-config hash).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path

from gwpop_search.grammar import ModelGraph
from gwpop_search.inference.fidelity import (
    DeterministicHBIEvaluator,
    fidelity_config_sha256,
)
from gwpop_search.inference.numpyro import _code_identity
from gwpop_search.inference.synthetic import (
    OBSERVATION_MODEL_NOISY,
    SyntheticSurveyConfig,
)
from gwpop_search.models import DEFAULT_BASELINE_HYPERPARAMETERS
from gwpop_search.production import ProductionCampaignConfig
from gwpop_search.production.freeze import model_graph_hash
from gwpop_search.production.runner import (
    collect_best_available_evidence,
    model_prior_from_config,
)
from gwpop_search.search import (
    Fidelity,
    SearchExecutionConfig,
    evaluation_seed,
)

from .frozen_selection import (
    DEFAULT_MIN_RESAMPLING_ESS_PER_EVENT,
    frozen_selection_resampling_probabilities,
    required_resampling_ess,
)
from .pe_matching import (
    PE_SCALE_POLICY_DECLARED_FIXED,
    PE_SCALE_POLICY_MATCH_OBSERVED,
    measure_observed_pe_scales,
    pe_scale_comparison,
    require_pe_scale_policy,
)

EXACT_NULL_CONFIG_FORMAT_VERSION = "gwpop-search-exact-null-campaign-1.5"
LEGACY_EXACT_NULL_CONFIG_FORMATS = (
    "gwpop-search-exact-null-campaign-1.0",
    "gwpop-search-exact-null-campaign-1.1",
    "gwpop-search-exact-null-campaign-1.2",
    "gwpop-search-exact-null-campaign-1.3",
    "gwpop-search-exact-null-campaign-1.4",
)
EXACT_NULL_PLAN_FORMAT_VERSION = "gwpop-search-exact-null-plan-1.3"
EXACT_NULL_SUMMARY_FORMAT_VERSION = "gwpop-search-exact-null-summary-1.3"
EXACT_NULL_PRECHECK_FORMAT_VERSION = "gwpop-search-exact-null-precheck-1.0"
STATISTIC_F3_COMPLETION = "f3_completion"
NULL_STATISTICS = (STATISTIC_F3_COMPLETION,)
# Rungs a null replay runs and the evidence the statistic is computed from.
STATISTIC_STOP_FIDELITY = {STATISTIC_F3_COMPLETION: Fidelity.F3_EVIDENCE}
STATISTIC_EVIDENCE_FIDELITIES = {STATISTIC_F3_COMPLETION: ("F3",)}


def _default_null_survey() -> SyntheticSurveyConfig:
    return SyntheticSurveyConfig(observation_model=OBSERVATION_MODEL_NOISY)


def statistic_definition(statistic: str) -> dict[str, object]:
    """Machine-readable definition of a calibrated null statistic."""
    if statistic != STATISTIC_F3_COMPLETION:
        raise ValueError(f"unsupported null statistic {statistic!r}")
    return {
        "mode": STATISTIC_F3_COMPLETION,
        "evidence_fidelities": list(STATISTIC_EVIDENCE_FIDELITIES[statistic]),
        "replay_ladder_stop": STATISTIC_STOP_FIDELITY[statistic].value,
        "evidence_completion_required": True,
        "statistics": ["max_log_bayes_factor", "max_log_posterior_odds"],
        "definition": (
            "maximum over graph edges of ln BF (and of ln posterior odds with the frozen "
            "model prior), floored at 0, from F3 evidence after full-graph F3 completion; "
            "identical procedure for the observed data and every null; F4 never enters"
        ),
    }

from .replay import (
    SearchReplayResult,
    calibrate_search_replays,
    null_replay_seed,
)
from .search_replay import (
    build_null_replay_dataset,
    declared_null_root,
    run_baseline_null_search_replay,
    search_statistics_from_evidence,
)

NULL_MODEL_EVALUATION_FORMAT_VERSION = "gwpop-search-null-model-evaluation-1.0"


@dataclass(frozen=True)
class ExactNullCampaignConfig:
    """Frozen exact-null calibration settings (format 1.5).

    ``survey.observation_model`` must be ``noisy_observation`` in every data
    mode (zero-noise truth-centred PE is disqualifying). In
    ``frozen_selection_resample`` mode the survey supplies the event count and
    the PE settings; its injection options must stay unset.

    ``pe_scale_policy`` decides where the null PE precision comes from
    (:mod:`gwpop_search.nulls.pe_matching`). ``match_observed`` (default) takes
    the per-event measurement scales and the PE sample count from the frozen
    observed catalog, so a null's F3 Monte-Carlo regime is the observed run's
    regime, as D4's "identical F3 procedure" requires; ``declared_fixed``
    keeps the configured scalars and records the measured mismatch.

    The resampling gate is ``max(min_resampling_ess, min_resampling_ess_per_event
    * n_events)``: an absolute floor alone lets a catalog draw more events than
    there are effectively distinct truths.

    ``max_gpu_hours_per_null`` is the per-null compute ceiling in the
    evaluator's ``compute_cost`` units (wall-clock hours); it has no default
    because it must be frozen from cost calibration (``None`` = not yet set; a
    plan cannot be prepared without it). ``statistic`` is the calibrated search
    statistic (``f3_completion``).
    """

    n_nulls: int = 100
    root_seed: int = 20260918
    survey: SyntheticSurveyConfig = field(default_factory=_default_null_survey)
    truth_hyperparameters: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_BASELINE_HYPERPARAMETERS)
    )
    data_mode: str = "frozen_selection_resample"
    min_resampling_ess: float = 200.0
    min_resampling_ess_per_event: float = DEFAULT_MIN_RESAMPLING_ESS_PER_EVENT
    pe_scale_policy: str = PE_SCALE_POLICY_MATCH_OBSERVED
    max_gpu_hours_per_null: float | None = None
    statistic: str = STATISTIC_F3_COMPLETION
    format_version: str = EXACT_NULL_CONFIG_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version in LEGACY_EXACT_NULL_CONFIG_FORMATS:
            raise ValueError(
                f"unsupported exact null campaign format {self.format_version!r}: "
                "NUTS/JAXNS-era null configuration (truth-centred PE, full-ladder "
                f"statistic); re-freeze as {EXACT_NULL_CONFIG_FORMAT_VERSION}"
            )
        if self.format_version != EXACT_NULL_CONFIG_FORMAT_VERSION:
            raise ValueError(
                f"unsupported exact null campaign format {self.format_version!r}"
            )
        if self.statistic not in NULL_STATISTICS:
            raise ValueError(
                f"unsupported null statistic {self.statistic!r}; supported: "
                f"{NULL_STATISTICS}"
            )
        if self.n_nulls <= 0:
            raise ValueError("n_nulls must be positive")
        truth = {
            str(name): float(value)
            for name, value in self.truth_hyperparameters.items()
        }
        missing = set(DEFAULT_BASELINE_HYPERPARAMETERS) - set(truth)
        if missing:
            raise ValueError(
                f"null truth hyperparameters missing {sorted(missing)}"
            )
        object.__setattr__(self, "truth_hyperparameters", truth)
        if self.data_mode not in {
            "synthetic_survey",
            "frozen_selection_resample",
        }:
            raise ValueError(f"unsupported null data mode {self.data_mode!r}")
        if self.survey.observation_model != OBSERVATION_MODEL_NOISY:
            raise ValueError(
                "null calibration requires survey observation_model='noisy_observation': "
                "zero-noise truth-centred PE is disqualifying (Essick & Fishbach 2023)"
            )
        if self.data_mode == "frozen_selection_resample" and (
            self.survey.injection_draw != "uniform_detector_box"
        ):
            raise ValueError(
                "frozen_selection_resample reuses the frozen production selection; "
                "survey injection_draw options do not apply"
            )
        if (
            not math.isfinite(self.min_resampling_ess)
            or self.min_resampling_ess <= 0.0
        ):
            raise ValueError("min_resampling_ess must be finite and positive")
        if (
            not math.isfinite(self.min_resampling_ess_per_event)
            or self.min_resampling_ess_per_event <= 0.0
        ):
            raise ValueError(
                "min_resampling_ess_per_event must be finite and positive"
            )
        object.__setattr__(
            self,
            "pe_scale_policy",
            require_pe_scale_policy(str(self.pe_scale_policy)),
        )
        if (
            self.data_mode != "frozen_selection_resample"
            and self.pe_scale_policy != PE_SCALE_POLICY_DECLARED_FIXED
        ):
            raise ValueError(
                "pe_scale_policy='match_observed' matches the frozen observed "
                "catalog and applies to data_mode='frozen_selection_resample' only"
            )
        if self.max_gpu_hours_per_null is not None:
            ceiling = float(self.max_gpu_hours_per_null)
            if not math.isfinite(ceiling) or ceiling <= 0.0:
                raise ValueError(
                    "max_gpu_hours_per_null must be finite and positive when set"
                )
            object.__setattr__(self, "max_gpu_hours_per_null", ceiling)

    def require_compute_ceiling(self) -> float:
        """The per-null ceiling; refuses a configuration where it is not yet set."""
        if self.max_gpu_hours_per_null is None:
            raise ValueError(
                "max_gpu_hours_per_null is not set: the per-null compute ceiling must be "
                "frozen explicitly (after cost calibration) before a null plan is prepared"
            )
        return float(self.max_gpu_hours_per_null)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "n_nulls": int(self.n_nulls),
            "root_seed": int(self.root_seed),
            "survey": self.survey.to_dict(),
            "truth_hyperparameters": dict(self.truth_hyperparameters),
            "data_mode": self.data_mode,
            "min_resampling_ess": float(self.min_resampling_ess),
            "min_resampling_ess_per_event": float(
                self.min_resampling_ess_per_event
            ),
            "pe_scale_policy": self.pe_scale_policy,
            "max_gpu_hours_per_null": (
                None
                if self.max_gpu_hours_per_null is None
                else float(self.max_gpu_hours_per_null)
            ),
            "statistic": self.statistic,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "ExactNullCampaignConfig":
        version = payload.get("format_version")
        if version != EXACT_NULL_CONFIG_FORMAT_VERSION:
            raise ValueError(
                f"unsupported exact null campaign format {version!r}: re-freeze as "
                f"{EXACT_NULL_CONFIG_FORMAT_VERSION} (noisy-observation PE matched to "
                "the frozen catalog, catalog-scaled resampling-ESS gate, "
                "f3_completion statistic, explicit per-null compute ceiling)"
            )
        ceiling = payload.get("max_gpu_hours_per_null")
        return cls(
            n_nulls=int(payload["n_nulls"]),
            root_seed=int(payload["root_seed"]),
            survey=SyntheticSurveyConfig.from_dict(payload["survey"]),
            truth_hyperparameters={
                str(name): float(value)
                for name, value in dict(
                    payload["truth_hyperparameters"]
                ).items()
            },
            data_mode=str(
                payload.get("data_mode", "frozen_selection_resample")
            ),
            min_resampling_ess=float(payload["min_resampling_ess"]),
            min_resampling_ess_per_event=float(
                payload["min_resampling_ess_per_event"]
            ),
            pe_scale_policy=str(payload["pe_scale_policy"]),
            max_gpu_hours_per_null=None if ceiling is None else float(ceiling),
            statistic=str(payload["statistic"]),
            format_version=str(version),
        )


def save_exact_null_campaign_config(
    path: str | Path,
    config: ExactNullCampaignConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), sort_keys=True, indent=2))


def load_exact_null_campaign_config(
    path: str | Path,
) -> ExactNullCampaignConfig:
    return ExactNullCampaignConfig.from_dict(json.loads(Path(path).read_text()))


def null_search_seed(root_seed: int, null_index: int) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:null-search:{int(null_index)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def build_exact_null_campaign_plan(
    graph: ModelGraph,
    campaign: ProductionCampaignConfig,
    config: ExactNullCampaignConfig,
) -> dict[str, object]:
    # Any registered root hyperprior profile is admissible; the profile is
    # resolved from the graph so the null is drawn under the priors the search
    # actually scores (production: ``gwtc5-v1``).
    _, root_profile = declared_null_root(graph)
    if config.n_nulls > campaign.budget.max_null_replays:
        raise ValueError(
            f"requested {config.n_nulls} nulls exceeds frozen campaign budget "
            f"{campaign.budget.max_null_replays}"
        )
    if len(graph.nodes) > campaign.budget.max_f3_models:
        raise ValueError(
            "exact null calibration requires the production campaign to permit "
            "full-graph F3 evidence completion"
        )
    ceiling = config.require_compute_ceiling()
    stop = STATISTIC_STOP_FIDELITY[config.statistic]
    ladder = list(campaign.scheduler.ladder)
    if stop.value not in ladder:
        raise ValueError(
            f"the {config.statistic} statistic needs {stop.value} in the production "
            f"ladder {ladder}"
        )
    plan = {
        "format_version": EXACT_NULL_PLAN_FORMAT_VERSION,
        "code": _code_identity(),
        "production_campaign_hash": campaign.campaign_hash,
        "graph_hash": model_graph_hash(graph),
        "graph_root_hash": graph.root_hash,
        "root_hyperprior_profile": root_profile,
        "null_config": config.to_dict(),
        "statistic": statistic_definition(config.statistic),
        "sampler_backend": dict(campaign.sampler_backend),
        "fidelity_config_sha256": fidelity_config_sha256(campaign.fidelity),
        "production_dataset_manifest_hash": (
            campaign.dataset_manifest_hash
            if config.data_mode == "frozen_selection_resample"
            else None
        ),
        "replayed_production_search": {
            "production_ladder": ladder,
            "null_ladder": ladder[: ladder.index(stop.value) + 1],
            "stop_fidelity": stop.value,
            "scheduler": asdict(campaign.scheduler),
            "max_gpu_hours": ceiling,
            "source_production_max_gpu_hours": campaign.budget.max_gpu_hours,
            "max_f3_models": campaign.budget.max_f3_models,
            "evidence_completion_required": True,
        },
        "seed_policy": [
            {
                "null_index": index,
                "data_seed": null_replay_seed(config.root_seed, index),
                "search_seed": null_search_seed(config.root_seed, index),
            }
            for index in range(config.n_nulls)
        ],
    }
    # JSON-normalized (tuples -> lists) so a written plan compares equal on reload.
    return json.loads(json.dumps(plan))


def _write_plan_once(path: Path, plan: dict[str, object]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError(
                "existing exact null campaign plan does not match request"
            )
    else:
        path.write_text(json.dumps(plan, sort_keys=True, indent=2))


def _load_replay_results(
    root: Path,
    n_nulls: int,
) -> tuple[SearchReplayResult, ...]:
    results = []
    for index in range(n_nulls):
        payload = json.loads(
            (root / f"null_{index:05d}.json").read_text()
        )
        results.append(SearchReplayResult(**payload))
    return tuple(results)


def _validate_exact_null_inputs(
    campaign: ProductionCampaignConfig,
    config: ExactNullCampaignConfig,
    *,
    production_posterior=None,
    production_selection=None,
    production_dataset_identity: str | None = None,
) -> None:
    if config.data_mode == "frozen_selection_resample":
        if production_posterior is None or production_selection is None:
            raise ValueError(
                "frozen_selection_resample requires the frozen production "
                "posterior and selection catalogs"
            )
        if production_dataset_identity != campaign.dataset_manifest_hash:
            raise ValueError(
                "production dataset identity does not match the frozen campaign"
            )
        if int(config.survey.n_events) != int(production_posterior.n_events):
            raise ValueError(
                "null config n_events must equal the frozen observed event count"
            )
        if config.pe_scale_policy == PE_SCALE_POLICY_MATCH_OBSERVED:
            observed = measure_observed_pe_scales(
                production_posterior
            ).uniform_samples_per_event()
            if int(config.survey.posterior_samples_per_event) != observed:
                raise ValueError(
                    "pe_scale_policy='match_observed' requires the null PE sample "
                    "count to equal the frozen catalog's so that the nulls and the "
                    "observed run share the F3 Monte-Carlo regime (D4); freeze "
                    f"survey.posterior_samples_per_event={observed}, got "
                    f"{int(config.survey.posterior_samples_per_event)}"
                )


def exact_null_data_precheck(
    graph: ModelGraph,
    config: ExactNullCampaignConfig,
    *,
    production_posterior,
    production_selection,
) -> dict[str, object]:
    """Measure, before any replay is launched, what the null catalogs will be.

    Reports the frozen selection's resampling ESS at the null truth against the
    gate the replays apply, the effective truth pool per catalog and the null
    PE precision against the observed catalog. Cheap (one population-density
    pass over the selection rows) and always run by
    :func:`prepare_exact_null_campaign`.
    """
    if config.data_mode != "frozen_selection_resample":
        raise ValueError(
            "the null data precheck applies to data_mode='frozen_selection_resample'"
        )
    root_spec, root_profile = declared_null_root(graph)
    _, diagnostics = frozen_selection_resampling_probabilities(
        production_selection,
        root_spec,
        config.truth_hyperparameters,
    )
    n_events = int(production_posterior.n_events)
    required = required_resampling_ess(
        n_events,
        min_resampling_ess=config.min_resampling_ess,
        min_resampling_ess_per_event=config.min_resampling_ess_per_event,
    )
    scales = measure_observed_pe_scales(production_posterior)
    ess = float(diagnostics["resampling_ess"])
    return {
        "format_version": EXACT_NULL_PRECHECK_FORMAT_VERSION,
        "graph_root_hash": graph.root_hash,
        "root_hyperprior_profile": root_profile,
        "n_events": n_events,
        "truth_hyperparameters": dict(config.truth_hyperparameters),
        "resampling": {
            **{str(k): v for k, v in diagnostics.items()},
            "required_resampling_ess": required,
            "resampling_ess_per_event": ess / n_events,
            "passes_gate": bool(ess >= required),
        },
        "pe_scale_policy": config.pe_scale_policy,
        "pe_scales": pe_scale_comparison(config.survey, scales),
    }


def prepare_exact_null_campaign(
    root: str | Path,
    graph: ModelGraph,
    campaign: ProductionCampaignConfig,
    config: ExactNullCampaignConfig,
    *,
    production_posterior=None,
    production_selection=None,
    production_dataset_identity: str | None = None,
) -> dict[str, object]:
    """Validate and freeze the immutable exact-null campaign plan."""
    _validate_exact_null_inputs(
        campaign,
        config,
        production_posterior=production_posterior,
        production_selection=production_selection,
        production_dataset_identity=production_dataset_identity,
    )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan = build_exact_null_campaign_plan(graph, campaign, config)
    _write_plan_once(root / "null_campaign_plan.json", plan)
    if config.data_mode == "frozen_selection_resample":
        precheck = exact_null_data_precheck(
            graph,
            config,
            production_posterior=production_posterior,
            production_selection=production_selection,
        )
        (root / "null_data_precheck.json").write_text(
            json.dumps(precheck, sort_keys=True, indent=2)
        )
        if not precheck["resampling"]["passes_gate"]:
            raise ValueError(
                "the frozen selection cannot support the requested null catalogs: "
                f"resampling ESS {precheck['resampling']['resampling_ess']:.6g} < "
                f"{precheck['resampling']['required_resampling_ess']:.6g} required for "
                f"{precheck['n_events']} events; every replay would fail the same gate"
            )
    return plan


def _require_exact_null_plan(
    root: Path,
    graph: ModelGraph,
    campaign: ProductionCampaignConfig,
    config: ExactNullCampaignConfig,
) -> dict[str, object]:
    path = root / "null_campaign_plan.json"
    if not path.is_file():
        raise ValueError(
            "exact-null campaign plan is missing; run prepare-null-search-calibration "
            "before indexed replay jobs"
        )
    expected = build_exact_null_campaign_plan(graph, campaign, config)
    actual = json.loads(path.read_text())
    if actual != expected:
        raise ValueError("exact-null campaign plan does not match current inputs")
    return actual


def run_exact_null_index(
    root: str | Path,
    graph: ModelGraph,
    campaign: ProductionCampaignConfig,
    config: ExactNullCampaignConfig,
    *,
    null_index: int,
    production_posterior=None,
    production_selection=None,
    production_dataset_identity: str | None = None,
) -> SearchReplayResult:
    """Run/resume exactly one null replay; safe for unique Slurm array indices."""
    _validate_exact_null_inputs(
        campaign,
        config,
        production_posterior=production_posterior,
        production_selection=production_selection,
        production_dataset_identity=production_dataset_identity,
    )
    index = int(null_index)
    if index < 0 or index >= config.n_nulls:
        raise ValueError(
            f"null_index={index} is outside [0, {config.n_nulls})"
        )

    root = Path(root)
    _require_exact_null_plan(root, graph, campaign, config)
    result_path = root / f"null_{index:05d}.json"
    data_seed = null_replay_seed(config.root_seed, index)
    if result_path.exists():
        result = SearchReplayResult(**json.loads(result_path.read_text()))
        if result.null_index != index or result.seed != data_seed:
            raise ValueError(
                f"null replay checkpoint mismatch at index {index}"
            )
        return result

    model_prior = model_prior_from_config(campaign.model_prior)
    ceiling = config.require_compute_ceiling()
    null_campaign = replace(
        campaign,
        budget=replace(
            campaign.budget,
            max_gpu_hours=ceiling,
        ),
    )
    result = run_baseline_null_search_replay(
        index,
        data_seed,
        root=root,
        graph=graph,
        model_prior=model_prior,
        execution_config=SearchExecutionConfig(
            root_seed=null_search_seed(config.root_seed, index),
            scheduler=campaign.scheduler,
            stop_fidelity=STATISTIC_STOP_FIDELITY[config.statistic],
            max_models_by_fidelity={"F3": campaign.budget.max_f3_models},
            max_total_compute_cost=ceiling,
        ),
        fidelity_config=campaign.fidelity,
        survey_config=config.survey,
        truth_hyperparameters=config.truth_hyperparameters,
        completion_campaign=null_campaign,
        completion_seed_root=null_search_seed(config.root_seed, index),
        data_mode=config.data_mode,
        observed_posterior=production_posterior,
        frozen_selection=production_selection,
        production_dataset_identity=production_dataset_identity,
        min_resampling_ess=config.min_resampling_ess,
        min_resampling_ess_per_event=config.min_resampling_ess_per_event,
        pe_scale_policy=config.pe_scale_policy,
        statistic=config.statistic,
    )
    if result.null_index != index or result.seed != data_seed:
        raise ValueError(
            "null replay callback returned inconsistent index/seed"
        )
    result_path.write_text(
        json.dumps(result.to_dict(), sort_keys=True, indent=2)
    )
    return result


def null_replay_model_run_dir(
    root: str | Path,
    null_index: int,
    model_hash: str,
    fidelity: Fidelity,
) -> Path:
    """The run directory the replay at ``null_index`` uses for one evaluation.

    Identical to what :func:`gwpop_search.search.execute_search` and
    :func:`gwpop_search.production.completion.complete_graph_evidence` derive
    inside :func:`run_baseline_null_search_replay`
    (``<root>/searches/null_<index>/artifacts/<fidelity>/<model hash>``).
    """
    return (
        Path(root)
        / "searches"
        / f"null_{int(null_index):05d}"
        / "artifacts"
        / Fidelity(fidelity).value
        / str(model_hash)
    )


def run_null_model_evaluation(
    root: str | Path,
    graph: ModelGraph,
    campaign: ProductionCampaignConfig,
    config: ExactNullCampaignConfig,
    *,
    null_index: int,
    model_hash: str,
    production_posterior=None,
    production_selection=None,
    production_dataset_identity: str | None = None,
) -> dict[str, object]:
    """Pre-compute one model's evidence rung inside one null replay's tree.

    Execution-only plumbing. :func:`run_exact_null_index` evaluates every graph
    model serially in a single process; this runs exactly one of those
    evaluations, with the replay's own null catalog, dataset identity,
    evaluation seed and run directory, so the later replay reloads it from its
    artifact tree instead of recomputing it. The numerics, seeds and gates are
    the replay's own: nothing here chooses them.

    Nothing is written to any state database (the replay records its own rows,
    exactly as it does today), so this is safe to run concurrently for distinct
    models. The compute it spends is *not* charged against
    ``max_gpu_hours_per_null``, which only counts what the replay process
    itself evaluates.
    """
    _validate_exact_null_inputs(
        campaign,
        config,
        production_posterior=production_posterior,
        production_selection=production_selection,
        production_dataset_identity=production_dataset_identity,
    )
    index = int(null_index)
    if index < 0 or index >= config.n_nulls:
        raise ValueError(
            f"null_index={index} is outside [0, {config.n_nulls})"
        )
    model_hash = str(model_hash)
    if model_hash not in graph.by_hash:
        raise ValueError(f"unknown graph model hash {model_hash!r}")

    root = Path(root)
    _require_exact_null_plan(root, graph, campaign, config)

    data_seed = null_replay_seed(config.root_seed, index)
    dataset = build_null_replay_dataset(
        data_seed,
        graph=graph,
        survey_config=config.survey,
        truth_hyperparameters=config.truth_hyperparameters,
        data_mode=config.data_mode,
        observed_posterior=production_posterior,
        frozen_selection=production_selection,
        production_dataset_identity=production_dataset_identity,
        min_resampling_ess=config.min_resampling_ess,
        min_resampling_ess_per_event=config.min_resampling_ess_per_event,
        pe_scale_policy=config.pe_scale_policy,
    )
    evaluator = DeterministicHBIEvaluator(
        dataset.posterior,
        dataset.selection,
        config=campaign.fidelity,
        dataset_identity=dataset.dataset_identity,
    )
    # The rung the statistic reads, which is where the replay spends its
    # compute; the F0 rung stays with the replay (it is cheap and it is what
    # screens the model before the executor promotes it).
    fidelity = STATISTIC_STOP_FIDELITY[config.statistic]
    seed = evaluation_seed(
        null_search_seed(config.root_seed, index),
        model_hash,
        fidelity,
    )
    run_dir = null_replay_model_run_dir(root, index, model_hash, fidelity)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = evaluator.evaluate(
        graph.by_hash[model_hash],
        fidelity,
        seed=seed,
        run_dir=run_dir,
    )
    return {
        "format_version": NULL_MODEL_EVALUATION_FORMAT_VERSION,
        "state_database_written": False,
        "null_index": index,
        "model_hash": model_hash,
        "fidelity": fidelity.value,
        "data_seed": int(data_seed),
        "seed": int(seed),
        "dataset_identity": dataset.dataset_identity,
        "fidelity_config_sha256": evaluator.fidelity_config_sha256,
        "diagnostics_pass": record.diagnostics_pass,
        "screen_value": record.screen_value,
        "compute_cost_hours": record.compute_cost,
        "run_dir": str(run_dir),
        "evaluation": str(run_dir / "evaluation.json"),
    }


def _null_data_diagnostics(
    results: tuple[SearchReplayResult, ...],
) -> dict[str, object]:
    """Per-null truth-pool and PE-precision summary over a finished campaign.

    Each null catalog draws its truths with replacement from the frozen
    selection, so ``unique_truth_fraction`` (distinct truth rows per event) and
    the resampling ESS say how independent the replicates really are.
    """
    def _collect(key: str) -> list[float]:
        values = []
        for item in results:
            payload = (item.metadata or {}).get("null_data_metadata") or {}
            if key in payload:
                values.append(float(payload[key]))
        return values

    out: dict[str, object] = {"n_nulls": len(results)}
    for key in ("resampling_ess", "unique_truth_fraction", "n_unique_truth_rows"):
        values = _collect(key)
        if not values:
            continue
        out[key] = {
            "min": min(values),
            "median": float(sorted(values)[len(values) // 2]),
            "max": max(values),
            "n_reported": len(values),
        }
    policies = sorted(
        {
            str(
                ((item.metadata or {}).get("null_data_metadata") or {})
                .get("pe_approximation", {})
                .get("pe_scale_policy", "unrecorded")
            )
            for item in results
        }
    )
    out["pe_scale_policies"] = policies
    return out


def finalize_exact_null_campaign(
    root: str | Path,
    graph: ModelGraph,
    campaign: ProductionCampaignConfig,
    config: ExactNullCampaignConfig,
    *,
    observed_state_database: str | Path | None = None,
    production_dataset_identity: str | None = None,
) -> dict[str, object]:
    """Aggregate a complete indexed campaign and calibrate the observed search."""
    root = Path(root)
    if (
        config.data_mode == "frozen_selection_resample"
        and production_dataset_identity != campaign.dataset_manifest_hash
    ):
        raise ValueError(
            "production dataset identity does not match the frozen campaign"
        )
    _require_exact_null_plan(root, graph, campaign, config)

    missing = [
        index
        for index in range(config.n_nulls)
        if not (root / f"null_{index:05d}.json").is_file()
    ]
    if missing:
        preview = missing[:20]
        suffix = "" if len(missing) <= 20 else f" ... (+{len(missing) - 20} more)"
        raise ValueError(
            f"cannot finalize exact-null campaign; missing indices {preview}{suffix}"
        )

    results = _load_replay_results(root, config.n_nulls)
    for index, result in enumerate(results):
        expected_seed = null_replay_seed(config.root_seed, index)
        if result.null_index != index or result.seed != expected_seed:
            raise ValueError(
                f"null replay checkpoint mismatch at index {index}"
            )

    model_prior = model_prior_from_config(campaign.model_prior)
    observed = None
    if observed_state_database is not None:
        evidence = collect_best_available_evidence(
            observed_state_database,
            fidelities=STATISTIC_EVIDENCE_FIDELITIES[config.statistic],
            fidelity_config_sha256=fidelity_config_sha256(campaign.fidelity),
        )
        # An empty evidence set is not "no observed run requested": the caller
        # passed a state database precisely to obtain a calibrated p-value, so a
        # summary without one would look complete while carrying no result.
        if len(evidence) != len(graph.nodes):
            raise ValueError(
                "observed production state lacks complete valid "
                f"{'/'.join(STATISTIC_EVIDENCE_FIDELITIES[config.statistic])} evidence "
                f"({len(evidence)} of {len(graph.nodes)} graph models); run "
                "complete-production-evidence before null calibration"
            )
        observed = search_statistics_from_evidence(
            graph,
            evidence,
            model_prior=model_prior,
        )

    calibrated = calibrate_search_replays(
        results,
        observed_max_log_bayes_factor=(
            None if observed is None else observed["max_log_bayes_factor"]
        ),
        observed_max_log_posterior_odds=(
            None
            if observed is None
            else observed["max_log_posterior_odds"]
        ),
    )
    summary = {
        "format_version": EXACT_NULL_SUMMARY_FORMAT_VERSION,
        "production_campaign_hash": campaign.campaign_hash,
        "statistic": statistic_definition(config.statistic),
        "sampler_backend": dict(campaign.sampler_backend),
        "graph_root_hash": graph.root_hash,
        "root_hyperprior_profile": declared_null_root(graph)[1],
        "null_data_mode": config.data_mode,
        "pe_scale_policy": config.pe_scale_policy,
        "max_gpu_hours_per_null": config.require_compute_ceiling(),
        "production_dataset_identity": production_dataset_identity,
        "null_data_diagnostics": _null_data_diagnostics(results),
        "observed_search_statistics": observed,
        "calibration": calibrated,
    }
    (root / "exact_null_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return summary


def run_exact_null_campaign(
    root: str | Path,
    graph: ModelGraph,
    campaign: ProductionCampaignConfig,
    config: ExactNullCampaignConfig,
    *,
    observed_state_database: str | Path | None = None,
    production_posterior=None,
    production_selection=None,
    production_dataset_identity: str | None = None,
) -> dict[str, object]:
    """Serial convenience wrapper around prepare/index/finalize operations."""
    prepare_exact_null_campaign(
        root,
        graph,
        campaign,
        config,
        production_posterior=production_posterior,
        production_selection=production_selection,
        production_dataset_identity=production_dataset_identity,
    )
    for index in range(config.n_nulls):
        run_exact_null_index(
            root,
            graph,
            campaign,
            config,
            null_index=index,
            production_posterior=production_posterior,
            production_selection=production_selection,
            production_dataset_identity=production_dataset_identity,
        )
    return finalize_exact_null_campaign(
        root,
        graph,
        campaign,
        config,
        observed_state_database=observed_state_database,
        production_dataset_identity=production_dataset_identity,
    )
