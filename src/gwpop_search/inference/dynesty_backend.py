"""dynesty nested sampling for the standardized HBI shape likelihood.

This backend produces hyperposteriors *and* evidences with dynesty's static
``NestedSampler`` while evaluating the common HBI likelihood in fixed-size
GPU batches.

Likelihood contract
-------------------
The target is exactly the standardized, rate-marginalized shape likelihood of
:mod:`gwpop_search.hbi.jax_backend`::

    log L(Lambda) = sum_i log ell_i(Lambda) - N log A(Lambda)

Nothing about the likelihood, the data semantics, the priors or the model
grammar is changed here. :class:`BatchedShapeLogLikelihood` evaluates the
same event terms and selection exposure as
``gwpop_search.hbi.build_jax_shape_log_likelihood`` (parity is pinned by the
tests), vectorized over a fixed ``[batch_size, ndim]`` block of
hyperparameter vectors with ``jax.jit(jax.vmap(...))``.

* With ``HBIConfig.variance_taper`` set, the target is the *tapered* shape
  likelihood ``log L + ln T(sigma^2_lnL)`` of :mod:`gwpop_search.hbi.taper`
  (GWTC-5 / Callister & Farr 2024), with ``sigma^2_lnL`` computed in the same
  device pass; the evidence is then ``ln int L T pi``. The taper is part of
  the HBI configuration and therefore of the likelihood identity.
  :func:`posterior_taper_mass` reports the posterior fraction inside the
  taper region.
* ``-inf`` is genuine zero population support and is handed to dynesty
  unchanged; dynesty's own initial-volume/plateau machinery accounts for the
  zero-likelihood part of the prior. (dynesty internally stores such points
  with ``logl = -1e300``; :class:`DynestyResult` reports them as ``-inf``.)
* NaN or ``+inf`` anywhere in an event term or in the selection exposure is a
  bug and raises :class:`gwpop_search.hbi.PopulationDensityError`. It is never
  mapped to a likelihood value.
* As in the JAX builder, an exposure estimate with no finite population
  support (``log A = -inf``) while every event keeps support gives
  ``log L = -inf`` (the NumPy reference raises ``SelectionSupportError``
  there). Such evaluations are counted, summed across checkpoint/resume
  sessions, reported as ``DynestyResult.n_selection_unsupported`` and
  announced with :class:`SelectionSupportWarning`: they remove prior volume
  from the evidence because the injections do not cover the population.

Evidence contract
-----------------
The prior transform maps the unit cube onto the declared normalized
hyperpriors, so ``log_evidence = ln int L(Lambda) pi(Lambda) dLambda`` for the
shape likelihood. The shape likelihood is the Poisson likelihood marginalized
over the rate with ``p(R) ~ 1/R``; the omitted constant (``log Gamma(N)`` plus
the normalization of that improper rate prior) is the same for every
population model evaluated on the same PE catalog and selection product, so it
cancels in Bayes factors between such models. Evidences are comparable only
under those conditions (same data, same rate treatment). Further caveats:

* The hyperpriors are the model's declared priors. For a declarative model
  (``population_model.spec``), :func:`run_dynesty_population` refuses priors
  that differ from ``prior_specs_from_model_spec(model.spec)`` because those
  priors are part of the model hash that the evidence is recorded under.
* The prior is not renormalized over hyperparameters that a density encodes
  as invalid (``-inf``). If a prior admitted such a region, ``ln Z`` would
  include ``ln P(valid)``. Genuine zero support, where the data exclude the
  hyperparameters (e.g. an event outside the mass range), is correctly part
  of ``ln Z``.
* Monte Carlo noise in ``log L`` enters ``ln Z`` at order ``Var[log L]``;
  check it on the posterior with :func:`importance_diagnostics_over_posterior`.

Identity and provenance
-----------------------
A resume (and, for population runs, the reuse of a finished result) must be
the same computation: the checkpoint and the run manifest pin the parameter
names, seed, the trajectory-relevant configuration
(:meth:`DynestyConfig.identity_dict`), the code (package version, git commit,
``git_dirty`` and a SHA-256 of the package sources, so uncommitted edits are
detected), the software versions and the runtime (JAX backend, device kind,
platform version, ``XLA_FLAGS``, CPU model). The likelihood identity (names,
HBI config, model configuration, data digests) is stored with every result so
that diagnostics can refuse to evaluate a different estimator.

Batching and determinism
------------------------
dynesty is run with ``queue_size = batch_size`` and a :class:`ThreadBatchPool`.
dynesty then proposes ``batch_size`` independent live-point replacements per
queue refill (each with its own seed spawned from the sampler's generator) and
maps its internal sampler over them. The pool runs every proposal in its own
worker thread and routes all their likelihood calls through one dispatcher
that evaluates the pending requests together. The composition of each device
batch therefore depends on thread timing, but no result does: every row is a
pure function of its hyperparameters (verified by the tests on CPU and by the
GPU smoke check), the per-proposal random streams are fixed, and results are
returned in item order. A run is therefore reproducible bit for bit for a
fixed seed, configuration, device and software stack.

Checkpoint/resume
-----------------
dynesty checkpoints (pickles) the whole sampler, including its random
generator, live points, bounds and queued proposals. Resuming from a
checkpoint written during the main loop (the normal case after a job is
killed) continues the identical trajectory, so the final evidence and samples
equal those of an uninterrupted run with the same seed. ``maxiter``/``maxcall``
are total budgets across sessions (dynesty counts them per call; the remaining
budget is passed on resume). ``checkpoint_every`` and
``num_posterior_samples`` do not affect the trajectory and may change between
sessions. Run counters (likelihood evaluations, zero-likelihood and
selection-unsupported evaluations, sampling wall time) travel inside the
checkpoint and are totals over the sessions that produced the result.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields as dataclass_fields, replace
import hashlib
import importlib
import json
import math
import numbers
import os
from pathlib import Path
import pickle
import platform
import subprocess
import threading
import time
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np
from scipy.special import logsumexp, ndtri

from gwpop_search.hbi.common import PopulationDensityError

from .priors import PriorSpec, serialize_prior_map

MANIFEST_FORMAT_VERSION = "gwpop-search-dynesty-1.1"
RESULT_FORMAT_VERSION = "gwpop-search-dynesty-result-1.1"
CHECKPOINT_FORMAT_VERSION = "gwpop-search-dynesty-checkpoint-1.1"
# Results written before provenance/counters were recorded load with those
# fields empty (None / {}).
_READABLE_RESULT_FORMATS = ("gwpop-search-dynesty-result-1.0", RESULT_FORMAT_VERSION)

# Attributes stamped on every sampler created by run_dynesty. dynesty pickles
# the sampler's __dict__, so checkpoints carry them: the metadata lets a resume
# verify that the requested run is the one the checkpoint was made for, and the
# tally carries the run counters across sessions.
_CHECKPOINT_META_ATTR = "_gwpop_search_checkpoint_meta"
_TALLY_ATTR = "_gwpop_search_run_tally"

# DynestyConfig fields that do not influence the sampling trajectory: the
# checkpoint cadence (I/O only) and the equal-weight resampling size
# (post-processing of the finished run).
_NON_TRAJECTORY_FIELDS = ("checkpoint_every", "num_posterior_samples")

# Entropy tag for the equal-weight resampling stream. The dynesty run itself
# uses ``np.random.default_rng(seed)``; resampling uses
# ``np.random.default_rng([seed, _RESAMPLE_STREAM])``, an independent stream.
_RESAMPLE_STREAM = 0x5245534D  # "RESM"

# Where the pool is used. Prior transforms and bound updates are cheap and are
# kept on the main thread; every likelihood evaluation goes through the pool.
_USE_POOL = {
    "prior_transform": False,
    "loglikelihood": True,
    "propose_point": True,
    "update_bound": False,
}

_BOUNDS = ("none", "single", "multi", "balls", "cubes")
_SAMPLERS = ("unif", "rwalk", "slice", "rslice")
_SLICE_SAMPLERS = ("slice", "rslice")
_PRIOR_FAMILIES = ("uniform", "log_uniform", "normal")

DEFAULT_DIAGNOSTIC_QUANTILES = (0.01, 0.1, 0.5, 0.9, 0.99)


class DynestyUnavailableError(ImportError):
    """Raised when the dynesty backend is requested without dynesty installed."""


class PoolCancelledError(RuntimeError):
    """A pooled task was abandoned because another task in the same map failed."""


class DirtyCodeWarning(UserWarning):
    """The package sources have uncommitted changes, so ``git_commit`` does not
    identify the code; the run is identified by ``code.source_sha256``.
    Production drivers should escalate this warning to an error."""


class SelectionSupportWarning(UserWarning):
    """Some evaluations had population support on every event but none on the
    selection injections; they were treated as zero likelihood, which removes
    that prior volume from the evidence."""


def _require_dynesty():
    try:
        import dynesty
        import dynesty.utils  # noqa: F401  (checkpoint/restore helpers)
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise DynestyUnavailableError(
            "dynesty nested sampling requires the inference extra: "
            "pip install 'gwpop-search[inference]'"
        ) from exc
    try:
        major, minor = (int(part) for part in dynesty.__version__.split(".")[:2])
    except ValueError:  # pragma: no cover - unexpected version string
        major, minor = (0, 0)
    if (major, minor) < (3, 1):
        raise DynestyUnavailableError(
            f"dynesty>=3.1 is required; found {dynesty.__version__}"
        )
    return dynesty


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "the batched HBI likelihood requires JAX: "
            "pip install 'gwpop-search[inference]'"
        ) from exc
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError(
            "the batched HBI likelihood requires 64-bit JAX; set JAX_ENABLE_X64=true "
            "or call jax.config.update('jax_enable_x64', True) before building it"
        )
    return jax, jnp


def _software_versions() -> dict[str, str]:
    """Versions of the numerical stack (importing them never initializes a device)."""
    versions = {}
    for module in ("numpy", "scipy", "dynesty", "jax", "jaxlib"):
        try:
            versions[module] = str(importlib.import_module(module).__version__)
        except ImportError:  # pragma: no cover - optional extras
            versions[module] = "unavailable"
    return dict(sorted(versions.items()))


# ---------------------------------------------------------------------------
# Code, runtime and likelihood identity
# ---------------------------------------------------------------------------


def _json_normalized(value):
    """``value`` as it reads back from JSON (tuples -> lists, sorted keys)."""
    return json.loads(json.dumps(_json_ready(value), sort_keys=True))


def _json_sha256(value) -> str:
    text = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_dir() -> Path:
    import gwpop_search

    return Path(gwpop_search.__file__).resolve().parent


def _source_digest(package_dir: Path) -> tuple[str, int]:
    """SHA-256 over every ``*.py`` file below ``package_dir`` (relative path + content)."""
    digest = hashlib.sha256()
    count = 0
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir)
        if "__pycache__" in relative.parts or not path.is_file():
            continue
        digest.update(relative.as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        count += 1
    return digest.hexdigest(), count


def _git_root(start: Path) -> Path | None:
    for parent in (start, *start.parents):
        if (parent / ".git").exists():  # a directory, or a file in a git worktree
            return parent
    return None


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _code_identity(package_dir: str | Path | None = None) -> dict[str, object]:
    """Identity of the code that runs.

    ``source_sha256`` hashes every ``*.py`` file of the package (relative path
    and content), so uncommitted edits change the identity even when the git
    commit does not, and it is available for installs without git metadata.
    ``git_dirty`` is ``True`` when ``git status`` reports modified, staged or
    untracked ``*.py`` files under the package directory and ``None`` when no
    git checkout (or no git) is found. ``GWPOP_GIT_COMMIT`` overrides the
    commit, as for the NUTS manifests.
    """
    from gwpop_search import __version__

    package_dir = _package_dir() if package_dir is None else Path(package_dir).resolve()
    source_sha256, n_files = _source_digest(package_dir)
    commit = os.environ.get("GWPOP_GIT_COMMIT") or None
    dirty = None
    repo = _git_root(package_dir)
    if repo is not None:
        if commit is None:
            head = _git(repo, "rev-parse", "HEAD")
            commit = None if head is None else (head.strip() or None)
        relative = package_dir.relative_to(repo).as_posix()
        pathspec = f":(glob){relative}/**/*.py" if relative != "." else ":(glob)**/*.py"
        status = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", pathspec)
        dirty = None if status is None else bool(status.strip())
    return {
        "package_version": str(__version__),
        "git_commit": commit or "unknown",
        "git_dirty": dirty,
        "source_sha256": source_sha256,
        "n_source_files": n_files,
    }


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:  # pragma: no cover - non-Linux hosts
        pass
    return platform.processor() or "unknown"  # pragma: no cover


def _runtime_identity(*, jax_devices: bool) -> dict[str, object]:
    """Where the computation runs (bit-identical resume needs the same runtime).

    The host part (Python, architecture, CPU model) always applies: dynesty's
    bounds and proposals run in NumPy on the host. With ``jax_devices`` the
    JAX backend is queried (and therefore initialized): backend, device
    platform and kind of the default device, platform (CUDA) version,
    ``XLA_FLAGS`` and the 64-bit flag.
    """
    info: dict[str, object] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
    }
    if jax_devices:
        jax, _ = _require_jax()
        device = jax.devices()[0]
        client = getattr(device, "client", None)
        info.update(
            {
                "jax_backend": str(jax.default_backend()),
                "device_platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "platform_version": str(getattr(client, "platform_version", "unknown")),
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
                "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            }
        )
    return info


def _model_identity(population_model) -> dict[str, object]:
    from .numpyro import _model_config

    payload = dict(_model_config(population_model))
    if not hasattr(population_model, "to_config"):
        # Plain callables have no configuration; record which callable it is.
        module = getattr(population_model, "__module__", None)
        qualname = getattr(population_model, "__qualname__", None)
        if module and qualname:
            payload["callable"] = f"{module}.{qualname}"
    return payload


def _update_digest(digest, label: str, array) -> None:
    array = np.ascontiguousarray(array)
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype.str).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())


def _posterior_digest(posterior, fields: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(posterior.event_names)).encode("utf-8"))
    _update_digest(digest, "offsets", np.asarray(posterior.offsets, dtype="<i8"))
    for name in fields:
        _update_digest(digest, f"samples/{name}", np.asarray(posterior.samples[name], dtype="<f8"))
    _update_digest(digest, "log_ref_density", np.asarray(posterior.log_ref_density, dtype="<f8"))
    return digest.hexdigest()


def _selection_digest(selection, fields: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in fields:
        _update_digest(digest, f"samples/{name}", np.asarray(selection.samples[name], dtype="<f8"))
    _update_digest(digest, "log_draw_density", np.asarray(selection.log_draw_density, dtype="<f8"))
    digest.update(json.dumps(selection.campaign_id.tolist()).encode("utf-8"))
    digest.update(str(selection.estimator_semantics).encode("utf-8"))
    return digest.hexdigest()


def _data_identity(posterior, selection, fields: Sequence[str]) -> dict[str, object]:
    return {
        "pe_basis": posterior.basis.identity,
        "selection_basis": selection.basis.identity,
        "event_names": list(posterior.event_names),
        "n_events": int(posterior.n_events),
        "pe_sample_counts": np.diff(posterior.offsets).astype(int).tolist(),
        "n_pe_samples": int(posterior.n_samples_total),
        "n_selected": int(selection.n_selected),
        "selection_mode": selection.mode.value,
        "selection_campaigns": [
            {
                "campaign_id": campaign.campaign_id,
                "n_draw": None if campaign.n_draw is None else int(campaign.n_draw),
                "observing_time_yr": (
                    None
                    if campaign.observing_time_yr is None
                    else float(campaign.observing_time_yr)
                ),
            }
            for campaign in selection.campaigns
        ],
        "density_fields": list(fields),
        "pe_sha256": _posterior_digest(posterior, fields),
        "selection_sha256": _selection_digest(selection, fields),
    }


def build_likelihood_identity(
    posterior,
    selection,
    population_model,
    names: Sequence[str],
    hbi_config=None,
) -> dict[str, object]:
    """What defines the shape log-likelihood function (JSON-normalized).

    ``parameter_names`` (column order), the HBI configuration, the population
    model configuration (``to_config()``; the qualified name for plain
    callables) and the data: event names, per-event sample counts, bases,
    selection mode/campaigns and SHA-256 digests of every array the
    likelihood reads. Stored with every result of an HBI likelihood so that
    diagnostics can verify they evaluate the same estimator.
    """
    from gwpop_search.hbi.common import density_required_fields

    from .numpyro import _hbi_config_dict

    names = _validated_names(names, what="hyperparameter names")
    fields = density_required_fields(population_model, posterior.basis)
    return _json_normalized(
        {
            "parameter_names": list(names),
            "hbi_config": _hbi_config_dict(hbi_config),
            "model": _model_identity(population_model),
            "data": _data_identity(posterior, selection, fields),
        }
    )


def _hbi_config_from_dict(payload: Mapping[str, object]):
    from gwpop_search.hbi import HBIConfig

    known = {
        "rate_treatment",
        "raw_selection_use_observing_time",
        "selection_chunk_size",
        "variance_taper",
    }
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown HBI configuration field(s) {unknown}")
    return HBIConfig(
        rate_treatment=payload["rate_treatment"],
        raw_selection_use_observing_time=bool(payload["raw_selection_use_observing_time"]),
        selection_chunk_size=payload["selection_chunk_size"],
        variance_taper=payload.get("variance_taper"),
    )


def _without_chunk_size(identity: Mapping[str, object]) -> dict[str, object]:
    """Likelihood identity minus the selection chunk size (evaluation order only)."""
    result = copy.deepcopy(dict(identity))
    hbi = result.get("hbi_config")
    if isinstance(hbi, dict):
        hbi.pop("selection_chunk_size", None)
    return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_int(name: str, value, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def _as_float(name: str, value, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number; got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value}")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


def _validated_names(names: Sequence[str], *, what: str = "names") -> tuple[str, ...]:
    result = tuple(str(name) for name in names)
    if not result:
        raise ValueError(f"{what} cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{what} must be unique; got {result}")
    return result


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynestyConfig:
    """Static nested-sampling settings.

    ``batch_size`` is simultaneously dynesty's ``queue_size`` (the number of
    live-point replacements proposed in parallel) and the fixed number of rows
    of every device likelihood call. ``maxiter``/``maxcall`` use dynesty's
    semantics (``maxiter=M`` performs ``M + 1`` iterations) and are total
    budgets across checkpoint/resume sessions. ``update_interval`` follows
    dynesty: an ``int`` counts likelihood calls, a ``float`` is a fraction of
    ``nlive``. ``checkpoint_every`` is in seconds. ``checkpoint_every`` and
    ``num_posterior_samples`` do not affect the sampling trajectory and are
    excluded from the resume identity (:meth:`identity_dict`).
    """

    nlive: int = 500
    bound: str = "multi"
    sample: str = "rslice"
    slices: int | None = None
    walks: int | None = None
    bootstrap: int | None = None
    enlarge: float | None = None
    update_interval: int | float | None = None
    dlogz: float = 0.1
    maxiter: int | None = None
    maxcall: int | None = None
    batch_size: int = 64
    checkpoint_every: float = 600.0
    num_posterior_samples: int = 4000

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "nlive", _as_int("nlive", self.nlive, minimum=2))
        if self.bound not in _BOUNDS:
            raise ValueError(f"bound must be one of {_BOUNDS}; got {self.bound!r}")
        if self.sample not in _SAMPLERS:
            raise ValueError(f"sample must be one of {_SAMPLERS}; got {self.sample!r}")
        if self.slices is not None:
            if self.sample not in _SLICE_SAMPLERS:
                raise ValueError("slices is only meaningful for sample='slice'/'rslice'")
            set_(self, "slices", _as_int("slices", self.slices, minimum=1))
        if self.walks is not None:
            if self.sample != "rwalk":
                raise ValueError("walks is only meaningful for sample='rwalk'")
            set_(self, "walks", _as_int("walks", self.walks, minimum=2))
        if self.bootstrap is not None:
            bootstrap = _as_int("bootstrap", self.bootstrap, minimum=0)
            if bootstrap == 1:
                raise ValueError("bootstrap must be 0 (disabled) or > 1")
            set_(self, "bootstrap", bootstrap)
        if self.enlarge is not None:
            enlarge = _as_float("enlarge", self.enlarge)
            if enlarge < 1.0:
                raise ValueError(f"enlarge must be >= 1; got {enlarge}")
            set_(self, "enlarge", enlarge)
        if (
            self.enlarge is not None
            and self.bootstrap is not None
            and self.bootstrap != 0
            and self.enlarge != 1.0
        ):
            raise ValueError(
                "enlarge and bootstrap together are only allowed with bootstrap=0 "
                "or enlarge=1 (dynesty contract)"
            )
        if self.update_interval is not None:
            if isinstance(self.update_interval, bool):
                raise TypeError("update_interval cannot be a bool")
            if isinstance(self.update_interval, numbers.Integral):
                set_(
                    self,
                    "update_interval",
                    _as_int("update_interval", self.update_interval, minimum=1),
                )
            else:
                set_(
                    self,
                    "update_interval",
                    _as_float("update_interval", self.update_interval, positive=True),
                )
        set_(self, "dlogz", _as_float("dlogz", self.dlogz, positive=True))
        if self.maxiter is not None:
            set_(self, "maxiter", _as_int("maxiter", self.maxiter, minimum=1))
        if self.maxcall is not None:
            set_(self, "maxcall", _as_int("maxcall", self.maxcall, minimum=1))
        set_(self, "batch_size", _as_int("batch_size", self.batch_size, minimum=1))
        set_(
            self,
            "checkpoint_every",
            _as_float("checkpoint_every", self.checkpoint_every, positive=True),
        )
        set_(
            self,
            "num_posterior_samples",
            _as_int("num_posterior_samples", self.num_posterior_samples, minimum=1),
        )

    def to_dict(self) -> dict[str, object]:
        return {item.name: getattr(self, item.name) for item in dataclass_fields(self)}

    def identity_dict(self) -> dict[str, object]:
        """Fields that determine the sampling trajectory and where it stops.

        Excludes ``checkpoint_every`` (I/O cadence) and
        ``num_posterior_samples`` (resampling of the finished run); a run can
        be resumed or its finished result reused with different values.
        ``maxiter``/``maxcall`` are included: they set the stopping point.
        """
        payload = self.to_dict()
        for name in _NON_TRAJECTORY_FIELDS:
            payload.pop(name)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DynestyConfig":
        known = {item.name for item in dataclass_fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown DynestyConfig field(s): {unknown}")
        return cls(**dict(payload))


# ---------------------------------------------------------------------------
# Prior transform
# ---------------------------------------------------------------------------


class PriorTransform:
    """Picklable unit-cube -> hyperparameter map for :class:`PriorSpec` priors.

    Column ``k`` of the unit-cube vector maps to ``names[k]``:

    * ``uniform``:     ``low + u * (high - low)``
    * ``log_uniform``: ``low * (high / low) ** u``
    * ``normal``:      ``loc + scale * ndtri(u)``

    Accepts ``u`` of shape ``[ndim]`` or ``[m, ndim]``. It is a plain
    top-level class so that dynesty checkpoints (pickles) can store it.
    """

    def __init__(self, names: Sequence[str], specs: Sequence[PriorSpec]):
        names = _validated_names(names, what="prior names")
        specs = tuple(specs)
        if len(specs) != len(names):
            raise ValueError("names and specs must have the same length")
        for name, spec in zip(names, specs):
            if not isinstance(spec, PriorSpec):
                raise TypeError(f"prior for {name!r} must be a PriorSpec; got {type(spec)}")
            if spec.family not in _PRIOR_FAMILIES:  # pragma: no cover - PriorSpec guards
                raise ValueError(f"unsupported prior family {spec.family!r}")
        self.names = names
        self.specs = specs
        self.ndim = len(names)

        def index(family):
            return np.asarray(
                [i for i, spec in enumerate(specs) if spec.family == family], dtype=np.int64
            )

        self._uniform = index("uniform")
        self._uniform_low = np.asarray([specs[i].low for i in self._uniform], dtype=float)
        self._uniform_width = np.asarray(
            [specs[i].high - specs[i].low for i in self._uniform], dtype=float
        )
        self._log_uniform = index("log_uniform")
        self._log_uniform_low = np.asarray(
            [specs[i].low for i in self._log_uniform], dtype=float
        )
        self._log_uniform_ratio = np.asarray(
            [specs[i].high / specs[i].low for i in self._log_uniform], dtype=float
        )
        self._normal = index("normal")
        self._normal_loc = np.asarray([specs[i].loc for i in self._normal], dtype=float)
        self._normal_scale = np.asarray([specs[i].scale for i in self._normal], dtype=float)

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        if u.ndim not in (1, 2) or u.shape[-1] != self.ndim:
            raise ValueError(
                f"unit-cube input must have shape [{self.ndim}] or [m, {self.ndim}]; "
                f"got {u.shape}"
            )
        theta = np.empty(u.shape, dtype=float)
        if self._uniform.size:
            theta[..., self._uniform] = (
                self._uniform_low + u[..., self._uniform] * self._uniform_width
            )
        if self._log_uniform.size:
            theta[..., self._log_uniform] = self._log_uniform_low * np.power(
                self._log_uniform_ratio, u[..., self._log_uniform]
            )
        if self._normal.size:
            theta[..., self._normal] = self._normal_loc + self._normal_scale * ndtri(
                u[..., self._normal]
            )
        return theta

    def to_dict(self) -> dict[str, object]:
        priors = serialize_prior_map(dict(zip(self.names, self.specs)))
        return {"names": list(self.names), "priors": priors}

    def __eq__(self, other) -> bool:
        if not isinstance(other, PriorTransform):
            return NotImplemented
        return self.names == other.names and self.specs == other.specs

    def __hash__(self) -> int:
        return hash((self.names, self.specs))

    def __repr__(self) -> str:
        return f"PriorTransform(names={self.names!r})"


def prior_transform_for(
    priors: Mapping[str, PriorSpec],
) -> tuple[tuple[str, ...], PriorTransform]:
    """Return sorted parameter names and the unit-cube transform for ``priors``."""
    if not priors:
        raise ValueError("at least one hyperprior is required")
    names = tuple(sorted(str(name) for name in priors))
    if len(names) != len(priors):
        raise ValueError("prior names must be unique after str() conversion")
    lookup = {str(name): spec for name, spec in priors.items()}
    transform = PriorTransform(names, [lookup[name] for name in names])
    return names, transform


# ---------------------------------------------------------------------------
# Batched HBI likelihood
# ---------------------------------------------------------------------------


def _pad_rows(X: np.ndarray, batch_size: int) -> tuple[np.ndarray, int]:
    m = X.shape[0]
    n_blocks = -(-m // batch_size)
    padded = np.empty((n_blocks * batch_size, X.shape[1]), dtype=np.float64)
    padded[:m] = X
    padded[m:] = X[0]
    return padded, n_blocks


def _describe_rows(names, X, rows, *, limit: int = 4) -> str:
    parts = []
    for row in rows[:limit]:
        values = ", ".join(f"{name}={X[row, k]:.17g}" for k, name in enumerate(names))
        parts.append(f"row {int(row)}: {{{values}}}")
    more = "" if len(rows) <= limit else f" (+{len(rows) - limit} more)"
    return "; ".join(parts) + more


class BatchedShapeLogLikelihood:
    """Standardized HBI shape log-likelihood for a batch of hyperparameter vectors.

    ``f(X)`` with ``X`` of shape ``[m, ndim]`` (columns ordered as ``names``)
    returns ``log L`` of shape ``[m]`` as float64 NumPy. ``m`` is padded to a
    multiple of ``batch_size`` with copies of the first row so that the jitted
    ``vmap`` is compiled for a single ``[batch_size, ndim]`` shape.

    Each row is exactly the value of ``build_jax_shape_log_likelihood`` at
    ``{names[k]: X[row, k]}``; the event terms and log exposure are computed by
    the same :func:`gwpop_search.hbi.jax_backend.build_terms_function`. The
    extra outputs of the vmapped function only detect NaN/``+inf`` (which
    raise) and count exposure estimates without population support while
    every event is supported (``stats()["n_selection_unsupported"]``; the
    NumPy reference raises ``SelectionSupportError`` at such points, the JAX
    builder and this class return ``-inf``). :func:`run_dynesty` sums that
    count over sessions and warns when it is non-zero.

    :meth:`likelihood_identity` (names, HBI config, model, data digests) and
    :meth:`runtime_identity` (device and software runtime) are recorded by
    :func:`run_dynesty` in checkpoints and results.

    Variance taper: with ``hbi_config.variance_taper`` set, every row is the
    *tapered* shape likelihood ``ln L + ln T(sigma^2)`` of
    :func:`gwpop_search.hbi.jax_backend.build_shape_log_likelihood_components`
    (the same value as ``build_jax_shape_log_likelihood`` with that
    configuration), so dynesty samples and integrates ``L T`` and the
    evidence is ``int L T pi``. Per evaluation the variance ``sigma^2`` and
    ``ln T`` are computed in the same device pass; :meth:`stats` tallies the
    evaluations inside the taper region (``n_in_taper_region``), above the
    threshold (``n_above_taper_threshold``) and the largest finite
    ``sigma^2`` seen, and :meth:`components` returns the per-row
    ``(ln L T, ln L, sigma^2, ln T)``. ``with_variance=True`` computes
    ``sigma^2`` without a taper (``ln T = 0``; values equal up to rounding).
    """

    def __init__(
        self,
        posterior,
        selection,
        population_model,
        names: Sequence[str],
        *,
        hbi_config=None,
        batch_size: int = 64,
        with_variance: bool | None = None,
    ):
        jax, jnp = _require_jax()
        from gwpop_search.hbi import HBIConfig, RateTreatment
        from gwpop_search.hbi.jax_backend import (
            build_terms_and_variance_function,
            build_terms_function,
            tapered_shape_value,
        )

        cfg = HBIConfig() if hbi_config is None else hbi_config
        if cfg.rate_treatment is not RateTreatment.SHAPE:
            raise ValueError("the dynesty backend currently supports the shape likelihood only")
        self.names = _validated_names(names, what="hyperparameter names")
        self.ndim = len(self.names)
        self.batch_size = _as_int("batch_size", batch_size, minimum=1)
        self.n_events = int(posterior.n_events)
        self.hbi_config = cfg
        self.variance_taper = cfg.variance_taper
        self.with_variance = (
            cfg.variance_taper is not None if with_variance is None else bool(with_variance)
        )
        if cfg.variance_taper is not None and not self.with_variance:
            raise ValueError("a variance taper needs the variance (with_variance=True)")
        self._identity_inputs = (posterior, selection, population_model)
        self._likelihood_identity: dict | None = None
        names_ = self.names
        n_events = self.n_events
        taper = cfg.variance_taper
        self._variance_fn = None

        if self.with_variance:
            vterms = build_terms_and_variance_function(
                posterior, selection, population_model, config=cfg, jit=False
            )

            def single_variance(x):
                hyperparameters = {name: x[k] for k, name in enumerate(names_)}
                event_terms, log_exposure, event_var, sel_var = vterms(hyperparameters)
                tapered, value, variance, log_t = tapered_shape_value(
                    event_terms,
                    log_exposure,
                    event_var,
                    sel_var,
                    n_events=n_events,
                    taper=taper,
                )
                invalid = (
                    jnp.any(jnp.isnan(event_terms) | (event_terms == jnp.inf))
                    | jnp.isnan(log_exposure)
                    | (log_exposure == jnp.inf)
                )
                events_finite = jnp.all(jnp.isfinite(event_terms))
                selection_unsupported = events_finite & (log_exposure == -jnp.inf)
                return tapered, invalid, selection_unsupported, value, variance, log_t

            self._variance_fn = jax.jit(jax.vmap(single_variance))

        terms = build_terms_function(
            posterior, selection, population_model, config=cfg, jit=False
        )

        def single(x):
            hyperparameters = {name: x[k] for k, name in enumerate(names_)}
            event_terms, log_exposure = terms(hyperparameters)
            # Identical to gwpop_search.hbi.jax_backend.build_shape_log_likelihood.
            events_finite = jnp.all(jnp.isfinite(event_terms))
            valid = events_finite & jnp.isfinite(log_exposure)
            safe_events = jnp.where(jnp.isfinite(event_terms), event_terms, 0.0)
            safe_exposure = jnp.where(jnp.isfinite(log_exposure), log_exposure, 0.0)
            value = jnp.sum(safe_events) - n_events * safe_exposure
            value = jnp.where(valid, value, -jnp.inf)
            # Diagnostics only: NaN/+inf are bugs, and an exposure without any
            # finite population support is a Monte-Carlo failure worth counting.
            invalid = (
                jnp.any(jnp.isnan(event_terms) | (event_terms == jnp.inf))
                | jnp.isnan(log_exposure)
                | (log_exposure == jnp.inf)
            )
            selection_unsupported = events_finite & (log_exposure == -jnp.inf)
            return value, invalid, selection_unsupported

        self._device_fn = (
            self._variance_fn if self._variance_fn is not None else jax.jit(jax.vmap(single))
        )
        self._jax = jax
        self._jnp = jnp
        self._stats_lock = threading.Lock()
        self._stats = {
            "n_calls": 0,
            "n_rows": 0,
            "n_padded_rows": 0,
            "n_device_batches": 0,
            "device_seconds": 0.0,
            "n_zero_support": 0,
            "n_selection_unsupported": 0,
        }
        if self.with_variance:
            self._stats.update(
                {
                    "n_in_taper_region": 0,
                    "n_above_taper_threshold": 0,
                    "max_finite_variance": 0.0,
                }
            )

    def _evaluate(self, X) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.ndim:
            raise ValueError(
                f"hyperparameter block must have shape [m, {self.ndim}] "
                f"(columns {self.names}); got {X.shape}"
            )
        m = X.shape[0]
        if m == 0:
            return np.empty(0, dtype=np.float64), None
        bad_input = ~np.all(np.isfinite(X), axis=1)
        if bad_input.any():
            rows = np.flatnonzero(bad_input)
            raise ValueError(
                "non-finite hyperparameters passed to the likelihood: "
                + _describe_rows(self.names, X, rows)
            )
        padded, n_blocks = _pad_rows(X, self.batch_size)
        values = np.empty(padded.shape[0], dtype=np.float64)
        invalid = np.empty(padded.shape[0], dtype=bool)
        unsupported = np.empty(padded.shape[0], dtype=bool)
        extra = None
        if self.with_variance:
            extra = {
                key: np.empty(padded.shape[0], dtype=np.float64)
                for key in ("log_likelihood_untapered", "variance", "log_taper")
            }
        start = time.perf_counter()
        for block in range(n_blocks):
            sl = slice(block * self.batch_size, (block + 1) * self.batch_size)
            out = self._jax.device_get(self._device_fn(self._jnp.asarray(padded[sl])))
            values[sl] = np.asarray(out[0], dtype=np.float64)
            invalid[sl] = np.asarray(out[1], dtype=bool)
            unsupported[sl] = np.asarray(out[2], dtype=bool)
            if extra is not None:
                extra["log_likelihood_untapered"][sl] = np.asarray(out[3], dtype=np.float64)
                extra["variance"][sl] = np.asarray(out[4], dtype=np.float64)
                extra["log_taper"][sl] = np.asarray(out[5], dtype=np.float64)
        elapsed = time.perf_counter() - start
        values = values[:m]
        if extra is not None:
            extra = {key: value[:m] for key, value in extra.items()}
        invalid = invalid[:m] | np.isnan(values) | np.isposinf(values)
        if invalid.any():
            rows = np.flatnonzero(invalid)
            raise PopulationDensityError(
                f"HBI likelihood produced NaN/+inf event terms or selection exposure for "
                f"{rows.size} hyperparameter vector(s): {_describe_rows(self.names, X, rows)}. "
                "-inf is allowed only for genuine zero population support."
            )
        with self._stats_lock:
            stats = self._stats
            stats["n_calls"] += 1
            stats["n_rows"] += m
            stats["n_padded_rows"] += padded.shape[0] - m
            stats["n_device_batches"] += n_blocks
            stats["device_seconds"] += elapsed
            stats["n_zero_support"] += int(np.count_nonzero(np.isneginf(values)))
            stats["n_selection_unsupported"] += int(np.count_nonzero(unsupported[:m]))
            if extra is not None:
                variance = extra["variance"]
                supported = np.isfinite(extra["log_likelihood_untapered"])
                taper = self.variance_taper
                if taper is not None:
                    stats["n_in_taper_region"] += int(
                        np.count_nonzero(supported & taper.in_region(variance))
                    )
                    stats["n_above_taper_threshold"] += int(
                        np.count_nonzero(supported & ~(variance <= taper.threshold))
                    )
                finite = variance[supported & np.isfinite(variance)]
                if finite.size:
                    stats["max_finite_variance"] = max(
                        float(stats["max_finite_variance"]), float(np.max(finite))
                    )
        return values, extra

    def __call__(self, X) -> np.ndarray:
        return self._evaluate(X)[0]

    def components(self, X) -> dict[str, np.ndarray]:
        """Per row: ``log_likelihood`` (tapered), ``log_likelihood_untapered``,
        ``variance`` (``sigma^2_lnL``) and ``log_taper`` (needs the variance)."""
        if not self.with_variance:
            raise ValueError("components() needs a likelihood built with the variance")
        values, extra = self._evaluate(X)
        if extra is None:  # m == 0
            empty = np.empty(0, dtype=np.float64)
            return {
                "log_likelihood": empty,
                "log_likelihood_untapered": empty,
                "variance": empty,
                "log_taper": empty,
            }
        return {"log_likelihood": values, **extra}

    def stats(self) -> dict[str, object]:
        with self._stats_lock:
            return dict(self._stats)

    def likelihood_identity(self) -> dict[str, object]:
        """:func:`build_likelihood_identity` of this likelihood (computed once)."""
        if self._likelihood_identity is None:
            posterior, selection, model = self._identity_inputs
            self._likelihood_identity = build_likelihood_identity(
                posterior, selection, model, self.names, self.hbi_config
            )
        return copy.deepcopy(self._likelihood_identity)

    def runtime_identity(self) -> dict[str, object]:
        """Host and JAX device runtime this likelihood evaluates on."""
        return _runtime_identity(jax_devices=True)


def build_batched_log_likelihood(
    posterior,
    selection,
    population_model,
    names: Sequence[str],
    *,
    hbi_config=None,
    batch_size: int = 64,
    with_variance: bool | None = None,
) -> BatchedShapeLogLikelihood:
    """Build ``f(X [m, ndim]) -> log L [m]`` for the standardized shape likelihood.

    Use ``HBIConfig(selection_chunk_size=None)`` (or a chunk at least as large
    as the selection) on GPUs: the chunked ``lax.scan`` is sequential and was
    measured ~40x slower at ``chunk=4096`` on the GWTC-5 candidate data.
    With ``hbi_config.variance_taper`` the rows are the tapered likelihood
    (see :class:`BatchedShapeLogLikelihood`).
    """
    return BatchedShapeLogLikelihood(
        posterior,
        selection,
        population_model,
        names,
        hbi_config=hbi_config,
        batch_size=batch_size,
        with_variance=with_variance,
    )


# ---------------------------------------------------------------------------
# Thread-batching pool
# ---------------------------------------------------------------------------


class _BatchFailure(Exception):
    """Carries a batched-evaluation exception to one waiting worker thread."""

    def __init__(self, original: BaseException):
        super().__init__(f"batched log-likelihood evaluation failed: {original!r}")
        self.original = original


class _Signal:
    """One-shot wake-up built on a pre-acquired lock (cheaper than threading.Event)."""

    __slots__ = ("_lock", "_fired")

    def __init__(self):
        self._lock = threading.Lock()
        self._lock.acquire()
        self._fired = False

    @property
    def fired(self) -> bool:
        return self._fired

    def set(self) -> None:
        # Callers hold the pool lock or own the request exclusively, so the
        # flag check cannot race; it makes delivery idempotent.
        if not self._fired:
            self._fired = True
            self._lock.release()

    def wait(self) -> None:
        self._lock.acquire()
        self._lock.release()


class _Request:
    __slots__ = ("x", "job", "event", "value", "error")

    def __init__(self, x: np.ndarray, job: "_MapJob"):
        self.x = x
        self.job = job
        self.event = _Signal()
        self.value = None
        self.error: BaseException | None = None


class _MapJob:
    __slots__ = (
        "func",
        "items",
        "n_items",
        "next_index",
        "n_done",
        "results",
        "errors",
        "cancelled",
    )

    def __init__(self, func, items):
        self.func = func
        self.items = items
        self.n_items = len(items)
        self.next_index = 0
        self.n_done = 0
        self.results = [None] * self.n_items
        self.errors: list[BaseException | None] = [None] * self.n_items
        self.cancelled = False


class PooledPointLogLikelihood:
    """Per-point log-likelihood handed to dynesty; routes through a pool.

    dynesty checkpoints pickle the sampler including its likelihood. This
    proxy pickles without its pool (threads, locks and device buffers cannot
    be pickled); :func:`run_dynesty` rebinds a fresh pool after a restore.
    """

    def __init__(self, pool: "ThreadBatchPool | None" = None):
        self._pool = pool

    def bind(self, pool: "ThreadBatchPool") -> None:
        self._pool = pool

    def __call__(self, theta) -> float:
        pool = self._pool
        if pool is None:
            raise RuntimeError(
                "pooled log-likelihood is not bound to a ThreadBatchPool "
                "(restored from a checkpoint without rebinding?)"
            )
        return pool.evaluate(theta)

    def __getstate__(self):
        # Must be truthy: pickle skips __setstate__ for a falsy state.
        return {"pool": None}

    def __setstate__(self, state):
        self._pool = None


class ThreadBatchPool:
    """dynesty-compatible pool that batches likelihood calls across threads.

    ``map(func, items)`` runs ``func(item)`` for every item concurrently in
    ``n_workers`` persistent worker threads and returns the results in item
    order (dynesty uses it both for the initial live-point likelihoods and for
    its queue of live-point proposals). Inside a task, every likelihood call
    made through :attr:`point_loglikelihood` (or :meth:`evaluate`) is queued
    and blocks until a single dispatcher thread evaluates the pending requests
    together with ``batched_loglikelihood(X [m, ndim]) -> [m]``.

    Flush rule: the dispatcher evaluates up to ``batch_size`` pending requests
    as soon as ``batch_size`` requests are pending or every busy worker is
    blocked on a request.

    Deadlock freedom: a busy worker that is not blocked on a request is
    executing task code; it either finishes its task (``n_busy`` decreases) or
    submits a request (``n_waiting`` increases). Both events re-evaluate the
    flush rule, so the dispatcher can only wait while some worker is making
    progress, and every pending request is eventually evaluated or failed.

    Errors: NaN/``+inf`` values from the batched callable raise
    :class:`PopulationDensityError`; any exception inside a batch is delivered
    to every request of that batch. The first failing task (in item order,
    ignoring cancellations) is re-raised by ``map`` with its original type
    after all tasks have stopped; other tasks of the same map fail fast with
    :class:`PoolCancelledError` at their next likelihood call.

    Determinism: results do not depend on which requests share a batch as
    long as ``batched_loglikelihood`` evaluates each row independently (the
    HBI batched likelihood does).
    """

    def __init__(
        self,
        batched_loglikelihood: Callable[[np.ndarray], np.ndarray],
        batch_size: int,
        *,
        n_workers: int | None = None,
    ):
        if not callable(batched_loglikelihood):
            raise TypeError("batched_loglikelihood must be callable")
        self._fn = batched_loglikelihood
        self.batch_size = _as_int("batch_size", batch_size, minimum=1)
        # dynesty infers queue_size from ``pool.size`` when restoring.
        self.size = self.batch_size
        self.n_workers = (
            self.batch_size if n_workers is None else _as_int("n_workers", n_workers, minimum=1)
        )
        self._lock = threading.Lock()
        self._work_cv = threading.Condition(self._lock)
        self._dispatch_cv = threading.Condition(self._lock)
        self._done_cv = threading.Condition(self._lock)
        self._eval_lock = threading.Lock()
        self._tls = threading.local()
        self._pending: list[_Request] = []
        self._in_flight: list[_Request] = []
        self._n_busy = 0
        self._n_waiting = 0
        self._job: _MapJob | None = None
        self._closed = False
        self._broken: BaseException | None = None
        self._created = time.perf_counter()
        self._stats = {
            "n_maps": 0,
            "n_tasks": 0,
            "n_evaluations": 0,
            "n_zero_likelihood": 0,
            "n_batches": 0,
            "n_full_batches": 0,
            "evaluation_seconds": 0.0,
        }
        self.point_loglikelihood = PooledPointLogLikelihood(self)
        self._threads = [
            threading.Thread(
                target=self._worker_main, name=f"gwpop-dynesty-worker-{i}", daemon=True
            )
            for i in range(self.n_workers)
        ]
        self._dispatcher = threading.Thread(
            target=self._dispatch_main, name="gwpop-dynesty-dispatcher", daemon=True
        )
        for thread in self._threads:
            thread.start()
        self._dispatcher.start()

    # -- public API -------------------------------------------------------

    @property
    def batched_loglikelihood(self) -> Callable[[np.ndarray], np.ndarray]:
        """The batched callable every likelihood request is evaluated with."""
        return self._fn

    def map(self, func, iterable) -> list:
        items = list(iterable)
        if not items:
            return []
        if getattr(self._tls, "is_worker", False):
            raise RuntimeError("ThreadBatchPool.map cannot be called from its own worker threads")
        job = _MapJob(func, items)
        with self._lock:
            self._check_usable_locked()
            if self._job is not None:
                raise RuntimeError("ThreadBatchPool.map is not re-entrant")
            self._job = job
            self._stats["n_maps"] += 1
            self._stats["n_tasks"] += job.n_items
            self._work_cv.notify_all()
            try:
                while job.n_done < job.n_items:
                    if self._broken is not None:
                        raise RuntimeError("ThreadBatchPool dispatcher failed") from self._broken
                    if self._closed:
                        raise RuntimeError("ThreadBatchPool was closed during map()")
                    self._done_cv.wait()
            except BaseException:
                self._cancel_job_locked(job)
                raise
            finally:
                if self._job is job:
                    self._job = None
        failures = [error for error in job.errors if error is not None]
        if failures:
            primary = next(
                (error for error in failures if not isinstance(error, PoolCancelledError)),
                failures[0],
            )
            if isinstance(primary, _BatchFailure):
                raise primary.original
            raise primary
        return job.results

    def evaluate(self, theta) -> float:
        """Log-likelihood of one hyperparameter vector (batched when called from a task)."""
        x = np.asarray(theta, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"expected a 1-D hyperparameter vector; got shape {x.shape}")
        job = getattr(self._tls, "job", None)
        if job is None:
            # Not inside one of this pool's tasks: evaluate synchronously.
            with self._lock:
                self._check_usable_locked()
            return float(self._evaluate_rows(x[None, :])[0])
        request = _Request(x, job)
        with self._lock:
            self._check_usable_locked()
            if job.cancelled:
                raise PoolCancelledError("task abandoned: another task in this map failed")
            self._pending.append(request)
            self._n_waiting += 1
            self._wake_dispatcher_if_flushable_locked()
        request.event.wait()
        if request.error is not None:
            if isinstance(request.error, PoolCancelledError):
                raise PoolCancelledError(str(request.error))
            raise _BatchFailure(request.error)
        return request.value

    def stats(self) -> dict[str, object]:
        with self._lock:
            stats = dict(self._stats)
        stats["batch_size"] = self.batch_size
        stats["n_workers"] = self.n_workers
        stats["wall_seconds"] = time.perf_counter() - self._created
        stats["mean_batch_fill"] = (
            stats["n_evaluations"] / stats["n_batches"] if stats["n_batches"] else 0.0
        )
        stats["evaluations_per_second"] = (
            stats["n_evaluations"] / stats["wall_seconds"] if stats["wall_seconds"] > 0 else 0.0
        )
        return stats

    def close(self, timeout: float = 30.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._job is not None:
                self._cancel_job_locked(self._job)
            self._work_cv.notify_all()
            self._dispatch_cv.notify_all()
            self._done_cv.notify_all()
        deadline = time.monotonic() + float(timeout)
        for thread in [*self._threads, self._dispatcher]:
            thread.join(max(0.0, deadline - time.monotonic()))

    def __enter__(self) -> "ThreadBatchPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __getstate__(self):
        raise TypeError("ThreadBatchPool holds threads and cannot be pickled")

    # -- internals --------------------------------------------------------

    def _check_usable_locked(self) -> None:
        if self._closed:
            raise RuntimeError("ThreadBatchPool is closed")
        if self._broken is not None:
            raise RuntimeError("ThreadBatchPool dispatcher failed") from self._broken

    def _expected_busy_locked(self) -> int:
        """Busy workers plus idle workers that are about to claim unclaimed tasks.

        Without the second term the first worker to reach a likelihood call
        after a new map() starts would find ``n_waiting == n_busy`` before the
        other workers have claimed their tasks and flush an almost empty batch.
        """
        job = self._job
        if job is None or self._closed:
            return self._n_busy
        unclaimed = job.n_items - job.next_index
        idle = self.n_workers - self._n_busy
        return self._n_busy + max(0, min(idle, unclaimed))

    def _flushable_locked(self) -> bool:
        n_pending = len(self._pending)
        return n_pending > 0 and (
            n_pending >= self.batch_size or self._n_waiting >= self._expected_busy_locked()
        )

    def _wake_dispatcher_if_flushable_locked(self) -> None:
        if self._flushable_locked():
            self._dispatch_cv.notify()

    def _cancel_job_locked(self, job: _MapJob) -> None:
        if job.cancelled:
            return
        job.cancelled = True
        keep = []
        for request in self._pending:
            if request.job is job:
                request.error = PoolCancelledError(
                    "task abandoned: another task in this map failed"
                )
                self._n_waiting -= 1
                request.event.set()
            else:
                keep.append(request)
        self._pending = keep
        self._wake_dispatcher_if_flushable_locked()

    def _finish_task_locked(self, job: _MapJob) -> None:
        job.n_done += 1
        if job.n_done == job.n_items:
            self._done_cv.notify_all()

    def _worker_main(self) -> None:
        self._tls.is_worker = True
        lock = self._lock
        busy = False
        lock.acquire()
        try:
            while True:
                job = self._job
                if not self._closed and job is not None and job.next_index < job.n_items:
                    index = job.next_index
                    job.next_index += 1
                    if not busy:
                        busy = True
                        self._n_busy += 1
                    if job.cancelled:
                        job.errors[index] = PoolCancelledError(
                            "task skipped: another task in this map failed"
                        )
                        self._finish_task_locked(job)
                        continue
                    self._tls.job = job
                    lock.release()
                    try:
                        value = job.func(job.items[index])
                        error = None
                    except BaseException as exc:  # noqa: BLE001 - re-raised by map()
                        value, error = None, exc
                    finally:
                        lock.acquire()
                        self._tls.job = None
                    job.results[index] = value
                    if error is not None:
                        job.errors[index] = error
                        if not isinstance(error, PoolCancelledError):
                            self._cancel_job_locked(job)
                    self._finish_task_locked(job)
                    continue
                if busy:
                    busy = False
                    self._n_busy -= 1
                    self._wake_dispatcher_if_flushable_locked()
                if self._closed:
                    return
                self._work_cv.wait()
        finally:
            if busy:
                self._n_busy -= 1
            lock.release()

    def _evaluate_rows(self, X: np.ndarray) -> np.ndarray:
        with self._eval_lock:
            start = time.perf_counter()
            values = np.asarray(self._fn(X), dtype=np.float64)
            elapsed = time.perf_counter() - start
        if values.shape != (X.shape[0],):
            raise ValueError(
                f"batched log-likelihood returned shape {values.shape}; expected ({X.shape[0]},)"
            )
        bad = np.isnan(values) | np.isposinf(values)
        if bad.any():
            rows = np.flatnonzero(bad)
            raise PopulationDensityError(
                f"batched log-likelihood returned NaN/+inf for {rows.size} row(s); first rows "
                f"{rows[:8].tolist()} with parameters {X[rows[:4]].tolist()}. "
                "-inf is allowed only for genuine zero support."
            )
        with self._lock:
            stats = self._stats
            stats["n_evaluations"] += X.shape[0]
            stats["n_zero_likelihood"] += int(np.count_nonzero(np.isneginf(values)))
            stats["n_batches"] += 1
            stats["n_full_batches"] += int(X.shape[0] == self.batch_size)
            stats["evaluation_seconds"] += elapsed
        return values

    def _dispatch_main(self) -> None:
        lock = self._lock
        try:
            while True:
                with lock:
                    while not self._closed and not self._flushable_locked():
                        self._dispatch_cv.wait()
                    if self._closed:
                        for request in self._pending:
                            request.error = RuntimeError("ThreadBatchPool closed")
                            request.event.set()
                        self._n_waiting -= len(self._pending)
                        self._pending = []
                        return
                    batch = self._pending[: self.batch_size]
                    del self._pending[: self.batch_size]
                    self._in_flight = batch
                try:
                    values = self._evaluate_rows(np.stack([request.x for request in batch]))
                    error = None
                except BaseException as exc:  # noqa: BLE001 - delivered to every request
                    values, error = None, exc
                with lock:
                    self._n_waiting -= len(batch)
                    self._in_flight = []
                for k, request in enumerate(batch):
                    if error is None:
                        request.value = float(values[k])
                    else:
                        request.error = error
                    request.event.set()
        except BaseException as exc:  # pragma: no cover - defensive: never strand workers
            with lock:
                self._broken = exc
                stranded = [*self._in_flight, *self._pending]
                self._pending = []
                self._in_flight = []
                self._done_cv.notify_all()
            for request in stranded:
                if not request.event.fired:
                    request.error = RuntimeError("ThreadBatchPool dispatcher failed")
                    request.event.set()
            raise


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DynestyResult:
    """Posterior and evidence from one static dynesty run.

    ``samples``/``log_weights``/``log_likelihoods``/``log_volumes`` are the
    dead points followed by the final live points, exactly as dynesty
    integrates them (``log_weights`` are dynesty's unnormalized ``logwt``;
    points with zero likelihood, stored by dynesty as ``logl = -1e300``, are
    reported with ``-inf`` likelihood and weight). ``posterior_samples`` are
    ``config.num_posterior_samples`` equal-weight draws (systematic
    resampling, see :func:`equal_weight_resample`, with the generator
    ``default_rng([seed, tag])``; :meth:`with_num_posterior_samples` redraws
    them exactly as a run with that size would).

    Counters are totals over the checkpoint/resume sessions that produced the
    result (a killed session's evaluations after its last checkpoint are
    repeated by the resume and counted once, so they equal an uninterrupted
    run): ``n_likelihood_evaluations`` is the exact number of likelihood
    evaluations; ``n_selection_unsupported`` counts evaluations whose events
    were all supported but whose selection exposure had no population
    support (treated as zero likelihood; ``None`` when the likelihood does not
    report it); ``elapsed_seconds`` is the sampling wall time. dynesty's
    ``ncall`` (initial live points included) differs from
    ``n_likelihood_evaluations``: it also counts slice steps outside the unit
    cube, which are rejected without an evaluation, and it does not count the
    evaluations of proposals still queued when the run stopped
    (``diagnostics["n_queued_evaluations"]``; ``rwalk`` also re-evaluates,
    uncounted, the start of a walk that accepted no step). For
    ``sample="unif"``, ``ncall + n_queued_evaluations ==
    n_likelihood_evaluations`` exactly; for the slice samplers
    ``ncall + n_queued_evaluations >= n_likelihood_evaluations``.
    ``efficiency`` is dynesty's ``eff`` in percent (``niter / ncall``).

    ``provenance`` records ``code`` (package version, git commit,
    ``git_dirty``, source digest), ``versions``, ``runtime`` (host and device),
    ``likelihood`` (:func:`build_likelihood_identity` of an HBI likelihood,
    else ``None``) and ``run`` (caller identity, e.g. the population
    manifest digest).
    """

    names: tuple[str, ...]
    log_evidence: float
    log_evidence_error: float
    information: float
    niter: int
    ncall: int
    efficiency: float
    samples: np.ndarray
    log_weights: np.ndarray
    log_likelihoods: np.ndarray
    log_volumes: np.ndarray
    posterior_samples: np.ndarray
    kish_ess: float
    elapsed_seconds: float
    config: DynestyConfig
    seed: int
    versions: Mapping[str, str]
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    n_likelihood_evaluations: int | None = None
    n_selection_unsupported: int | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    @property
    def weights(self) -> np.ndarray:
        """Normalized importance weights of :attr:`samples`."""
        return _normalized_weights(self.log_weights)

    @property
    def likelihood_identity(self) -> Mapping[str, object] | None:
        """The likelihood this result sampled (``None`` for non-HBI likelihoods)."""
        return self.provenance.get("likelihood")

    def posterior(self) -> dict[str, np.ndarray]:
        """Equal-weight posterior samples keyed by parameter name."""
        return {name: self.posterior_samples[:, k] for k, name in enumerate(self.names)}

    def with_num_posterior_samples(self, num_posterior_samples: int) -> "DynestyResult":
        """This result with ``num_posterior_samples`` equal-weight draws.

        The draws are identical to those of a run with
        ``config.num_posterior_samples = num_posterior_samples`` (the sampling
        trajectory does not depend on it); ``config`` is updated accordingly.
        """
        config = replace(self.config, num_posterior_samples=num_posterior_samples)
        draws = _equal_weight_posterior(
            self.samples, self.log_weights, config.num_posterior_samples, self.seed
        )
        return replace(self, config=config, posterior_samples=draws)


def equal_weight_resample(samples, weights, n: int, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling of weighted ``samples`` into ``n`` equal-weight draws.

    This is dynesty's :func:`dynesty.utils.resample_equal` (Hol, Schon &
    Gustafsson 2006) generalized to an arbitrary output size: one uniform
    offset, ``n`` evenly spaced positions, then a random permutation. For
    ``n == len(weights)`` it reproduces ``resample_equal`` exactly for the same
    generator state (tested). Each input appears ``floor(n w_i)`` or
    ``ceil(n w_i)`` times.
    """
    samples = np.asarray(samples)
    weights = np.asarray(weights, dtype=np.float64)
    n = _as_int("n", n, minimum=1)
    if weights.ndim != 1 or weights.size != samples.shape[0] or weights.size == 0:
        raise ValueError("weights must be 1-D with one entry per sample")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    positions = (rng.random() + np.arange(n)) / n
    index = np.searchsorted(cumulative, positions, side="right")
    return rng.permutation(samples[index])


def _normalized_weights(log_weights) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=np.float64)
    return np.exp(log_weights - logsumexp(log_weights))


def _equal_weight_posterior(samples, log_weights, n: int, seed: int) -> np.ndarray:
    """The run's equal-weight draws: one seeded stream per (seed, size)."""
    rng = np.random.default_rng([int(seed), _RESAMPLE_STREAM])
    return equal_weight_resample(samples, _normalized_weights(log_weights), n, rng)


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _result_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def save_dynesty_result(path: str | Path, result: DynestyResult) -> None:
    """Write ``result`` as ``path`` (npz arrays) plus ``path + '.json'`` metadata.

    Both files are written atomically; the npz is written last and marks a
    complete result.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": RESULT_FORMAT_VERSION,
        "names": list(result.names),
        "log_evidence": float(result.log_evidence),
        "log_evidence_error": float(result.log_evidence_error),
        "information": float(result.information),
        "niter": int(result.niter),
        "ncall": int(result.ncall),
        "efficiency": float(result.efficiency),
        "kish_ess": float(result.kish_ess),
        "elapsed_seconds": float(result.elapsed_seconds),
        "config": result.config.to_dict(),
        "seed": int(result.seed),
        "versions": dict(result.versions),
        "diagnostics": _json_ready(result.diagnostics),
        "n_likelihood_evaluations": _optional_int(result.n_likelihood_evaluations),
        "n_selection_unsupported": _optional_int(result.n_selection_unsupported),
        "provenance": _json_ready(result.provenance),
    }
    _atomic_write_text(
        _result_sidecar(path), json.dumps(metadata, sort_keys=True, indent=2, allow_nan=True)
    )
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            samples=np.asarray(result.samples, dtype=np.float64),
            log_weights=np.asarray(result.log_weights, dtype=np.float64),
            log_likelihoods=np.asarray(result.log_likelihoods, dtype=np.float64),
            log_volumes=np.asarray(result.log_volumes, dtype=np.float64),
            posterior_samples=np.asarray(result.posterior_samples, dtype=np.float64),
        )
    os.replace(tmp, path)


def dynesty_result_exists(path: str | Path) -> bool:
    path = Path(path)
    return path.exists() and _result_sidecar(path).exists()


def load_dynesty_result(path: str | Path) -> DynestyResult:
    path = Path(path)
    sidecar = _result_sidecar(path)
    if not path.exists() or not sidecar.exists():
        raise FileNotFoundError(f"incomplete dynesty result: need {path} and {sidecar}")
    metadata = json.loads(sidecar.read_text())
    if metadata.get("format_version") not in _READABLE_RESULT_FORMATS:
        raise ValueError(
            f"unsupported dynesty result format {metadata.get('format_version')!r} in {sidecar}"
        )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    names = tuple(metadata["names"])
    n_points = arrays["log_weights"].shape[0]
    expected = {
        "samples": (n_points, len(names)),
        "log_likelihoods": (n_points,),
        "log_volumes": (n_points,),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{path}: {key} has shape {arrays[key].shape}; expected {shape}")
    if arrays["posterior_samples"].ndim != 2 or arrays["posterior_samples"].shape[1] != len(names):
        raise ValueError(f"{path}: posterior_samples has shape {arrays['posterior_samples'].shape}")
    return DynestyResult(
        names=names,
        log_evidence=float(metadata["log_evidence"]),
        log_evidence_error=float(metadata["log_evidence_error"]),
        information=float(metadata["information"]),
        niter=int(metadata["niter"]),
        ncall=int(metadata["ncall"]),
        efficiency=float(metadata["efficiency"]),
        samples=arrays["samples"],
        log_weights=arrays["log_weights"],
        log_likelihoods=arrays["log_likelihoods"],
        log_volumes=arrays["log_volumes"],
        posterior_samples=arrays["posterior_samples"],
        kish_ess=float(metadata["kish_ess"]),
        elapsed_seconds=float(metadata["elapsed_seconds"]),
        config=DynestyConfig.from_dict(metadata["config"]),
        seed=int(metadata["seed"]),
        versions=dict(metadata["versions"]),
        diagnostics=dict(metadata.get("diagnostics", {})),
        n_likelihood_evaluations=_optional_int(metadata.get("n_likelihood_evaluations")),
        n_selection_unsupported=_optional_int(metadata.get("n_selection_unsupported")),
        provenance=dict(metadata.get("provenance") or {}),
    )


# ---------------------------------------------------------------------------
# Running dynesty
# ---------------------------------------------------------------------------


def _checkpoint_meta(
    names: Sequence[str],
    seed: int,
    config: DynestyConfig,
    identity: Mapping[str, object],
) -> dict:
    """Resume identity stamped on every sampler (JSON-normalized)."""
    return _json_normalized(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "names": list(names),
            "seed": int(seed),
            "config": config.identity_dict(),
            "identity": dict(identity),
        }
    )


def _run_identity(loglike_batched, run: Mapping[str, object] | None = None) -> dict:
    """Code, software, runtime and likelihood identity of a run (JSON-normalized).

    A batched likelihood may expose ``likelihood_identity()`` and
    ``runtime_identity()`` (:class:`BatchedShapeLogLikelihood` does).
    Otherwise the likelihood identity is ``None`` and only the host runtime is
    recorded, without initializing a JAX backend. ``run`` is an optional
    caller identity (e.g. the population manifest digest).
    """
    likelihood_fn = getattr(loglike_batched, "likelihood_identity", None)
    runtime_fn = getattr(loglike_batched, "runtime_identity", None)
    return _json_normalized(
        {
            "code": _code_identity(),
            "versions": _software_versions(),
            "runtime": (
                runtime_fn() if callable(runtime_fn) else _runtime_identity(jax_devices=False)
            ),
            "likelihood": likelihood_fn() if callable(likelihood_fn) else None,
            "run": None if run is None else dict(run),
        }
    )


def _add_delta(base, now, start):
    if base is None or now is None or start is None:
        return None
    return int(base) + int(now) - int(start)


class _RunTally:
    """Run counters that travel inside dynesty checkpoints.

    dynesty pickles the sampler only between iterations, never while a
    ``pool.map`` is running, and the pickled state reflects every likelihood
    evaluation made so far (queued proposals included). Pickling therefore
    stores ``base + (current session - session start)``: evaluations that a
    killed session made after its last checkpoint are lost with it and are
    repeated by the resume, so the totals equal those of an uninterrupted
    run. A counter a session cannot measure (a likelihood without ``stats()``)
    becomes ``None`` for the whole run. The variance-taper counters
    (``_TAPER_KEYS``: likelihood evaluations inside the taper region and
    above its threshold) are recorded only when the likelihood reports them.
    """

    _POOL_KEYS = ("n_evaluations", "n_zero_likelihood")
    _LIKELIHOOD_KEYS = ("n_selection_unsupported",)
    _TAPER_KEYS = ("n_in_taper_region", "n_above_taper_threshold")

    def __init__(self):
        self.base: dict[str, object] = {
            key: 0 for key in (*self._POOL_KEYS, *self._LIKELIHOOD_KEYS)
        }
        self.base["elapsed_seconds"] = 0.0
        self._pool: ThreadBatchPool | None = None
        self._likelihood_stats = None
        self._start: dict | None = None

    def bind(self, pool: ThreadBatchPool, clock_start: float) -> None:
        """Count this session's evaluations from ``pool`` (and its likelihood's stats)."""
        stats = getattr(pool.batched_loglikelihood, "stats", None)
        self._pool = pool
        self._likelihood_stats = stats if callable(stats) else None
        self._start = {
            "pool": pool.stats(),
            "likelihood": None if self._likelihood_stats is None else self._likelihood_stats(),
            "clock": float(clock_start),
        }

    def snapshot(self) -> dict[str, object]:
        totals = dict(self.base)
        if self._pool is None:
            return totals
        start = self._start
        pool_now = self._pool.stats()
        for key in self._POOL_KEYS:
            totals[key] = _add_delta(self.base.get(key), pool_now.get(key), start["pool"].get(key))
        like_now = None if self._likelihood_stats is None else self._likelihood_stats()
        for key in self._LIKELIHOOD_KEYS:
            if like_now is None or start["likelihood"] is None:
                totals[key] = None
            else:
                totals[key] = _add_delta(
                    self.base.get(key), like_now.get(key), start["likelihood"].get(key)
                )
        for key in self._TAPER_KEYS:
            if (
                like_now is not None
                and start["likelihood"] is not None
                and key in like_now
                and key in start["likelihood"]
            ):
                totals[key] = _add_delta(
                    self.base.get(key, 0), like_now[key], start["likelihood"][key]
                )
        totals["elapsed_seconds"] = float(self.base["elapsed_seconds"]) + (
            time.perf_counter() - start["clock"]
        )
        return totals

    def __getstate__(self):
        return {"base": self.snapshot()}

    def __setstate__(self, state):
        self.base = dict(state["base"])
        self._pool = None
        self._likelihood_stats = None
        self._start = None


def _new_sampler(
    dynesty,
    pool: ThreadBatchPool,
    prior_transform,
    ndim: int,
    *,
    seed: int,
    config: DynestyConfig,
    names: Sequence[str] | None = None,
    identity: Mapping[str, object] | None = None,
    tally: _RunTally | None = None,
):
    """Construct the static sampler (evaluates the initial live points via the pool).

    ``identity`` defaults to the run identity of the pool's likelihood and
    ``tally`` to a fresh tally bound to ``pool``; both are stamped on the
    sampler so that checkpoints carry them.
    """
    names = tuple(f"x{k}" for k in range(ndim)) if names is None else tuple(names)
    if identity is None:
        identity = _run_identity(pool.batched_loglikelihood)
    if tally is None:
        tally = _RunTally()
        tally.bind(pool, time.perf_counter())
    sampler = dynesty.NestedSampler(
        pool.point_loglikelihood,
        prior_transform,
        ndim,
        nlive=config.nlive,
        bound=config.bound,
        sample=config.sample,
        update_interval=config.update_interval,
        rstate=np.random.default_rng(seed),
        queue_size=config.batch_size,
        pool=pool,
        use_pool=dict(_USE_POOL),
        walks=config.walks,
        slices=config.slices,
        bootstrap=config.bootstrap,
        enlarge=config.enlarge,
    )
    setattr(sampler, _CHECKPOINT_META_ATTR, _checkpoint_meta(names, seed, config, identity))
    setattr(sampler, _TALLY_ATTR, tally)
    return sampler


def _restore_sampler(
    dynesty,
    checkpoint: Path,
    pool: ThreadBatchPool,
    prior_transform,
    ndim: int,
    config: DynestyConfig,
    *,
    seed: int,
    names: tuple[str, ...],
    identity: Mapping[str, object],
    clock_start: float,
):
    """Restore a checkpoint of the requested run; returns ``(sampler, tally)``."""
    sampler = dynesty.NestedSampler.restore(str(checkpoint), pool=pool)
    wrapper = getattr(getattr(sampler, "loglikelihood", None), "loglikelihood", None)
    proxy = getattr(wrapper, "func", None)
    meta = getattr(sampler, _CHECKPOINT_META_ATTR, None)
    if not isinstance(proxy, PooledPointLogLikelihood) or not isinstance(meta, dict):
        raise RuntimeError(
            f"{checkpoint} was not written by gwpop-search run_dynesty "
            "(missing pooled likelihood or checkpoint metadata)"
        )
    requested = _checkpoint_meta(names, seed, config, identity)
    if meta != requested:
        diffs = _manifest_differences(meta, requested)
        raise ValueError(
            f"{checkpoint} was written for a different run; differing keys: {diffs}"
        )
    tally = getattr(sampler, _TALLY_ATTR, None)
    if not isinstance(tally, _RunTally):
        raise RuntimeError(f"{checkpoint} carries no gwpop-search run tally")
    if int(sampler.ndim) != ndim or int(sampler.nlive) != config.nlive:
        raise ValueError(
            f"checkpoint has ndim={sampler.ndim}, nlive={sampler.nlive}; "
            f"requested ndim={ndim}, nlive={config.nlive}"
        )
    if int(sampler.queue_size) != config.batch_size:
        raise ValueError(
            f"checkpoint queue_size={sampler.queue_size} differs from "
            f"batch_size={config.batch_size}"
        )
    ptform = getattr(sampler, "prior_transform", None)
    if not hasattr(ptform, "func"):
        raise RuntimeError(f"{checkpoint}: unexpected prior-transform wrapper")
    stored = ptform.func
    if isinstance(stored, PriorTransform) and isinstance(prior_transform, PriorTransform):
        if stored != prior_transform:
            raise ValueError("checkpoint prior transform differs from the requested priors")
    else:
        ptform.func = prior_transform
    proxy.bind(pool)
    tally.bind(pool, clock_start)
    return sampler, tally


def _remaining_budget(sampler, config: DynestyConfig) -> tuple[int | None, int | None]:
    """Remaining (maxiter, maxcall) in dynesty's per-call semantics.

    dynesty compares ``maxiter``/``maxcall`` against counters that restart at
    zero on every ``run_nested`` call. An uninterrupted run with ``maxiter=M``
    performs ``M + 1`` iterations and stops once the main-loop likelihood
    calls exceed ``maxcall``; subtracting what the checkpoint already did
    reproduces exactly that stopping point after a resume.
    """
    done_iterations = int(sampler.it) - 1
    done_calls = int(np.sum(sampler.saved_run["nc"])) if len(sampler.saved_run["nc"]) else 0
    maxiter = None if config.maxiter is None else config.maxiter - done_iterations
    maxcall = None if config.maxcall is None else config.maxcall - done_calls
    return maxiter, maxcall


def _run_session(sampler, config: DynestyConfig, checkpoint: Path | None, *, resume: bool) -> None:
    maxiter, maxcall = _remaining_budget(sampler, config)
    sampler.run_nested(
        maxiter=maxiter,
        maxcall=maxcall,
        dlogz=config.dlogz,
        print_progress=False,
        save_bounds=False,
        checkpoint_file=None if checkpoint is None else str(checkpoint),
        checkpoint_every=config.checkpoint_every,
        resume=resume,
    )


def _queued_evaluations(sampler) -> int | None:
    """Likelihood evaluations of proposals still queued (not included in ``ncall``)."""
    try:
        return int(sum(len(item.evaluation_history) for item in sampler.queue))
    except (AttributeError, TypeError):  # pragma: no cover - other dynesty layouts
        return None


def _result_from_sampler(
    sampler,
    *,
    names: tuple[str, ...],
    seed: int,
    config: DynestyConfig,
    totals: Mapping[str, object],
    diagnostics: Mapping[str, object],
    provenance: Mapping[str, object],
) -> DynestyResult:
    import dynesty.utils as dyutils

    results = sampler.results
    ndim = len(names)
    samples = np.asarray(results["samples"], dtype=np.float64).reshape(-1, ndim)
    log_likelihoods = np.array(results["logl"], dtype=np.float64)
    log_weights = np.array(results["logwt"], dtype=np.float64)
    log_volumes = np.array(results["logvol"], dtype=np.float64)
    log_evidence = float(np.asarray(results["logz"])[-1])
    log_evidence_error = float(np.asarray(results["logzerr"])[-1])
    information = float(np.asarray(results["information"])[-1])
    if not np.isfinite(log_evidence):
        raise RuntimeError(f"dynesty returned a non-finite log evidence ({log_evidence})")
    lowl = float(getattr(dyutils, "_LOWL_VAL", -1e300))
    zero = log_likelihoods <= lowl
    log_likelihoods[zero] = -np.inf
    log_weights[zero] = -np.inf
    weights = _normalized_weights(log_weights)
    kish_ess = float(1.0 / np.sum(weights**2))
    posterior = _equal_weight_posterior(samples, log_weights, config.num_posterior_samples, seed)

    niter = int(results["niter"])
    info = dict(diagnostics)
    if 0 < niter < log_likelihoods.size:
        dead_logz = logsumexp(log_weights[:niter])
        final_delta = float(
            np.logaddexp(0.0, np.max(log_likelihoods[niter:]) + log_volumes[niter - 1] - dead_logz)
        )
        info["final_delta_logz"] = final_delta
        info["converged"] = bool(final_delta < config.dlogz)
    info["n_zero_likelihood_points"] = int(np.count_nonzero(zero))
    info["n_queued_evaluations"] = _queued_evaluations(sampler)
    info["cumulative"] = dict(totals)
    return DynestyResult(
        names=names,
        log_evidence=log_evidence,
        log_evidence_error=log_evidence_error,
        information=information,
        niter=niter,
        ncall=int(sampler.ncall),
        efficiency=float(sampler.eff),
        samples=samples,
        log_weights=log_weights,
        log_likelihoods=log_likelihoods,
        log_volumes=log_volumes,
        posterior_samples=posterior,
        kish_ess=kish_ess,
        elapsed_seconds=float(totals["elapsed_seconds"]),
        config=config,
        seed=int(seed),
        versions=_software_versions(),
        diagnostics=info,
        n_likelihood_evaluations=_optional_int(totals.get("n_evaluations")),
        n_selection_unsupported=_optional_int(totals.get("n_selection_unsupported")),
        provenance=copy.deepcopy(dict(provenance)),
    )


def posterior_taper_mass(
    results: Sequence[DynestyResult],
    loglike: BatchedShapeLogLikelihood,
    *,
    log_likelihood_atol: float = 1.0e-8,
    log_likelihood_rtol: float = 1.0e-8,
    log_likelihood_hard_rtol: float = 1.0e-4,
    max_support_mismatch_fraction: float = 1.0e-3,
) -> dict[str, object]:
    """Posterior fraction inside the variance-taper region, per run and pooled.

    ``sigma^2_lnL`` is re-evaluated with ``loglike`` (a tapered
    :class:`BatchedShapeLogLikelihood` of exactly the estimator the runs
    sampled; the likelihood identity is verified, ``selection_chunk_size``
    aside) at every weighted point of each run (dead points and final live
    points) and summarised with the run's importance weights by
    :func:`gwpop_search.hbi.taper.taper_region_summary`. The pooled block is
    the equal-weight mixture of the separately normalised runs.

    Reproduction of the sampled likelihood is a *recorded diagnostic*: the
    re-evaluation may run on another device or batch size than the sampling
    (e.g. sampled on the H100, summarised on the same process's backend,
    recorded as ``reevaluation_backend``), so rounding-level differences are
    expected. ``reproduces`` is ``|d ln L| <= atol + rtol |ln L|`` on every
    finite point and identical support; the largest deviation and the number
    of support mismatches are reported. Only a gross mismatch -- relative
    deviation above ``log_likelihood_hard_rtol`` or support differing on more
    than ``max_support_mismatch_fraction`` of the points, i.e. a different
    likelihood -- raises.
    """
    from gwpop_search.hbi.taper import taper_region_summary

    results = tuple(results)
    if not results:
        raise ValueError("at least one dynesty result is required")
    taper = getattr(loglike, "variance_taper", None)
    if taper is None:
        raise ValueError("posterior_taper_mass needs a likelihood with a variance taper")
    names = tuple(loglike.names)
    requested = _without_chunk_size(loglike.likelihood_identity())
    variances, weights, runs = [], [], []
    worst = 0.0
    reproduces = True
    support_mismatches = 0
    for index, result in enumerate(results):
        if tuple(result.names) != names:
            raise ValueError(f"run {index} has different parameter names than the likelihood")
        stored = result.likelihood_identity
        if stored is None:
            raise ValueError(f"run {index} carries no likelihood identity")
        diffs = _manifest_differences(_without_chunk_size(stored), requested)
        if diffs:
            raise ValueError(
                f"run {index} sampled a different likelihood than the one supplied; "
                f"differing keys: {diffs}"
            )
        comps = loglike.components(result.samples)
        stored_logl = np.asarray(result.log_likelihoods, dtype=float)
        finite = np.isfinite(stored_logl) & np.isfinite(comps["log_likelihood"])
        n_mismatch = int(np.sum(np.isfinite(stored_logl) != np.isfinite(comps["log_likelihood"])))
        support_mismatches += n_mismatch
        if n_mismatch:
            reproduces = False
            if n_mismatch > max_support_mismatch_fraction * max(stored_logl.size, 1):
                raise ValueError(
                    f"run {index}: re-evaluated likelihood support differs from the run at "
                    f"{n_mismatch} of {stored_logl.size} points"
                )
        if finite.any():
            new, old = comps["log_likelihood"][finite], stored_logl[finite]
            absdev = np.abs(new - old)
            dev = absdev / np.maximum(1.0, np.abs(old))
            run_worst = float(np.max(dev))
            worst = max(worst, run_worst)
            if np.any(absdev > log_likelihood_atol + log_likelihood_rtol * np.abs(old)):
                reproduces = False
            if run_worst > log_likelihood_hard_rtol:
                raise ValueError(
                    f"run {index}: re-evaluated tapered log-likelihood deviates from the "
                    f"sampled one by {run_worst:.3g} (relative): not the sampled likelihood"
                )
        w = result.weights
        summary = taper_region_summary(comps["variance"], w, taper)
        summary["repeat"] = index
        runs.append(summary)
        variances.append(comps["variance"])
        weights.append(w / len(results))
    pooled = taper_region_summary(np.concatenate(variances), np.concatenate(weights), taper)
    return {
        "taper": taper.to_dict(),
        "definition": (
            "posterior (importance-weighted dead + live points) fraction with "
            f"T(sigma^2) < {1.0 - taper.region_suppression:g}, i.e. sigma^2 > "
            f"{taper.region_onset:.6g}; sigma^2 = sum_i Var[ln I_i] + N^2 Var[xi]/xi^2"
        ),
        "runs": runs,
        "pooled": pooled,
        "max_relative_log_likelihood_mismatch": worst,
        "reproduces_sampled_log_likelihood": bool(reproduces),
        "reproduction_tolerance": {"atol": float(log_likelihood_atol), "rtol": float(log_likelihood_rtol),
                                   "hard_rtol": float(log_likelihood_hard_rtol)},
        "support_mismatches": int(support_mismatches),
        "reevaluation_backend": _jax_backend_name(),
    }


def _jax_backend_name() -> str | None:
    try:
        import jax

        return str(jax.default_backend())
    except Exception:  # pragma: no cover - JAX is a hard dependency of the tapered path
        return None


def _warn_selection_unsupported(result: DynestyResult, *, stacklevel: int = 3) -> None:
    n = result.n_selection_unsupported
    if n:
        warnings.warn(
            SelectionSupportWarning(
                f"{n} likelihood evaluation(s) of this run had population support on every "
                "event but none on the selection injections (log A = -inf). They were treated "
                "as zero likelihood, which removes that prior volume from the evidence; the "
                "NumPy reference raises SelectionSupportError there. Check the injection "
                "coverage before using this evidence."
            ),
            stacklevel=stacklevel,
        )


def run_dynesty(
    loglike_batched: Callable[[np.ndarray], np.ndarray],
    prior_transform: Callable[[np.ndarray], np.ndarray],
    ndim: int,
    *,
    seed: int,
    config: DynestyConfig | None = None,
    checkpoint_file: str | Path | None = None,
    resume: bool = False,
    names: Sequence[str] | None = None,
    identity: Mapping[str, object] | None = None,
) -> DynestyResult:
    """Run static dynesty with a batched likelihood and return posterior + evidence.

    ``loglike_batched(X [m, ndim]) -> [m]`` receives up to ``config.batch_size``
    rows per call. ``prior_transform(u [ndim]) -> theta [ndim]`` maps the unit
    cube to the prior and must be picklable when checkpointing (dynesty pickles
    the sampler). With ``resume=True`` the sampler is restored from
    ``checkpoint_file``; a checkpoint that already holds a finished run is
    converted to a result without further sampling. Starting a fresh run over
    an existing checkpoint is refused.

    Checkpoints pin the names, seed, :meth:`DynestyConfig.identity_dict`, the
    code identity, software versions, runtime (host, and the JAX device when
    the likelihood exposes ``runtime_identity()``), the likelihood identity
    (when it exposes ``likelihood_identity()``) and the optional caller
    ``identity`` (a JSON-serializable mapping, stored as
    ``result.provenance["run"]``); a resume refuses any difference. Warns with
    :class:`SelectionSupportWarning` when the run's total
    ``n_selection_unsupported`` is non-zero.
    """
    dynesty = _require_dynesty()
    cfg = DynestyConfig() if config is None else config
    if not isinstance(cfg, DynestyConfig):
        raise TypeError("config must be a DynestyConfig")
    ndim = _as_int("ndim", ndim, minimum=1)
    seed = _as_int("seed", seed, minimum=0)
    names = (
        tuple(f"x{k}" for k in range(ndim))
        if names is None
        else _validated_names(names, what="parameter names")
    )
    if len(names) != ndim:
        raise ValueError(f"{len(names)} names given for ndim={ndim}")
    if not callable(prior_transform):
        raise TypeError("prior_transform must be callable")
    if identity is not None and not isinstance(identity, Mapping):
        raise TypeError("identity must be a mapping")
    checkpoint = None if checkpoint_file is None else Path(checkpoint_file)
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"cannot resume: checkpoint {checkpoint} does not exist")
    elif checkpoint is not None:
        if checkpoint.exists():
            raise FileExistsError(
                f"checkpoint {checkpoint} exists; pass resume=True or remove it explicitly"
            )
        try:
            pickle.dumps(prior_transform)
        except Exception as exc:  # noqa: BLE001 - report the concrete pickling failure
            raise TypeError(
                "prior_transform must be picklable when checkpointing (dynesty pickles "
                "the sampler); use prior_transform_for() or a top-level callable"
            ) from exc
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    run_identity = _run_identity(loglike_batched, identity)

    start = time.perf_counter()
    with ThreadBatchPool(loglike_batched, cfg.batch_size) as pool:
        if resume:
            sampler, tally = _restore_sampler(
                dynesty,
                checkpoint,
                pool,
                prior_transform,
                ndim,
                cfg,
                seed=seed,
                names=names,
                identity=run_identity,
                clock_start=start,
            )
        else:
            tally = _RunTally()
            tally.bind(pool, start)
            sampler = _new_sampler(
                dynesty,
                pool,
                prior_transform,
                ndim,
                seed=seed,
                config=cfg,
                names=names,
                identity=run_identity,
                tally=tally,
            )
        finished_in_checkpoint = bool(sampler.added_live)
        if not finished_in_checkpoint:
            _run_session(sampler, cfg, checkpoint, resume=resume)
        pool_stats = pool.stats()
        totals = tally.snapshot()
    session_elapsed = time.perf_counter() - start

    diagnostics: dict[str, object] = {
        "resumed": bool(resume),
        "finished_in_checkpoint": finished_in_checkpoint,
        "session_elapsed_seconds": session_elapsed,
        "session_pool": pool_stats,
    }
    likelihood_stats = getattr(loglike_batched, "stats", None)
    if callable(likelihood_stats):
        diagnostics["session_likelihood"] = likelihood_stats()
    result = _result_from_sampler(
        sampler,
        names=names,
        seed=seed,
        config=cfg,
        totals=totals,
        diagnostics=diagnostics,
        provenance=run_identity,
    )
    _warn_selection_unsupported(result)
    return result


# ---------------------------------------------------------------------------
# Population runs with manifests
# ---------------------------------------------------------------------------


def _require_declared_priors(population_model, priors: Mapping[str, PriorSpec]) -> None:
    """For a declarative model the evidence is defined by the spec's own hyperpriors."""
    from gwpop_search.grammar import ModelSpec

    from .model_spec import prior_specs_from_model_spec

    spec = getattr(population_model, "spec", None)
    if not isinstance(spec, ModelSpec):
        return
    declared = prior_specs_from_model_spec(spec)
    missing = sorted(set(declared) - set(priors))
    extra = sorted(set(priors) - set(declared))
    changed = sorted(name for name in set(declared) & set(priors) if priors[name] != declared[name])
    if not (missing or extra or changed):
        return
    details = []
    if missing:
        details.append(f"missing {missing}")
    if extra:
        details.append(f"not parameters of the model {extra}")
    details.extend(
        f"{name}: {priors[name].to_dict()} instead of {declared[name].to_dict()}"
        for name in changed
    )
    raise ValueError(
        "priors must be the hyperpriors declared by the model spec "
        f"(model_hash {spec.model_hash}), which define its evidence and its hash: "
        + "; ".join(details)
        + ". Use prior_specs_from_model_spec(model.spec)."
    )


def build_dynesty_manifest(
    posterior,
    selection,
    population_model,
    priors: Mapping[str, PriorSpec],
    *,
    seed: int,
    config: DynestyConfig,
    hbi_config=None,
) -> dict[str, object]:
    """Resume identity of a population run (JSON-normalized).

    Pins the likelihood (:func:`build_likelihood_identity`: parameter names,
    HBI configuration, model configuration and the data: event names,
    per-event sample counts, bases, selection mode/campaigns and SHA-256
    digests of every array the likelihood reads), the priors, the dynesty
    trajectory configuration (:meth:`DynestyConfig.identity_dict`), the seed,
    the code identity (package version, git commit, ``git_dirty`` and a
    SHA-256 of the package sources), the software versions and the runtime
    (host and JAX device). For a model with a ``spec`` the priors must equal
    ``prior_specs_from_model_spec(model.spec)`` (``ValueError`` otherwise).
    """
    from .numpyro import _validated_priors

    if not isinstance(config, DynestyConfig):
        raise TypeError("config must be a DynestyConfig")
    prior_map = _validated_priors(priors)
    _require_declared_priors(population_model, prior_map)
    names, _ = prior_transform_for(prior_map)
    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "root_seed": _as_int("seed", seed, minimum=0),
        "code": _code_identity(),
        "versions": _software_versions(),
        "runtime": _runtime_identity(jax_devices=True),
        "dynesty_config": config.identity_dict(),
        "priors": serialize_prior_map(prior_map),
        **build_likelihood_identity(posterior, selection, population_model, names, hbi_config),
    }
    return _json_normalized(manifest)


def _manifest_differences(existing: Mapping, requested: Mapping, prefix: str = "") -> list[str]:
    keys = sorted(set(existing) | set(requested))
    diffs = []
    for key in keys:
        a, b = existing.get(key, "<missing>"), requested.get(key, "<missing>")
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            diffs.extend(_manifest_differences(a, b, prefix=f"{prefix}{key}."))
        elif a != b:
            diffs.append(f"{prefix}{key}")
    return diffs


def run_dynesty_population(
    posterior,
    selection,
    population_model,
    priors: Mapping[str, PriorSpec],
    *,
    seed: int,
    config: DynestyConfig | None = None,
    hbi_config=None,
    run_dir: str | Path,
) -> DynestyResult:
    """Resumable dynesty run of the standardized HBI shape likelihood.

    ``run_dir`` holds ``manifest.json``, dynesty's ``checkpoint.pkl`` and the
    final ``result.npz`` (+ ``result.npz.json``). If a complete result exists
    for an identical manifest it is loaded (with its equal-weight draws redone
    for a different ``num_posterior_samples``; the stored files are never
    rewritten, and the returned ``config`` is the requested one); if a
    checkpoint exists the run is resumed; otherwise a new run starts. A
    manifest that differs from the requested run raises ``ValueError``
    (nothing is overwritten). For a model with a ``spec`` the priors must be
    ``prior_specs_from_model_spec(model.spec)``.

    Warns with :class:`DirtyCodeWarning` when the package sources have
    uncommitted changes (the manifest then identifies the code by its source
    digest) and with :class:`SelectionSupportWarning` when the result has
    evaluations without selection support.
    """
    cfg = DynestyConfig() if config is None else config
    run_dir = Path(run_dir)
    manifest = build_dynesty_manifest(
        posterior,
        selection,
        population_model,
        priors,
        seed=seed,
        config=cfg,
        hbi_config=hbi_config,
    )
    manifest_sha256 = _json_sha256(manifest)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            diffs = _manifest_differences(existing, manifest)
            raise ValueError(
                f"{manifest_path} does not match the requested run; differing keys: {diffs}"
            )
    else:
        _atomic_write_text(manifest_path, json.dumps(manifest, sort_keys=True, indent=2))
    code = manifest["code"]
    if code["git_dirty"]:
        warnings.warn(
            DirtyCodeWarning(
                "gwpop_search sources have uncommitted changes: git commit "
                f"{code['git_commit']} does not contain the code of this run, which is "
                f"identified by source_sha256 {code['source_sha256']}. Commit before "
                "production runs."
            ),
            stacklevel=2,
        )

    names, transform = prior_transform_for(priors)
    result_path = run_dir / "result.npz"
    if dynesty_result_exists(result_path):
        result = load_dynesty_result(result_path)
        stored_run = result.provenance.get("run") or {}
        if (
            result.names != names
            or result.seed != int(seed)
            or result.config.identity_dict() != cfg.identity_dict()
            or stored_run.get("manifest_sha256") != manifest_sha256
        ):
            raise ValueError(f"{result_path} is inconsistent with {manifest_path}")
        if result.config.num_posterior_samples != cfg.num_posterior_samples:
            result = result.with_num_posterior_samples(cfg.num_posterior_samples)
        result = replace(result, config=cfg)
        _warn_selection_unsupported(result)
        return result

    checkpoint = run_dir / "checkpoint.pkl"
    loglike = build_batched_log_likelihood(
        posterior,
        selection,
        population_model,
        names,
        hbi_config=hbi_config,
        batch_size=cfg.batch_size,
    )
    result = run_dynesty(
        loglike,
        transform,
        len(names),
        seed=seed,
        config=cfg,
        checkpoint_file=checkpoint,
        resume=checkpoint.exists(),
        names=names,
        identity={"manifest_sha256": manifest_sha256},
    )
    save_dynesty_result(result_path, result)
    return result


# ---------------------------------------------------------------------------
# JAX importance diagnostics
# ---------------------------------------------------------------------------


def _log_weight_stats(values, valid, axis, jnp):
    """(max, sum exp(v - max), sum exp(2 (v - max)), n_zero) over finite entries."""
    finite = valid & jnp.isfinite(values)
    n_zero = jnp.sum(valid & (values == -jnp.inf), axis=axis, dtype=jnp.int64)
    peak = jnp.max(jnp.where(finite, values, -jnp.inf), axis=axis)
    has = jnp.isfinite(peak)
    shift = jnp.expand_dims(jnp.where(has, peak, 0.0), axis)
    scaled = jnp.where(finite, jnp.exp(values - shift), 0.0)
    return peak, jnp.sum(scaled, axis=axis), jnp.sum(scaled * scaled, axis=axis), n_zero


def _merge_log_weight_stats(a, b, jnp):
    peak = jnp.maximum(a[0], b[0])
    shift = jnp.where(jnp.isfinite(peak), peak, 0.0)
    fa = jnp.where(jnp.isfinite(a[0]), jnp.exp(a[0] - shift), 0.0)
    fb = jnp.where(jnp.isfinite(b[0]), jnp.exp(b[0] - shift), 0.0)
    return (
        peak,
        a[1] * fa + b[1] * fb,
        a[2] * fa * fa + b[2] * fb * fb,
        a[3] + b[3],
    )


def _importance_from_stats(stats, n_draw, jnp):
    """NumPy-reference importance diagnostics from (max, S1, S2, n_zero).

    With ``S1 = sum exp(v - max)`` and ``S2 = sum exp(2 (v - max))``:
    ``logsumexp = max + log S1``, ``ESS = S1^2 / S2``, the maximum normalized
    weight is ``1 / S1`` and ``Var[log I] = max(1/ESS - 1/n_draw, 0)``. No
    finite weight gives ESS = 0, max weight = 0 and infinite variance, as in
    :func:`gwpop_search.hbi.common.importance_diagnostics`.
    """
    peak, s1, s2, n_zero = stats
    has = jnp.isfinite(peak) & (s1 > 0)
    safe_s1 = jnp.where(has, s1, 1.0)
    safe_s2 = jnp.where(has, s2, 1.0)
    lse = jnp.where(has, peak + jnp.log(safe_s1), -jnp.inf)
    inv_ess = safe_s2 / (safe_s1 * safe_s1)
    ess = jnp.where(has, 1.0 / inv_ess, 0.0)
    max_weight = jnp.where(has, 1.0 / safe_s1, 0.0)
    variance = jnp.where(has, jnp.maximum(inv_ess - 1.0 / n_draw, 0.0), jnp.inf)
    return lse, ess, max_weight, variance, n_zero


@dataclass(frozen=True, eq=False)
class ImportanceDiagnosticsBatch:
    """Importance-sampling diagnostics for ``K`` hyperparameter vectors.

    Matches :func:`gwpop_search.hbi.evaluate_catalog_terms`: per-event ESS,
    maximum normalized weight, ``Var[log ell_i] ~= 1/ESS - 1/n_i`` and number
    of zero-support samples; combined selection ESS (over the effective
    exposure weights, fraction relative to all generated draws) and maximum
    weight; per-campaign exposures/diagnostics; ``Var[log A]`` combined from
    campaigns with squared exposure fractions (NaN when a campaign variance is
    infinite, as in the reference); and
    ``Var[log L] ~= sum_i Var[log ell_i] + N^2 Var[log A]``.

    Variance taper: ``taper_variance`` is the ``sigma^2`` the likelihood's
    taper sees (as ``shape_log_likelihood_variance``, except that a campaign
    without population support contributes zero instead of making the
    selection variance NaN; the two agree whenever every campaign has
    support), ``log_taper`` is ``ln T(taper_variance)`` (0 without a taper),
    ``log_likelihood`` is the likelihood that was sampled (tapered when the
    HBI configuration has a taper) and ``log_likelihood_untapered`` is
    ``ln L`` without it.
    """

    names: tuple[str, ...]
    event_names: tuple[str, ...]
    campaign_ids: tuple[str, ...]
    hyperparameters: np.ndarray
    log_likelihood: np.ndarray
    event_log_likelihoods: np.ndarray
    event_ess: np.ndarray
    event_ess_fraction: np.ndarray
    event_max_weight: np.ndarray
    event_variance: np.ndarray
    event_n_zero_weight: np.ndarray
    log_exposure: np.ndarray
    selection_ess: np.ndarray
    selection_ess_fraction: np.ndarray
    selection_max_weight: np.ndarray
    selection_n_zero_weight: np.ndarray
    campaign_log_exposure: np.ndarray
    campaign_ess: np.ndarray
    campaign_max_weight: np.ndarray
    campaign_variance: np.ndarray
    selection_variance: np.ndarray
    event_variance_total: np.ndarray
    shape_log_likelihood_variance: np.ndarray
    log_likelihood_untapered: np.ndarray | None = None
    taper_variance: np.ndarray | None = None
    log_taper: np.ndarray | None = None

    @property
    def n_events(self) -> int:
        return len(self.event_names)

    def summary_statistics(self) -> dict[str, np.ndarray]:
        """Per-vector scalar summaries (worst event, selection, variance budget)."""
        return {
            "log_likelihood": self.log_likelihood,
            "shape_log_likelihood_variance": self.shape_log_likelihood_variance,
            "event_variance_total": self.event_variance_total,
            "selection_variance_term": self.n_events**2 * self.selection_variance,
            "min_event_ess": np.min(self.event_ess, axis=1),
            "max_event_max_weight": np.max(self.event_max_weight, axis=1),
            "selection_ess": self.selection_ess,
            "selection_ess_fraction": self.selection_ess_fraction,
            "selection_max_weight": self.selection_max_weight,
            **(
                {}
                if self.taper_variance is None
                else {
                    "log_likelihood_untapered": self.log_likelihood_untapered,
                    "taper_variance": self.taper_variance,
                    "log_taper": self.log_taper,
                }
            ),
        }


class ImportanceDiagnosticsFunction:
    """Jitted, vmapped importance diagnostics; ``f(X [K, ndim]) -> ImportanceDiagnosticsBatch``.

    The reference NumPy implementation needs seconds per hyperparameter
    vector on the ~1M-row GWTC-5 candidate selection; this evaluates a fixed
    ``[batch_size, ndim]`` block per device call. ``selection_chunk_size``
    bounds device memory by scanning each campaign in chunks with an online
    (max, S1, S2) merge (results agree with the unchunked evaluation up to
    rounding); when omitted it defaults to ``hbi_config.selection_chunk_size``,
    and ``None`` there evaluates each campaign in one block.
    :meth:`likelihood_identity` identifies the estimator it diagnoses (see
    :func:`build_likelihood_identity`).
    """

    def __init__(
        self,
        posterior,
        selection,
        population_model,
        names: Sequence[str],
        *,
        hbi_config=None,
        batch_size: int = 16,
        selection_chunk_size: int | None = None,
    ):
        jax, jnp = _require_jax()
        from jax import lax

        from gwpop_search.data import validate_pair
        from gwpop_search.hbi import HBIConfig, RateTreatment
        from gwpop_search.hbi.common import density_required_fields, selection_log_factors
        from gwpop_search.hbi.jax_backend import _pad_events

        cfg = HBIConfig() if hbi_config is None else hbi_config
        if cfg.rate_treatment is not RateTreatment.SHAPE:
            raise ValueError("importance diagnostics are implemented for the shape likelihood only")
        self.names = _validated_names(names, what="hyperparameter names")
        self.ndim = len(self.names)
        self.batch_size = _as_int("batch_size", batch_size, minimum=1)
        self.hbi_config = cfg
        self._identity_inputs = (posterior, selection, population_model)
        self._likelihood_identity: dict | None = None
        fields = density_required_fields(population_model, posterior.basis)
        validate_pair(posterior, selection, fields)
        self.event_names = tuple(posterior.event_names)
        n_events = posterior.n_events
        pe = _pad_events(posterior, fields)
        counts = jnp.asarray(np.diff(posterior.offsets).astype(np.float64))

        factors = selection_log_factors(
            selection, use_observing_time=cfg.raw_selection_use_observing_time
        )
        if selection_chunk_size is None:
            selection_chunk_size = cfg.selection_chunk_size
        if selection_chunk_size is not None:
            selection_chunk_size = _as_int("selection_chunk_size", selection_chunk_size, minimum=1)
        groups = []
        campaign_ids = []
        for campaign in selection.campaigns:
            rows = selection.rows_for_campaign(campaign.campaign_id)
            if rows.size == 0:
                continue  # the NumPy reference skips empty campaigns
            factor = float(factors[rows[0]])
            if not np.all(factors[rows] == factor):  # pragma: no cover - by construction
                raise ValueError("selection log factors must be constant within a campaign")
            n = int(rows.size)
            size = n if selection_chunk_size is None else min(selection_chunk_size, n)
            n_chunks = -(-n // size)
            total = n_chunks * size
            mask = np.arange(total) < n

            def padded(values, rows=rows, total=total, n=n, n_chunks=n_chunks, size=size):
                values = np.asarray(values, dtype=np.float64)[rows]
                out = np.empty(total, dtype=np.float64)
                out[:n] = values
                out[n:] = values[0]
                return jnp.asarray(out.reshape(n_chunks, size))

            groups.append(
                {
                    "samples": {name: padded(selection.samples[name]) for name in fields},
                    "log_draw": padded(selection.log_draw_density),
                    "mask": jnp.asarray(mask.reshape(n_chunks, size)),
                    "factor": factor,
                    "n_draw": float(n if campaign.n_draw is None else campaign.n_draw),
                }
            )
            campaign_ids.append(campaign.campaign_id)
        if not groups:  # pragma: no cover - SelectionCatalog requires rows
            raise ValueError("selection has no retained rows")
        self.campaign_ids = tuple(campaign_ids)
        total_draw = float(
            sum(int(c.n_draw) for c in selection.campaigns)
            if all(c.n_draw is not None for c in selection.campaigns)
            else selection.n_selected
        )
        names_ = self.names
        taper = cfg.variance_taper

        def single(x):
            hyperparameters = {name: x[k] for k, name in enumerate(names_)}
            # Events.
            log_pop = population_model(pe.samples, hyperparameters)
            raw = log_pop - pe.log_ref_density
            invalid = jnp.any(pe.mask & (jnp.isnan(raw) | (raw == jnp.inf)))
            event_stats = _log_weight_stats(raw, pe.mask, 1, jnp)
            lse, ess, max_w, var, n_zero = _importance_from_stats(event_stats, counts, jnp)
            event_log_like = lse - jnp.log(counts)
            out = {
                "event_log_likelihoods": event_log_like,
                "event_ess": ess,
                "event_ess_fraction": ess / counts,
                "event_max_weight": max_w,
                "event_variance": var,
                "event_n_zero_weight": n_zero,
            }
            # Selection, one campaign at a time.
            campaign_stats = []
            for group in groups:

                def body(carry, xs):
                    chunk, log_draw, mask = xs
                    base = population_model(chunk, hyperparameters) - log_draw
                    bad = jnp.any(mask & (jnp.isnan(base) | (base == jnp.inf)))
                    stats = _log_weight_stats(base, mask, 0, jnp)
                    return (_merge_log_weight_stats(carry[0], stats, jnp), carry[1] | bad), None

                init = (
                    (
                        jnp.asarray(-jnp.inf, dtype=jnp.float64),
                        jnp.asarray(0.0, dtype=jnp.float64),
                        jnp.asarray(0.0, dtype=jnp.float64),
                        jnp.asarray(0, dtype=jnp.int64),
                    ),
                    jnp.asarray(False),
                )
                (stats, bad), _ = lax.scan(
                    body, init, (group["samples"], group["log_draw"], group["mask"])
                )
                invalid = invalid | bad
                campaign_stats.append(stats)

            combined = None
            c_logexp, c_ess, c_maxw, c_var = [], [], [], []
            for group, stats in zip(groups, campaign_stats):
                lse_k, ess_k, maxw_k, var_k, _ = _importance_from_stats(
                    stats, group["n_draw"], jnp
                )
                c_logexp.append(lse_k + group["factor"])
                c_ess.append(ess_k)
                c_maxw.append(maxw_k)
                c_var.append(var_k)
                effective = (stats[0] + group["factor"], stats[1], stats[2], stats[3])
                combined = (
                    effective
                    if combined is None
                    else _merge_log_weight_stats(combined, effective, jnp)
                )
            log_exposure, sel_ess, sel_maxw, _, sel_n_zero = _importance_from_stats(
                combined, total_draw, jnp
            )
            c_logexp = jnp.stack(c_logexp)
            c_var = jnp.stack(c_var)
            fractions = jnp.exp(c_logexp - log_exposure)
            sel_var = jnp.where(
                jnp.all(jnp.isfinite(c_var)), jnp.sum(fractions**2 * c_var), jnp.nan
            )
            event_var_total = jnp.sum(var)
            events_ok = jnp.all(jnp.isfinite(event_log_like))
            log_like = jnp.where(
                events_ok & jnp.isfinite(log_exposure),
                jnp.sum(jnp.where(jnp.isfinite(event_log_like), event_log_like, 0.0))
                - n_events * jnp.where(jnp.isfinite(log_exposure), log_exposure, 0.0),
                -jnp.inf,
            )
            supported = jnp.isfinite(c_logexp)
            taper_sel_var = jnp.sum(
                jnp.where(supported, fractions**2 * jnp.where(supported, c_var, 0.0), 0.0)
            )
            taper_variance = event_var_total + n_events**2 * taper_sel_var
            taper_variance = jnp.where(jnp.isnan(taper_variance), jnp.inf, taper_variance)
            if taper is None:
                log_t = jnp.zeros_like(taper_variance)
                tapered = log_like
            else:
                log_t = taper.log_taper_jax(taper_variance)
                tapered = jnp.where(jnp.isfinite(log_like), log_like + log_t, -jnp.inf)
            out.update(
                {
                    "log_exposure": log_exposure,
                    "selection_ess": sel_ess,
                    "selection_ess_fraction": sel_ess / total_draw,
                    "selection_max_weight": sel_maxw,
                    "selection_n_zero_weight": sel_n_zero,
                    "campaign_log_exposure": c_logexp,
                    "campaign_ess": jnp.stack(c_ess),
                    "campaign_max_weight": jnp.stack(c_maxw),
                    "campaign_variance": c_var,
                    "selection_variance": sel_var,
                    "event_variance_total": event_var_total,
                    "shape_log_likelihood_variance": event_var_total + n_events**2 * sel_var,
                    "log_likelihood": tapered,
                    "log_likelihood_untapered": log_like,
                    "taper_variance": taper_variance,
                    "log_taper": log_t,
                    "invalid": invalid,
                }
            )
            return out

        self._device_fn = jax.jit(jax.vmap(single))
        self._jax = jax
        self._jnp = jnp

    def likelihood_identity(self) -> dict[str, object]:
        """:func:`build_likelihood_identity` of the diagnosed likelihood (computed once)."""
        if self._likelihood_identity is None:
            posterior, selection, model = self._identity_inputs
            self._likelihood_identity = build_likelihood_identity(
                posterior, selection, model, self.names, self.hbi_config
            )
        return copy.deepcopy(self._likelihood_identity)

    def __call__(self, X) -> ImportanceDiagnosticsBatch:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.ndim or X.shape[0] == 0:
            raise ValueError(
                f"hyperparameter block must have shape [K>0, {self.ndim}]; got {X.shape}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("non-finite hyperparameters passed to importance diagnostics")
        m = X.shape[0]
        padded, n_blocks = _pad_rows(X, self.batch_size)
        chunks = []
        for block in range(n_blocks):
            sl = slice(block * self.batch_size, (block + 1) * self.batch_size)
            chunks.append(self._jax.device_get(self._device_fn(self._jnp.asarray(padded[sl]))))
        out = {key: np.concatenate([np.asarray(c[key]) for c in chunks])[:m] for key in chunks[0]}
        invalid = out.pop("invalid").astype(bool)
        if self.hbi_config.variance_taper is None:
            # Untapered estimator: the batch keeps its pre-taper shape.
            for key in ("log_likelihood_untapered", "taper_variance", "log_taper"):
                out.pop(key)
        if invalid.any():
            rows = np.flatnonzero(invalid)
            raise PopulationDensityError(
                "population density returned NaN/+inf importance weights for "
                + _describe_rows(self.names, X, rows)
            )
        return ImportanceDiagnosticsBatch(
            names=self.names,
            event_names=self.event_names,
            campaign_ids=self.campaign_ids,
            hyperparameters=X.copy(),
            **{key: np.asarray(value) for key, value in out.items()},
        )


def build_importance_diagnostics(
    posterior,
    selection,
    population_model,
    names: Sequence[str],
    *,
    hbi_config=None,
    batch_size: int = 16,
    selection_chunk_size: int | None = None,
) -> ImportanceDiagnosticsFunction:
    """Build the jitted diagnostics function (see :class:`ImportanceDiagnosticsFunction`)."""
    return ImportanceDiagnosticsFunction(
        posterior,
        selection,
        population_model,
        names,
        hbi_config=hbi_config,
        batch_size=batch_size,
        selection_chunk_size=selection_chunk_size,
    )


@dataclass(frozen=True, eq=False)
class PosteriorImportanceSummary:
    """Importance diagnostics at the posterior median and over posterior draws.

    ``over_posterior[statistic][f"q{q}"]`` is the empirical ``q`` quantile
    (``numpy.quantile(method="inverted_cdf")``) of ``statistic`` over the
    ``n_draws`` equal-weight posterior draws. Large values are bad for
    variances and maximum weights (read upper quantiles); small values are bad
    for ESS (read lower quantiles).
    """

    names: tuple[str, ...]
    median_hyperparameters: Mapping[str, float]
    at_median: Mapping[str, object]
    quantiles: tuple[float, ...]
    over_posterior: Mapping[str, Mapping[str, float]]
    n_draws: int
    batch: ImportanceDiagnosticsBatch

    def to_dict(self) -> dict[str, object]:
        return _json_ready(
            {
                "names": list(self.names),
                "median_hyperparameters": dict(self.median_hyperparameters),
                "at_median": dict(self.at_median),
                "quantiles": list(self.quantiles),
                "over_posterior": {k: dict(v) for k, v in self.over_posterior.items()},
                "n_draws": int(self.n_draws),
            }
        )


def importance_diagnostics_over_posterior(
    result: DynestyResult,
    posterior,
    selection,
    population_model,
    *,
    hbi_config=None,
    n_draws: int = 1000,
    quantiles: Sequence[float] = DEFAULT_DIAGNOSTIC_QUANTILES,
    seed: int = 0,
    batch_size: int = 16,
    selection_chunk_size: int | None = None,
    diagnostics_fn: ImportanceDiagnosticsFunction | None = None,
) -> PosteriorImportanceSummary:
    """Importance diagnostics at the posterior median and over ``n_draws`` posterior draws.

    The median is the per-parameter median of ``result.posterior_samples``;
    the draws are a seeded random subset (without replacement) of the
    equal-weight samples.

    The diagnostics must describe the estimator the run sampled. When the
    result carries its likelihood identity (every HBI result does),
    ``hbi_config`` defaults to the run's configuration and any difference in
    parameter names, HBI configuration (other than ``selection_chunk_size``,
    which only changes the summation order), model or data between the run
    and the requested diagnostics raises ``ValueError``. A result without a
    likelihood identity (a non-HBI likelihood or an older result format)
    requires an explicit ``hbi_config`` or ``diagnostics_fn``.
    """
    n_draws = _as_int("n_draws", n_draws, minimum=1)
    quantiles = tuple(float(q) for q in quantiles)
    if not quantiles or any(not 0.0 <= q <= 1.0 for q in quantiles):
        raise ValueError("quantiles must lie in [0, 1]")
    stored = result.likelihood_identity
    if stored is None:
        if hbi_config is None and diagnostics_fn is None:
            raise ValueError(
                "the result carries no likelihood identity (non-HBI likelihood or an older "
                "result format); pass the run's hbi_config explicitly"
            )
    elif hbi_config is None:
        hbi_config = _hbi_config_from_dict(stored["hbi_config"])
    fn = diagnostics_fn
    if fn is None:
        fn = build_importance_diagnostics(
            posterior,
            selection,
            population_model,
            result.names,
            hbi_config=hbi_config,
            batch_size=batch_size,
            selection_chunk_size=selection_chunk_size,
        )
    elif fn.names != tuple(result.names):
        raise ValueError("diagnostics_fn parameter names differ from the result")
    if stored is not None:
        diffs = _manifest_differences(
            _without_chunk_size(stored), _without_chunk_size(fn.likelihood_identity())
        )
        if diffs:
            raise ValueError(
                "the requested diagnostics evaluate a different likelihood than the one the "
                f"run sampled; differing keys: {diffs}"
            )
    draws = np.asarray(result.posterior_samples, dtype=np.float64)
    rng = np.random.default_rng(_as_int("seed", seed, minimum=0))
    k = min(n_draws, draws.shape[0])
    chosen = draws[np.sort(rng.choice(draws.shape[0], size=k, replace=False))]
    median = np.median(draws, axis=0)
    batch = fn(np.vstack([median[None, :], chosen]))
    stats = batch.summary_statistics()
    at_median: dict[str, object] = {key: float(value[0]) for key, value in stats.items()}
    at_median["worst_event"] = batch.event_names[int(np.argmin(batch.event_ess[0]))]
    over = {
        key: {
            f"q{q:g}": float(np.quantile(value[1:], q, method="inverted_cdf"))
            for q in quantiles
        }
        for key, value in stats.items()
    }
    return PosteriorImportanceSummary(
        names=tuple(result.names),
        median_hyperparameters={name: float(median[i]) for i, name in enumerate(result.names)},
        at_median=at_median,
        quantiles=quantiles,
        over_posterior=over,
        n_draws=k,
        batch=batch,
    )
