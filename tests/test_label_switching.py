"""Evidence-preserving canonical labels for exchangeable mixture components."""

from dataclasses import replace
import json
import pickle

import numpy as np
import pytest
from scipy.special import logsumexp

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

dynesty = pytest.importorskip("dynesty")

from gwpop_search.data.fixtures import (  # noqa: E402
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.grammar import (  # noqa: E402
    PriorConfig,
    baseline_model_spec,
    enumerate_model_graph,
)
from gwpop_search.hbi import HBIConfig  # noqa: E402
from gwpop_search.inference import (  # noqa: E402
    DynestyConfig,
    PriorSpec,
    build_batched_log_likelihood,
    prior_specs_from_model_spec,
    prior_transform_for,
    run_hbi_evidence,
)
from gwpop_search.inference.label_switching import (  # noqa: E402
    IDENTITY,
    ORDERED_PAIRS_PARAMETERIZATION,
    ExchangeableComponents,
    ModelParameterization,
    OrderedPairPriorTransform,
    canonicalize_samples,
    exchangeable_components_for_spec,
    parameterization_for_spec,
)
from gwpop_search.inference.synthetic import (  # noqa: E402
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.models import compile_model_spec  # noqa: E402

# ---------------------------------------------------------------------------
# A toy exchangeable two-component mixture likelihood (3 parameters)
# ---------------------------------------------------------------------------

SIGMA = 0.25
_rng = np.random.default_rng(1)
_component = _rng.random(24) < 0.35
DATA = np.where(_component, _rng.normal(-0.45, SIGMA, 24), _rng.normal(0.4, SIGMA, 24))
_C = -0.5 * np.log(2.0 * np.pi * SIGMA**2)
TOY_PRIORS = {
    "f": PriorSpec("uniform", low=0.0, high=1.0),
    "mu_1": PriorSpec("uniform", low=-1.0, high=1.0),
    "mu_2": PriorSpec("uniform", low=-1.0, high=1.0),
}
TOY_GROUP = ExchangeableComponents(
    order_by=("mu_1", "mu_2"), reflected=("f",), source="toy-mixture"
)


def toy_mixture_loglike(X):
    """ln L of the data for rows ``(f, mu_1, mu_2)``; weight ``f`` on component 2."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    f, m1, m2 = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    l1 = _C - 0.5 * ((DATA[None, :] - m1) / SIGMA) ** 2
    l2 = _C - 0.5 * ((DATA[None, :] - m2) / SIGMA) ** 2
    with np.errstate(divide="ignore"):
        return np.sum(np.logaddexp(np.log1p(-f) + l1, np.log(f) + l2), axis=1)


def _scalar_loglike(x):
    return float(toy_mixture_loglike(x)[0])


def _unit_cube_log_evidence(transform, grid=81):
    """Midpoint rule for ln int_[0,1]^3 L(T(u)) du (converged to ~1e-8 at 81^3)."""
    u = (np.arange(grid) + 0.5) / grid
    cube = np.stack(np.meshgrid(u, u, u, indexing="ij"), axis=-1).reshape(-1, 3)
    return float(logsumexp(toy_mixture_loglike(transform(cube))) - np.log(cube.shape[0]))


def _theta_space_log_evidence(grid=81):
    """The same evidence integrated directly over the prior box (independent route)."""
    f = (np.arange(grid) + 0.5) / grid
    m = -1.0 + 2.0 * (np.arange(grid) + 0.5) / grid
    X = np.stack(np.meshgrid(f, m, m, indexing="ij"), axis=-1).reshape(-1, 3)
    cell = np.log(1.0 / grid) + 2.0 * np.log(2.0 / grid)
    return float(logsumexp(toy_mixture_loglike(X)) + cell + np.log(0.25))


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------


def test_ordered_transform_is_a_bijection_onto_the_ordered_half_space():
    names, base = prior_transform_for(TOY_PRIORS)
    ordered = OrderedPairPriorTransform(base, [("mu_1", "mu_2")])
    assert ordered.names == names == ("f", "mu_1", "mu_2")

    u = np.random.default_rng(3).random((200_000, 3))
    theta = ordered(u)
    assert np.all(theta[:, 1] <= theta[:, 2])
    np.testing.assert_array_equal(theta[:, 0], base(u)[:, 0])  # untouched coordinate
    # Density 2 pi(mu_1) pi(mu_2) on the half-space: the ordered pair is distributed
    # as the order statistics of two iid prior draws.
    iid = base(np.random.default_rng(4).random((200_000, 3)))
    order_stats = np.sort(iid[:, 1:], axis=1)
    for k in (0, 1):
        np.testing.assert_allclose(
            np.quantile(theta[:, 1 + k], [0.1, 0.25, 0.5, 0.75, 0.9]),
            np.quantile(order_stats[:, k], [0.1, 0.25, 0.5, 0.75, 0.9]),
            atol=0.01,
        )
    # One-to-one: the inverse map recovers u from the canonical parameters.
    v1 = (theta[:, 1] + 1.0) / 2.0
    v2 = (theta[:, 2] + 1.0) / 2.0
    u1 = 1.0 - (1.0 - v1) ** 2
    u2 = (v2 - v1) / (1.0 - v1)
    np.testing.assert_allclose(u1, u[:, 1], atol=1e-12)
    np.testing.assert_allclose(u2, u[:, 2], atol=1e-9)
    # Single rows, edges and pickling (dynesty checkpoints pickle the transform).
    np.testing.assert_array_equal(ordered(u[0]), theta[0])
    edges = ordered(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 1e-18, 0.0]]))
    np.testing.assert_allclose(edges[:, 1:], [[-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]])
    clone = pickle.loads(pickle.dumps(ordered))
    assert clone == ordered
    np.testing.assert_array_equal(clone(u[:10]), theta[:10])
    assert ordered.to_dict()["pairs"] == [["mu_1", "mu_2"]]


def test_ordered_transform_refuses_non_exchangeable_pairs():
    priors = dict(TOY_PRIORS, mu_2=PriorSpec("uniform", low=-1.0, high=1.5))
    _, base = prior_transform_for(priors)
    with pytest.raises(ValueError, match="different priors"):
        OrderedPairPriorTransform(base, [("mu_1", "mu_2")])
    _, base = prior_transform_for(TOY_PRIORS)
    with pytest.raises(ValueError, match="not a model parameter"):
        OrderedPairPriorTransform(base, [("mu_1", "nope")])
    with pytest.raises(ValueError, match="more than one pair"):
        OrderedPairPriorTransform(base, [("mu_1", "mu_2"), ("mu_2", "f")])
    lopsided = ExchangeableComponents(order_by=("mu_1", "mu_2"), reflected=("f",))
    problems = lopsided.exchangeability_violations(
        dict(TOY_PRIORS, f=PriorSpec("uniform", low=0.0, high=0.9))
    )
    assert problems and "not symmetric" in problems[0]
    parameterization = ModelParameterization(ORDERED_PAIRS_PARAMETERIZATION, (lopsided,))
    lopsided_priors = dict(TOY_PRIORS, f=PriorSpec("uniform", low=0.0, high=0.9))
    with pytest.raises(ValueError, match="would change the evidence"):
        parameterization.prior_transform(lopsided_priors)


def test_ordered_parameterization_preserves_the_evidence_exactly_in_quadrature():
    """int_[0,1]^d L(T_ordered(u)) du == int_[0,1]^d L(T_plain(u)) du == Z."""
    names, base = prior_transform_for(TOY_PRIORS)
    ordered = ModelParameterization(ORDERED_PAIRS_PARAMETERIZATION, (TOY_GROUP,))
    _, ordered_transform = ordered.prior_transform(TOY_PRIORS)
    truth = _theta_space_log_evidence()
    plain = _unit_cube_log_evidence(base)
    canonical = _unit_cube_log_evidence(ordered_transform)
    assert plain == pytest.approx(truth, abs=1e-6)
    assert canonical == pytest.approx(truth, abs=1e-6)


def test_dynesty_evidence_is_identical_within_sampling_error_and_posterior_is_canonical():
    names, base = prior_transform_for(TOY_PRIORS)
    ordered = OrderedPairPriorTransform(base, [("mu_1", "mu_2")])
    truth = _theta_space_log_evidence()
    estimates, errors, posteriors = {}, {}, {}
    for label, transform in (("plain", base), ("ordered", ordered)):
        for seed in (11, 13):
            sampler = dynesty.NestedSampler(
                _scalar_loglike,
                transform,
                3,
                nlive=150,
                bound="multi",
                sample="rslice",
                slices=12,
                rstate=np.random.default_rng(seed),
            )
            sampler.run_nested(dlogz=0.1, print_progress=False)
            result = sampler.results
            estimates.setdefault(label, []).append(float(result["logz"][-1]))
            errors.setdefault(label, []).append(float(result["logzerr"][-1]))
            weights = np.exp(result["logwt"] - result["logz"][-1])
            posteriors.setdefault(label, []).append((np.asarray(result["samples"]), weights))
    mean = {label: np.mean(values) for label, values in estimates.items()}
    se = {label: np.sqrt(np.mean(np.square(errors[label])) / 2.0) for label in errors}
    assert abs(mean["ordered"] - truth) < 4.0 * se["ordered"] + 0.05
    assert abs(mean["plain"] - truth) < 4.0 * se["plain"] + 0.05
    assert abs(mean["ordered"] - mean["plain"]) < 4.0 * np.hypot(se["ordered"], se["plain"])
    # Canonical labels: every ordered sample has mu_1 <= mu_2; the plain runs visit
    # both mirror modes, and relabelling them gives the same canonical posterior.
    for samples, _ in posteriors["ordered"]:
        assert np.all(samples[:, 1] <= samples[:, 2])
    plain_samples, plain_weights = posteriors["plain"][0]
    mirror = np.sum(plain_weights[plain_samples[:, 1] > plain_samples[:, 2]])
    assert 0.2 < mirror < 0.8
    relabelled = canonicalize_samples(plain_samples, names, [TOY_GROUP])
    ordered_samples, ordered_weights = posteriors["ordered"][0]
    for k in range(3):
        a = np.sum(relabelled[:, k] * plain_weights)
        b = np.sum(ordered_samples[:, k] * ordered_weights)
        assert a == pytest.approx(b, abs=0.05)


# ---------------------------------------------------------------------------
# The declarative chi_eff mixture
# ---------------------------------------------------------------------------


def _mixture_spec():
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=1)
    (spec,) = [node for node in graph.nodes if node.chieff.family == "gaussian_mixture"]
    return spec


def test_gaussian_mixture_is_detected_as_exchangeable_and_others_are_not():
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=1)
    canonical = [node for node in graph.nodes if exchangeable_components_for_spec(node)]
    assert [node.chieff.family for node in canonical] == ["gaussian_mixture"]
    spec = canonical[0]
    parameterization = parameterization_for_spec(spec)
    assert parameterization.kind == ORDERED_PAIRS_PARAMETERIZATION
    (group,) = parameterization.groups
    assert group.order_by == ("chi_mu_1", "chi_mu_2")
    assert group.swapped == (("chi_sigma_1", "chi_sigma_2"),)
    assert group.reflected == ("chi_fraction",)
    parameterization.validate(prior_specs_from_model_spec(spec))
    assert parameterization_for_spec(spec, canonicalize=False) is IDENTITY
    assert parameterization_for_spec(baseline_model_spec()) is IDENTITY
    payload = parameterization.to_dict()
    assert payload["kind"] == ORDERED_PAIRS_PARAMETERIZATION
    assert json.loads(json.dumps(payload)) == payload
    # A mixture whose component priors differ is not exchangeable: refused.
    priors = dict(spec.priors)
    priors["chi_mu_2"] = PriorConfig("uniform", {"low": -0.5, "high": 0.6})
    lopsided = replace(spec, priors=priors)
    with pytest.raises(ValueError, match="not exchangeable"):
        parameterization_for_spec(lopsided).validate(prior_specs_from_model_spec(lopsided))


def test_hbi_likelihood_of_the_mixture_is_invariant_under_the_label_swap():
    dataset = generate_baseline_synthetic_dataset(
        seed=5,
        config=SyntheticSurveyConfig(
            n_events=5,
            posterior_samples_per_event=24,
            n_injections=1_500,
            population_batch_size=512,
            redshift_sampling_grid=1024,
        ),
    )
    spec = _mixture_spec()
    priors = prior_specs_from_model_spec(spec)
    names, transform = prior_transform_for(priors)
    loglike = build_batched_log_likelihood(
        dataset.posterior,
        dataset.selection,
        compile_model_spec(spec),
        names,
        hbi_config=HBIConfig(selection_chunk_size=None),
        batch_size=32,
    )
    theta = transform(np.random.default_rng(8).random((64, len(names))))
    group = exchangeable_components_for_spec(spec)[0]
    index = {name: k for k, name in enumerate(names)}
    swapped = theta.copy()
    for a, b in (group.order_by, *group.swapped):
        swapped[:, [index[a], index[b]]] = theta[:, [index[b], index[a]]]
    swapped[:, index["chi_fraction"]] = 1.0 - theta[:, index["chi_fraction"]]
    original, mirrored = loglike(theta), loglike(swapped)
    finite = np.isfinite(original)
    assert finite.sum() > 10
    np.testing.assert_array_equal(np.isfinite(mirrored), finite)
    np.testing.assert_allclose(mirrored[finite], original[finite], rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# The reparameterized population run (HBI toy)
# ---------------------------------------------------------------------------

TOY_HBI_PRIORS = {
    "f": PriorSpec("uniform", low=0.0, high=1.0),
    "mu_1": PriorSpec("uniform", low=-0.6, high=0.6),
    "mu_2": PriorSpec("uniform", low=-0.6, high=0.6),
}


def toy_mixture_density(samples, hp):
    """Toy population: exchangeable two-Gaussian chi_eff mixture times fixed shapes."""
    chi = samples["chi_eff"]
    width = 0.3

    def component(mu):
        return -0.5 * ((chi - mu) / width) ** 2 - jnp.log(width * jnp.sqrt(2.0 * jnp.pi))

    f = hp["f"]
    mixture = jnp.logaddexp(
        jnp.log1p(-f) + component(hp["mu_1"]), jnp.log(f) + component(hp["mu_2"])
    )
    return mixture - 0.02 * samples["m1_source"] - 0.1 * samples["z"]


def test_reparameterized_population_run_is_canonical_resumable_and_pinned(tmp_path):
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    parameterization = ModelParameterization(ORDERED_PAIRS_PARAMETERIZATION, (TOY_GROUP,))
    cfg = DynestyConfig(
        nlive=40, bound="multi", sample="rslice", dlogz=0.5, batch_size=8, num_posterior_samples=200
    )

    def run(directory, **overrides):
        kwargs = dict(
            seed=3,
            config=cfg,
            hbi_config=HBIConfig(),
            run_dir=directory,
            parameterization=parameterization,
        )
        kwargs.update(overrides)
        return run_hbi_evidence(posterior, selection, toy_mixture_density, TOY_HBI_PRIORS, **kwargs)

    result = run(tmp_path / "ordered")
    assert result.names == ("f", "mu_1", "mu_2")
    assert np.all(result.posterior_samples[:, 1] <= result.posterior_samples[:, 2])
    assert np.all(result.samples[:, 1] <= result.samples[:, 2])
    assert np.isfinite(result.log_evidence)
    manifest = json.loads((tmp_path / "ordered" / "manifest.json").read_text())
    assert manifest["parameterization"]["kind"] == ORDERED_PAIRS_PARAMETERIZATION
    assert manifest["priors"] == {name: spec.to_dict() for name, spec in TOY_HBI_PRIORS.items()}
    assert result.provenance["run"]["manifest_sha256"]

    # Finished results are reused (identical), a different parameterization is refused.
    again = run(tmp_path / "ordered")
    np.testing.assert_array_equal(again.samples, result.samples)
    assert again.log_evidence == result.log_evidence
    with pytest.raises(ValueError, match="parameterization"):
        run(tmp_path / "ordered", parameterization=IDENTITY)
    with pytest.raises(ValueError, match="does not match"):
        run(tmp_path / "ordered", seed=4)

    # Same evidence as the identity parameterization, within sampling error.
    plain = run(tmp_path / "plain", parameterization=IDENTITY)
    assert not (tmp_path / "plain" / "manifest.json").read_text().count("parameterization")
    combined = np.hypot(plain.log_evidence_error, result.log_evidence_error)
    assert abs(plain.log_evidence - result.log_evidence) < 4.0 * combined
