"""Opt-in chi_eff follow-up models: Student-t, q-dependent mixture weight and
logistic-in-q width. Also regression-locks every existing frozen model hash and
the gwtc5-v1 enumeration, which the follow-up options must not change."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as st

from gwpop_search.cli import build_parser
from gwpop_search.grammar import (
    DEFAULT_COMPONENT_REGISTRY,
    DEFAULT_MUTATIONS,
    FOLLOWUP_MUTATION_TABLE,
    FOLLOWUP_MUTATIONS,
    BlockSpec,
    InapplicableMutation,
    ModelSpec,
    apply_mutation,
    baseline_model_spec,
    enumerate_model_graph,
    load_model_spec,
)
from gwpop_search.grammar.followup import (
    FOLLOWUP_MODEL_PATHS,
    FROZEN_GWTC5_ROOT_HASH,
    followup_model_specs,
    load_frozen_root,
    write_followup_specs,
)
from gwpop_search.inference import prior_specs_from_model_spec
from gwpop_search.inference.label_switching import (
    IDENTITY,
    ORDERED_PAIRS_PARAMETERIZATION,
    parameterization_for_spec,
)
from gwpop_search.models import (
    DeclarativeGwcatChiEffModel,
    chi_eff_logistic_mixture_logpdf,
    chi_eff_mixture_logpdf,
    chi_eff_student_t_logpdf,
    truncated_normal_logpdf,
    truncated_student_t_logpdf,
)
from gwpop_search.models.components import _log_student_t_interval_mass
from gwpop_search.models.declarative import chieff_logpdf_from_spec

REPO = Path(__file__).resolve().parents[1]
FROZEN_GRAPH = REPO / "tests" / "fixtures" / "frozen_gwtc5_bbh_v1a_model_graph.json"
FROZEN_GRAPH_ORIGINAL = Path(
    "/hildafs/projects/phy220048p/magana/gwpop-search-data/frozen/"
    "gwtc5-bbh-v1a/model_graph.json"
)
FOLLOWUP_SPEC_DIR = REPO / "followup_specs"

#: The 15 frozen gwtc5-bbh-v1a node hashes (root first), independent of the file.
FROZEN_HASHES = (
    "666c4ffc99f000e8c3783589866f0ab6bc120bf161fca5f47b5130983263ff67",
    "7a0d4c70f41125756312601fa4b8aedd596a7475d3849b7298ab684c9c708ffd",
    "190c9234543674d2c82209bbc91307bd284c5595cce7fc6b7559a052424f0088",
    "33715204a20b6ef5e11371c3b0be9a671535d81d28a2df744d27c380764d92ed",
    "243b4593705c7c9013c0d5c9b1e0875ef494244b780cbd88c202201b0e93dfc1",
    "467e4bf5859c10c6ca753e53fc2301d00ba2a305155aa19ead1521950f4a172e",
    "6f7fb264ddf35c37a7072c56471423b2b89b2f2ed5f989573bb527345a3a9127",
    "3e36f327c564a63994bc2a972a175b2ca1bf11ecc06b630009ddab849f679ccb",
    "6ef26a317a99482c68264638f3777b81033e251fc3b7b7ded697ee0de03f6ad3",
    "f5f5af3e9128ac5440a23dec9752054e5c86d9b56561ef1377650c33c294e870",
    "479f7064930ce97ffdd110fb6475d4b5cf4ddbbefa9a2ff3e8ec1373a69a5bd4",
    "cef0bafd3a7469cce31403a6b8e7b2cc767469b0f114941359155343b618c37f",
    "05dac13762cdd13d2a239a64cb6b46effeb305ff97c7883f969b4c10065e3de5",
    "78b6cf7e1e7a3803bd941631efb8b1f75358e38d1f9e3c708cfbae724a008625",
    "48e9e45a03c4334f6fb9a07d7de0438d1fbcaff1f4de4021ead81ac840274bb7",
)

FOLLOWUP_HASHES = {
    "chieff_student_t": (
        "5ba71a0ddf2a2f83aac1df628159a9eb51a7c22ea30f23075623c2729e5c9213"
    ),
    "chieff_mixture_logistic_q_fraction": (
        "39017291f77f89e14c162f7523e2e0b2feec42e05551acb56f5beb3c430f9b25"
    ),
    "chieff_width_logistic_q": (
        "2136018ba5c1ef75d50e23526f2ee33b8e4fd5c459c702fe7e706342e2935f57"
    ),
}


@pytest.fixture(autouse=True)
def _x64():
    previous = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _frozen_payload():
    return json.loads(FROZEN_GRAPH.read_text())


def _root() -> ModelSpec:
    return load_frozen_root(FROZEN_GRAPH)


def _specs() -> dict[str, ModelSpec]:
    return followup_model_specs(_root())


# ---------------------------------------------------------------------------
# Existing models are unchanged
# ---------------------------------------------------------------------------


def test_vendored_frozen_graph_matches_original_when_available():
    if not FROZEN_GRAPH_ORIGINAL.exists():
        pytest.skip("frozen data directory not mounted")
    digest = hashlib.sha256(FROZEN_GRAPH.read_bytes()).hexdigest()
    assert digest == hashlib.sha256(FROZEN_GRAPH_ORIGINAL.read_bytes()).hexdigest()


def test_frozen_node_hashes_are_unchanged_and_still_valid():
    payload = _frozen_payload()
    assert len(payload["nodes"]) == 15
    stored = tuple(node["model_hash"] for node in payload["nodes"])
    assert stored == FROZEN_HASHES
    for node in payload["nodes"]:
        spec = ModelSpec.from_dict(node["spec"])
        assert spec.model_hash == node["model_hash"]
        assert spec.canonical_dict() == node["spec"]
        DEFAULT_COMPONENT_REGISTRY.validate_model(spec)
        DeclarativeGwcatChiEffModel(spec)


def test_gwtc5_depth1_enumeration_is_unchanged():
    graph = enumerate_model_graph(baseline_model_spec("gwtc5-v1"), max_depth=1)
    assert graph.root_hash == FROZEN_GWTC5_ROOT_HASH
    assert graph.to_dict() == _frozen_payload()


def test_cli_enumerate_models_gwtc5_depth1_is_unchanged(tmp_path, capsys):
    out = tmp_path / "graph.json"
    args = build_parser().parse_args(
        [
            "enumerate-models",
            "--output",
            str(out),
            "--max-depth",
            "1",
            "--hyperprior-profile",
            "gwtc5-v1",
        ]
    )
    args.func(args)
    capsys.readouterr()
    assert json.loads(out.read_text()) == _frozen_payload()


def test_followup_atoms_are_not_default_atoms():
    default_ids = {item.mutation_id for item in DEFAULT_MUTATIONS}
    followup_ids = {item.mutation_id for item in FOLLOWUP_MUTATIONS}
    assert followup_ids == {
        "chieff.family.student_t",
        "chieff.fraction.logistic_q",
        "chieff.width.logistic_q",
    }
    assert not default_ids & followup_ids
    assert all(not item.companion_options for item in DEFAULT_MUTATIONS)


def test_default_depth2_enumeration_ignores_followup_options():
    # The existing depth-2 graphs only contain existing option values.
    for profile in ("phase3", "gwtc5-v1"):
        graph = enumerate_model_graph(baseline_model_spec(profile), max_depth=2)
        for model in graph.nodes:
            assert model.chieff.family in {"truncated_gaussian", "gaussian_mixture"}
            assert model.chieff.options.get("width_dependence") != "logistic_q"
            assert model.chieff.options.get("fraction_dependence", "constant") == "constant"
            assert "q_pivot" not in model.chieff.options or (
                model.chieff.family == "truncated_gaussian"
            )


# ---------------------------------------------------------------------------
# Specs, grammar and CLI validation
# ---------------------------------------------------------------------------


def test_followup_specs_hashes_and_priors():
    specs = _specs()
    assert set(specs) == set(FOLLOWUP_MODEL_PATHS)
    for name, spec in specs.items():
        assert spec.model_hash == FOLLOWUP_HASHES[name]
        assert spec.model_hash not in FROZEN_HASHES

    t = specs["chieff_student_t"]
    assert t.chieff.family == "truncated_student_t"
    p = t.priors
    assert p["chi_mu"].to_dict() == {
        "family": "uniform",
        "parameters": {"high": 0.3, "low": -0.3},
    }
    assert p["chi_sigma"].to_dict() == {
        "family": "log_uniform",
        "parameters": {"high": 0.5, "low": 0.03},
    }
    assert p["chi_nu"].to_dict() == {
        "family": "log_uniform",
        "parameters": {"high": 100.0, "low": 1.0},
    }

    mix = specs["chieff_mixture_logistic_q_fraction"]
    assert mix.chieff.options == {
        "components": 2,
        "fraction_dependence": "logistic_q",
        "q_pivot": 0.7,
    }
    assert isinstance(mix.chieff.options["q_pivot"], float)
    expected = {
        "chi_mu_1": ("uniform", -0.5, 0.5),
        "chi_mu_2": ("uniform", 0.0, 1.0),
        "chi_sigma_1": ("log_uniform", 0.02, 0.5),
        "chi_sigma_2": ("log_uniform", 0.02, 0.5),
        "logit_chi_fraction": ("uniform", -8.0, 2.0),
        "chi_fraction_q_slope": ("uniform", -10.0, 10.0),
    }
    for name, (family, low, high) in expected.items():
        assert mix.priors[name].family == family
        assert mix.priors[name].parameters == {"low": low, "high": high}
    assert "chi_fraction" not in mix.priors
    assert "chi_mu" not in mix.priors and "chi_sigma" not in mix.priors

    width = specs["chieff_width_logistic_q"]
    assert width.chieff.options["width_dependence"] == "logistic_q"
    expected = {
        "delta_log_chi_sigma": ("uniform", -1.0, 4.0),
        "q_transition": ("uniform", 0.1, 1.0),
        "q_transition_width": ("log_uniform", 0.01, 0.3),
    }
    for name, (family, low, high) in expected.items():
        assert width.priors[name].family == family
        assert width.priors[name].parameters == {"low": low, "high": high}

    # Root priors outside the chi_eff block are untouched.
    root = _root()
    for spec in specs.values():
        for name in ("alpha", "mmin", "mmax", "peak_fraction", "beta_q", "kappa"):
            assert spec.priors[name] == root.priors[name]


def test_committed_followup_spec_files_match_and_validate(tmp_path, capsys):
    specs = _specs()
    manifest = json.loads((FOLLOWUP_SPEC_DIR / "manifest.json").read_text())
    assert manifest["root_hash"] == FROZEN_GWTC5_ROOT_HASH
    for name, spec in specs.items():
        path = FOLLOWUP_SPEC_DIR / f"{name}.json"
        loaded = load_model_spec(path)
        assert loaded.model_hash == spec.model_hash
        assert manifest["models"][name]["model_hash"] == spec.model_hash
        args = build_parser().parse_args(["validate-model", "--spec", str(path)])
        args.func(args)
        assert spec.model_hash in capsys.readouterr().out

    written = write_followup_specs(_root(), tmp_path)
    assert written == manifest


def test_every_prior_is_a_used_hyperparameter():
    """The model consumes exactly the declared priors (no stale / missing ones)."""
    samples = _fake_samples(8)
    for spec in _specs().values():
        model = DeclarativeGwcatChiEffModel(spec)
        hp = _prior_centre(spec)
        assert np.all(np.isfinite(np.asarray(model(samples, hp))))
        reference = np.asarray(model(samples, hp))
        for name in spec.priors:
            prior = spec.priors[name]
            lo, hi = prior.parameters["low"], prior.parameters["high"]
            moved = dict(hp)
            if prior.family == "uniform":
                moved[name] = hp[name] + 0.05 * (hi - lo)
            else:
                moved[name] = hp[name] * 1.1
            changed = np.asarray(model(samples, moved))
            assert not np.allclose(changed, reference, rtol=1e-10, atol=0), name


def test_registry_and_mutation_guards():
    specs = _specs()
    mix = specs["chieff_mixture_logistic_q_fraction"]
    bad = ModelSpec.from_dict(mix.to_dict())
    options = dict(bad.chieff.options)
    del options["q_pivot"]
    bad = ModelSpec(
        mass=bad.mass,
        pairing=bad.pairing,
        chieff=BlockSpec("gaussian_mixture", options),
        redshift=bad.redshift,
        mixture=bad.mixture,
        priors=bad.priors,
    )
    with pytest.raises(ValueError, match="requires option"):
        DEFAULT_COMPONENT_REGISTRY.validate_model(bad)

    root = _root()
    with pytest.raises(InapplicableMutation):
        apply_mutation(root, FOLLOWUP_MUTATION_TABLE["chieff.fraction.logistic_q"])
    with pytest.raises(InapplicableMutation):
        apply_mutation(
            specs["chieff_student_t"], FOLLOWUP_MUTATION_TABLE["chieff.width.logistic_q"]
        )
    for option, value in (("mean_dependence", "linear_q"), ("width_dependence", "linear_q")):
        t = specs["chieff_student_t"]
        opts = dict(t.chieff.options)
        opts[option] = value
        with pytest.raises(ValueError):
            DEFAULT_COMPONENT_REGISTRY.definition("chieff", "truncated_student_t").validate(
                BlockSpec("truncated_student_t", opts)
            )


def test_width_logistic_removes_stale_width_slopes():
    table = {item.mutation_id: item for item in DEFAULT_MUTATIONS}
    parent = apply_mutation(_root(), table["chieff.width.linear_q"])
    child = apply_mutation(parent, FOLLOWUP_MUTATION_TABLE["chieff.width.logistic_q"])
    assert "log_chi_sigma_q_slope" not in child.priors
    assert child.chieff.options["width_dependence"] == "logistic_q"


def test_enumeration_with_followup_atoms_is_a_superset():
    root = baseline_model_spec("gwtc5-v1")
    base = enumerate_model_graph(root, max_depth=1)
    extended = enumerate_model_graph(
        root, mutations=DEFAULT_MUTATIONS + FOLLOWUP_MUTATIONS, max_depth=2, max_models=400
    )
    extended_hashes = {model.model_hash for model in extended.nodes}
    assert {model.model_hash for model in base.nodes} <= extended_hashes
    assert set(FOLLOWUP_HASHES.values()) <= extended_hashes


# ---------------------------------------------------------------------------
# Label switching
# ---------------------------------------------------------------------------


def test_logistic_mixture_uses_identity_parameterisation():
    specs = _specs()
    mix = specs["chieff_mixture_logistic_q_fraction"]
    parameterization = parameterization_for_spec(mix)
    assert parameterization == IDENTITY
    priors = prior_specs_from_model_spec(mix)
    parameterization.validate(priors)
    names, transform = parameterization.prior_transform(priors)
    u = np.random.default_rng(1).uniform(size=(64, len(names)))
    theta = transform(u)
    k1, k2 = names.index("chi_mu_1"), names.index("chi_mu_2")
    assert np.all((theta[:, k1] >= -0.5) & (theta[:, k1] <= 0.5))
    assert np.all((theta[:, k2] >= 0.0) & (theta[:, k2] <= 1.0))

    # The constant-fraction mixture keeps its canonicalisation.
    constant = ModelSpec.from_dict(_frozen_payload()["nodes"][1]["spec"])
    assert constant.chieff.family == "gaussian_mixture"
    assert parameterization_for_spec(constant).kind == ORDERED_PAIRS_PARAMETERIZATION
    for name in ("chieff_student_t", "chieff_width_logistic_q"):
        assert parameterization_for_spec(specs[name]) == IDENTITY


# ---------------------------------------------------------------------------
# Densities
# ---------------------------------------------------------------------------


def _student_t_reference_mass(mu, sigma, nu):
    a = (-1.0 - mu) / sigma
    b = (1.0 - mu) / sigma
    if a >= 0.0:
        return st.t.sf(a, nu) - st.t.sf(b, nu)
    return st.t.cdf(b, nu) - st.t.cdf(a, nu)


@pytest.mark.parametrize("nu", [1.0, 1.7, 4.0, 30.0, 100.0])
@pytest.mark.parametrize("sigma", [0.03, 0.1, 0.5])
@pytest.mark.parametrize("mu", [-0.3, 0.0, 0.17, 0.3])
def test_student_t_normalisation_matches_scipy(mu, sigma, nu):
    a = (-1.0 - mu) / sigma
    b = (1.0 - mu) / sigma
    mass = float(jnp.exp(_log_student_t_interval_mass(a, b, nu)))
    ref = _student_t_reference_mass(mu, sigma, nu)
    assert abs(mass / ref - 1.0) < 1e-10

    x = np.linspace(-1.0, 1.0, 41)
    ours = np.asarray(chi_eff_student_t_logpdf(x, mu=mu, sigma=sigma, nu=nu))
    ref_logpdf = st.t.logpdf(x, nu, loc=mu, scale=sigma) - np.log(ref)
    np.testing.assert_allclose(ours, ref_logpdf, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "mu,sigma,nu",
    [(1.5, 0.1, 3.0), (-1.5, 0.2, 100.0), (2.0, 0.5, 1.0), (1.2, 2.0, 10.0), (0.0, 50.0, 2.0)],
)
def test_student_t_normalisation_outside_prior(mu, sigma, nu):
    a = (-1.0 - mu) / sigma
    b = (1.0 - mu) / sigma
    mass = float(jnp.exp(_log_student_t_interval_mass(a, b, nu)))
    assert abs(mass / _student_t_reference_mass(mu, sigma, nu) - 1.0) < 1e-9


def test_student_t_invalid_hyperparameters_and_support():
    x = jnp.asarray([-1.5, 0.0, 1.5])
    out = np.asarray(chi_eff_student_t_logpdf(x, mu=0.0, sigma=0.1, nu=3.0))
    assert out[0] == -np.inf and out[2] == -np.inf and np.isfinite(out[1])
    for kwargs in ({"sigma": 0.0, "nu": 3.0}, {"sigma": 0.1, "nu": 0.0}, {"sigma": 0.1, "nu": np.nan}):
        out = np.asarray(chi_eff_student_t_logpdf(x, mu=0.0, **kwargs))
        assert np.all(out == -np.inf)


def _composite_gauss_legendre(n_intervals=800, order=16):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(-1.0, 1.0, n_intervals + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    x = (mid[:, None] + half[:, None] * nodes[None, :]).ravel()
    w = (half[:, None] * weights[None, :]).ravel()
    return x, w


def _draw_prior(spec, rng, n):
    draws = {}
    for name, prior in spec.priors.items():
        lo, hi = prior.parameters["low"], prior.parameters["high"]
        if prior.family == "uniform":
            draws[name] = rng.uniform(lo, hi, size=n)
        else:
            draws[name] = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
    return draws


def _prior_centre(spec):
    out = {}
    for name, prior in spec.priors.items():
        lo, hi = prior.parameters["low"], prior.parameters["high"]
        out[name] = 0.5 * (lo + hi) if prior.family == "uniform" else np.sqrt(lo * hi)
    return out


@pytest.mark.parametrize("name", sorted(FOLLOWUP_MODEL_PATHS))
def test_chieff_block_normalises_over_prior_draws(name):
    spec = _specs()[name]
    x, w = _composite_gauss_legendre()
    rng = np.random.default_rng(20260923)
    draws = _draw_prior(spec, rng, 40)
    # Include the prior corners that stress the quadrature / normalisation.
    if name == "chieff_student_t":
        for key, value in (("chi_nu", 1.0), ("chi_sigma", 0.03), ("chi_mu", 0.3)):
            draws[key][:4] = value
        draws["chi_mu"][4:8] = -0.3
        draws["chi_nu"][4:8] = 100.0

    def block(hp, q):
        return chieff_logpdf_from_spec(spec.chieff, x, 30.0, q, 0.3, hp)

    batched = jax.jit(jax.vmap(block, in_axes=(0, None)))
    hp = {k: jnp.asarray(v) for k, v in draws.items()}
    for q in (0.08, 0.35, 0.7, 1.0):
        logp = np.asarray(batched(hp, q))
        assert np.all(np.isfinite(logp))
        integral = np.exp(logp) @ w
        np.testing.assert_allclose(integral, 1.0, rtol=0, atol=2e-9)


def test_student_t_large_nu_approaches_truncated_gaussian():
    x = np.linspace(-1.0, 1.0, 801)
    mu, sigma = 0.05, 0.1
    gauss = np.asarray(truncated_normal_logpdf(x, mu=mu, sigma=sigma, low=-1.0, high=1.0))
    core = np.abs((x - mu) / sigma) < 4.0
    errors = []
    for nu in (1e4, 1e5, 1e6):
        t = np.asarray(
            truncated_student_t_logpdf(x, mu=mu, sigma=sigma, nu=nu, low=-1.0, high=1.0)
        )
        errors.append(np.max(np.abs(t - gauss)[core]))
    assert errors[-1] < 1e-4
    # O(1/nu) convergence.
    assert errors[0] / errors[1] == pytest.approx(10.0, rel=0.05)
    assert errors[1] / errors[2] == pytest.approx(10.0, rel=0.05)


def test_logistic_fraction_with_zero_slope_equals_constant_mixture():
    specs = _specs()
    mix = specs["chieff_mixture_logistic_q_fraction"]
    constant = ModelSpec.from_dict(_frozen_payload()["nodes"][1]["spec"])
    x = np.linspace(-1.0, 1.0, 201)
    q = np.linspace(0.05, 1.0, 201)
    for logit in (-8.0, -2.3, 0.0, 1.4, 2.0):
        base = {
            "chi_mu_1": 0.02,
            "chi_sigma_1": 0.07,
            "chi_mu_2": 0.35,
            "chi_sigma_2": 0.2,
        }
        new = chieff_logpdf_from_spec(
            mix.chieff,
            x,
            30.0,
            q,
            0.3,
            {**base, "logit_chi_fraction": logit, "chi_fraction_q_slope": 0.0},
        )
        old = chieff_logpdf_from_spec(
            constant.chieff,
            x,
            30.0,
            q,
            0.3,
            {**base, "chi_fraction": float(jax.nn.sigmoid(logit))},
        )
        np.testing.assert_allclose(np.asarray(new), np.asarray(old), rtol=1e-12, atol=1e-12)
        direct = chi_eff_mixture_logpdf(
            x, mu1=0.02, sigma1=0.07, mu2=0.35, sigma2=0.2, fraction=float(jax.nn.sigmoid(logit))
        )
        np.testing.assert_allclose(np.asarray(new), np.asarray(direct), rtol=1e-12, atol=1e-12)


def test_logistic_fraction_follows_q():
    mix = _specs()["chieff_mixture_logistic_q_fraction"]
    hp = {
        "chi_mu_1": 0.0,
        "chi_sigma_1": 0.05,
        "chi_mu_2": 0.6,
        "chi_sigma_2": 0.05,
        "logit_chi_fraction": -1.0,
        "chi_fraction_q_slope": -6.0,
    }
    for q in (0.1, 0.7, 0.95):
        f = float(jax.nn.sigmoid(-1.0 - 6.0 * (q - 0.7)))
        ours = chieff_logpdf_from_spec(mix.chieff, 0.6, 30.0, q, 0.3, hp)
        ref = chi_eff_logistic_mixture_logpdf(
            0.6, mu1=0.0, sigma1=0.05, mu2=0.6, sigma2=0.05, fraction_logit=np.log(f / (1 - f))
        )
        np.testing.assert_allclose(float(ours), float(ref), rtol=1e-12)
        direct = np.log(
            (1 - f) * st.truncnorm.pdf(0.6, -20.0, 20.0, loc=0.0, scale=0.05)
            + f * st.truncnorm.pdf(0.6, -32.0, 8.0, loc=0.6, scale=0.05)
        )
        np.testing.assert_allclose(float(ours), direct, rtol=1e-10)


def test_logistic_width_limits():
    width = _specs()["chieff_width_logistic_q"]
    root = _root()
    x = np.linspace(-1.0, 1.0, 101)
    q = np.linspace(0.05, 1.0, 101)
    base = {"chi_mu": 0.06, "chi_sigma": 0.08}
    new = chieff_logpdf_from_spec(
        width.chieff,
        x,
        30.0,
        q,
        0.3,
        {**base, "delta_log_chi_sigma": 0.0, "q_transition": 0.5, "q_transition_width": 0.05},
    )
    old = chieff_logpdf_from_spec(root.chieff, x, 30.0, q, 0.3, base)
    # Equal up to float64 rounding (a per-q width array vs a scalar width).
    np.testing.assert_allclose(np.asarray(new), np.asarray(old), rtol=1e-14, atol=0)

    # Far below / above the transition the width is chi_sigma*exp(delta) / chi_sigma.
    hp = {**base, "delta_log_chi_sigma": 1.5, "q_transition": 0.5, "q_transition_width": 0.01}
    low = chieff_logpdf_from_spec(width.chieff, x, 30.0, 0.1, 0.3, hp)
    high = chieff_logpdf_from_spec(width.chieff, x, 30.0, 0.95, 0.3, hp)
    np.testing.assert_allclose(
        np.asarray(low),
        np.asarray(truncated_normal_logpdf(x, mu=0.06, sigma=0.08 * np.exp(1.5), low=-1, high=1)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(high),
        np.asarray(truncated_normal_logpdf(x, mu=0.06, sigma=0.08, low=-1, high=1)),
        rtol=1e-12,
    )


def _fake_samples(n, seed=3):
    rng = np.random.default_rng(seed)
    return {
        "m1_detector": jnp.asarray(rng.uniform(30.0, 70.0, n)),
        "q": jnp.asarray(rng.uniform(0.4, 1.0, n)),
        "luminosity_distance": jnp.asarray(rng.uniform(300.0, 3000.0, n)),
        "ra": jnp.asarray(rng.uniform(0.0, 2.0 * np.pi, n)),
        "dec": jnp.asarray(rng.uniform(-1.0, 1.0, n)),
        "chi_eff": jnp.asarray(rng.uniform(-0.6, 0.8, n)),
    }


@pytest.mark.parametrize("name", sorted(FOLLOWUP_MODEL_PATHS))
def test_full_model_jit_vmap_matches_loop(name):
    spec = _specs()[name]
    model = DeclarativeGwcatChiEffModel(spec)
    samples = _fake_samples(64)
    draws = _draw_prior(spec, np.random.default_rng(7), 6)
    root_centre = _prior_centre(spec)
    for key in ("mmin", "mmax", "alpha", "peak_fraction", "peak_mu", "peak_sigma", "beta_q", "kappa"):
        draws[key] = np.full(6, root_centre[key])
    hp = {k: jnp.asarray(v) for k, v in draws.items()}
    batched = np.asarray(jax.jit(jax.vmap(lambda h: model(samples, h)))(hp))
    assert batched.shape == (6, 64)
    for i in range(6):
        single = np.asarray(model(samples, {k: v[i] for k, v in draws.items()}))
        np.testing.assert_allclose(batched[i], single, rtol=1e-12, atol=1e-12)
        assert np.all(np.isfinite(single[np.asarray(samples["m1_detector"]) > 0]) | (single == -np.inf))
    assert np.isfinite(batched).mean() > 0.5


def test_student_t_location_scale_gradients_are_finite():
    # dynesty is gradient-free, but masked branches must not poison gradients.
    def f(mu, sigma):
        return chi_eff_student_t_logpdf(0.1, mu=mu, sigma=sigma, nu=3.0)

    for mu, sigma in ((0.05, 0.1), (-0.3, 0.03), (0.3, 0.5), (1.5, 0.2)):
        grads = jax.grad(f, argnums=(0, 1))(mu, sigma)
        assert all(np.isfinite(float(g)) for g in grads)
        eps = 1e-6
        numeric = (float(f(mu + eps, sigma)) - float(f(mu - eps, sigma))) / (2 * eps)
        assert float(grads[0]) == pytest.approx(numeric, rel=1e-5, abs=1e-6)
