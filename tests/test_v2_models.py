"""Normalisation and support tests of every v2 model family."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gwpop_search.grammar import (  # noqa: E402
    DEFAULT_COMPONENT_REGISTRY,
    V2_ATOM_IDS,
    V2_MUTATION_TABLE,
    apply_mutation,
    baseline_model_spec,
    enumerate_v2_depth1,
    v2_root_model_spec,
)
from gwpop_search.grammar.v2_structure import required_hyperparameters  # noqa: E402
from gwpop_search.models import (  # noqa: E402
    FlatLambdaCDM,
    compile_model_spec,
    v2_mass_logpdf,
    v2_physical_hyperparameters,
)
from gwpop_search.models.components import (  # noqa: E402
    BrokenPowerLawPeaksMass,
    LVKMassGrid,
    PowerLawRedshiftNormTable,
    TaperedPowerLawPairing,
    log_eps_skewnorm,
    redshift_madau_dickinson_psi_logpdf,
    redshift_powerlaw_conditional_logpdf,
    redshift_rate_logpdf,
    truncated_normal_logpdf,
)
from gwpop_search.models.declarative import lvk_default_coordinates  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or np.trapz

#: A GWTC-5-like draw of the root (sampled coordinates).
LVK_LIKE = lvk_default_coordinates({
    "alpha_1": 1.5, "alpha_2": 5.4, "break_mass": 37.5, "mlow_1": 4.5, "delta_m_1": 3.5,
    "mlow_2": 3.5, "delta_m_2": 4.8, "lam_0": 0.40, "lam_1": 0.55, "mpp_1": 9.9, "sigpp_1": 0.8,
    "mpp_2": 32.3, "sigpp_2": 5.7, "beta": 1.04, "lamb": 2.5,
})


def atom(model, aid):
    return apply_mutation(model, V2_MUTATION_TABLE[V2_ATOM_IDS[aid]])


def prior_centre(spec):
    out = {}
    for name, prior in spec.priors.items():
        lo, hi = prior.parameters["low"], prior.parameters["high"]
        out[name] = math.sqrt(lo * hi) if prior.family == "log_uniform" else 0.5 * (lo + hi)
    return out


def hyper(spec, **overrides):
    hp = prior_centre(spec)
    hp.update({k: v for k, v in LVK_LIKE.items() if k in spec.priors})
    hp.update({"chi_mu": 0.05, "chi_log_sigma": math.log(0.12), "kappa": 2.5})
    hp = {k: v for k, v in hp.items() if k in spec.priors}
    hp.update(overrides)
    assert set(hp) == set(spec.priors)
    return hp


def fine_m1(lo=3.0, hi=300.0, n=40001):
    return np.geomspace(lo, hi, n)


# ---------------------------------------------------------------------------
# mass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("aid", [None, "M1", "M2", "M3", "M4", "M5"])
def test_primary_mass_normalises_for_every_mass_family(aid):
    root = v2_root_model_spec()
    spec = root if aid is None else atom(root, aid)
    model = compile_model_spec(spec)
    hp = v2_physical_hyperparameters(spec, hyper(spec))
    mass = model._v2.mass
    # exact on the model's own normalisation nodes
    g = model._v2.grid
    on_nodes = np.exp(np.asarray(mass.log_prob(g.m1s, g.log_m1s, hp)))
    assert abs(float(np.sum(on_nodes * np.asarray(g.w_m1))) - 1.0) < 1e-12
    # and to the trapezoid accuracy of the 1000-node grid on a fine grid
    m = fine_m1()
    p = np.exp(np.asarray(mass.log_prob(jnp.asarray(m), jnp.log(jnp.asarray(m)), hp)))
    assert abs(_trapz(p, m) - 1.0) < 2e-4
    assert np.all(p[m < hp["mlow_1"]] == 0.0)
    # Dirichlet weights sum to one
    total = sum(float(hp[f"lam_{c}"]) for c in mass.components)
    assert abs(total - 1.0) < 1e-14


def test_dirichlet_unit_coordinates_have_the_dirichlet_marginals():
    rng = np.random.default_rng(3)
    u = rng.uniform(size=(200000, 3))
    e = -np.log1p(-u)
    lam = e / e.sum(axis=1, keepdims=True)
    # Dirichlet(1, 1, 1): each weight ~ Beta(1, 2) (mean 1/3, var 1/18)
    assert np.allclose(lam.mean(axis=0), 1.0 / 3.0, atol=3e-3)
    assert np.allclose(lam.var(axis=0), 1.0 / 18.0, atol=2e-3)
    # the remaining two, renormalised, are uniform whatever the first is
    r = lam[:, 1] / (lam[:, 1] + lam[:, 2])
    assert abs(np.mean(r) - 0.5) < 3e-3 and abs(np.var(r) - 1.0 / 12.0) < 2e-3
    assert abs(np.corrcoef(r, lam[:, 0])[0, 1]) < 0.01


def test_mlow_2_fraction_is_uniform_between_mmin_and_mlow_1():
    spec = v2_root_model_spec()
    hp = hyper(spec, mlow_1=7.0, mlow_2_frac=0.25)
    assert v2_physical_hyperparameters(spec, hp)["mlow_2"] == pytest.approx(3.0 + 0.25 * 4.0)


@pytest.mark.parametrize("form", ["constant", "logistic_log_m1", "per_mass_component"])
@pytest.mark.parametrize("q_floor", [0.001, 0.05])
def test_pairing_conditional_normalises(form, q_floor):
    grid = LVKMassGrid(mmin=3.0, mmax=300.0, n_m1=1000, n_q=500, q_floor=q_floor)
    mass = BrokenPowerLawPeaksMass("broken", ("p10", "p35"), grid)
    pairing = TaperedPowerLawPairing(form, grid, mass.components)
    spec = v2_root_model_spec()
    hp = v2_physical_hyperparameters(spec, hyper(spec))
    hp.update(beta_low=-1.0, beta_high=4.0, beta_m_t=25.0, beta_width=0.3,
              beta_pl=0.5, beta_p10=3.0, beta_p35=-1.5)
    # at the normalisation nodes the conditional integrates to one exactly
    idx = np.array([150, 300, 500, 700, 999])
    m1_nodes = np.asarray(grid.m1s)[idx]
    q = np.asarray(grid.qs)
    mm, qq = np.meshgrid(m1_nodes, q)
    terms = mass.log_component_terms(jnp.asarray(mm), jnp.log(jnp.asarray(mm)), hp)
    lp = pairing.log_prob(jnp.asarray(qq), jnp.log(jnp.asarray(qq)), jnp.asarray(mm), jnp.log(jnp.asarray(mm)),
                          hp, mass_terms=terms)
    integral = np.sum(np.exp(np.asarray(lp)) * np.asarray(grid.w_q)[:, None], axis=0)
    assert np.allclose(integral, 1.0, atol=1e-10), integral
    # between nodes (Z_q interpolated linearly in ln m1) to interpolation accuracy
    m1_off = np.array([8.0, 20.0, 47.3, 150.0])
    qf = np.linspace(q_floor, 1.0, 40001)
    mm, qq = np.meshgrid(m1_off, qf)
    terms = mass.log_component_terms(jnp.asarray(mm), jnp.log(jnp.asarray(mm)), hp)
    lp = pairing.log_prob(jnp.asarray(qq), jnp.log(jnp.asarray(qq)), jnp.asarray(mm), jnp.log(jnp.asarray(mm)),
                          hp, mass_terms=terms)
    integral = _trapz(np.exp(np.asarray(lp)), qf, axis=0)
    assert np.allclose(integral, 1.0, atol=2e-4), integral
    # support: q >= max(q_floor, mlow_2 / m1)
    assert np.all(np.isneginf(np.asarray(lp))[(qq < q_floor) | (qq * mm < hp["mlow_2"])])


def test_joint_mass_density_normalises():
    spec = v2_root_model_spec()
    model = compile_model_spec(spec)
    hp = hyper(spec)
    m = np.geomspace(3.0, 300.0, 3001)
    q = np.linspace(0.05, 1.0, 2001)
    mm, qq = np.meshgrid(m, q)
    p = np.exp(np.asarray(v2_mass_logpdf(model, jnp.asarray(mm), jnp.asarray(qq), hp)))
    assert abs(_trapz(_trapz(p, q, axis=0), m) - 1.0) < 1e-3


def test_per_component_pairing_keeps_the_primary_marginal():
    root = v2_root_model_spec()
    spec = atom(root, "P2")
    model = compile_model_spec(spec)
    base = compile_model_spec(root)
    hp = hyper(spec, beta_pl=0.5, beta_p10=4.0, beta_p35=-1.0)
    g = model._v2.grid
    # at the normalisation nodes the q-marginal is exactly the primary density
    m, q, w_q = np.asarray(g.m1s), np.asarray(g.qs), np.asarray(g.w_q)
    mm, qq = np.meshgrid(m, q)
    marg = np.sum(np.exp(np.asarray(v2_mass_logpdf(model, jnp.asarray(mm), jnp.asarray(qq), hp))) * w_q[:, None],
                  axis=0)
    phys = v2_physical_hyperparameters(root, {**hyper(root), **{k: hp[k] for k in hp if k in root.priors}})
    p_m1 = np.exp(np.asarray(base._v2.mass.log_prob(jnp.asarray(m), jnp.log(jnp.asarray(m)), phys)))
    ok = p_m1 > 1e-8 * p_m1.max()
    assert np.max(np.abs(marg[ok] / p_m1[ok] - 1.0)) < 1e-10


# ---------------------------------------------------------------------------
# redshift
# ---------------------------------------------------------------------------


def test_redshift_families_normalise_on_the_support_zmax():
    cosmo = FlatLambdaCDM()
    z = np.linspace(0.0, 1.9, 200001)
    for kappa in (-8.0, 0.0, 2.5, 9.0):
        p = np.exp(np.asarray(redshift_rate_logpdf(jnp.asarray(z), kappa=kappa, zmax=1.9, cosmology=cosmo)))
        assert abs(_trapz(p, z) - 1.0) < 1e-8
    for gamma, kappa, zp in ((2.7, 5.6, 1.9), (-3.0, 0.5, 0.2), (8.0, 9.5, 3.9)):
        p = np.exp(np.asarray(redshift_madau_dickinson_psi_logpdf(
            jnp.asarray(z), gamma=gamma, kappa=kappa, z_peak=zp, zmax=1.9, cosmology=cosmo)))
        assert abs(_trapz(p, z) - 1.0) < 1e-8
    assert np.isneginf(np.asarray(redshift_rate_logpdf(jnp.asarray([1.95]), kappa=2.0, zmax=1.9,
                                                       cosmology=cosmo)))[0]


def test_madau_dickinson_with_zero_decline_is_the_power_law():
    cosmo = FlatLambdaCDM()
    z = jnp.linspace(1e-4, 1.9, 101)
    a = redshift_madau_dickinson_psi_logpdf(z, gamma=2.3, kappa=0.0, z_peak=1.7, zmax=1.9, cosmology=cosmo)
    b = redshift_rate_logpdf(z, kappa=2.3, zmax=1.9, cosmology=cosmo)
    assert np.max(np.abs(np.asarray(a) - np.asarray(b))) < 1e-12


def test_kappa_table_matches_direct_quadrature_and_normalises():
    cosmo = FlatLambdaCDM()
    table = PowerLawRedshiftNormTable(cosmo, 1.9)
    z = np.linspace(0.0, 1.9, 100001)
    kappas = np.array([-31.7, -10.0, -0.013, 0.0, 2.54, 7.777, 19.2, 39.99])
    for kappa in kappas:
        direct = np.asarray(redshift_rate_logpdf(jnp.asarray(z), kappa=kappa, zmax=1.9, cosmology=cosmo))
        tab = np.asarray(redshift_powerlaw_conditional_logpdf(jnp.asarray(z), jnp.asarray(kappa), zmax=1.9,
                                                              cosmology=cosmo, norm_table=table))
        assert np.array_equal(np.isfinite(tab), np.isfinite(direct))
        fin = np.isfinite(direct)
        assert fin[1:].all() and np.max(np.abs(tab[fin] - direct[fin])) < 1e-10
        assert abs(_trapz(np.exp(tab), z) - 1.0) < 1e-6  # trapz of the peaked extreme-kappa densities
    out = np.asarray(redshift_powerlaw_conditional_logpdf(jnp.asarray([0.5]), jnp.asarray(41.0), zmax=1.9,
                                                          cosmology=cosmo, norm_table=table))
    assert np.isneginf(out[0])


def test_kappa_of_m1_conditional_redshift_normalises_per_m1():
    spec = atom(v2_root_model_spec(), "Z2")
    model = compile_model_spec(spec)
    from gwpop_search.models.declarative import _v2_redshift_logpdf

    hp = v2_physical_hyperparameters(spec, hyper(spec, kappa_log_m1_slope=3.0))
    z = np.linspace(0.0, 1.9, 50001)
    for m1 in (4.0, 30.0, 250.0):
        lp = _v2_redshift_logpdf(model, jnp.asarray(z), jnp.log(jnp.full_like(jnp.asarray(z), m1)), hp)
        assert abs(_trapz(np.exp(np.asarray(lp)), z) - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# chi_eff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("aid", [None, "S1", "S2", "S3", "S4"])
@pytest.mark.parametrize("corr", [None, "C1", "C2", "C3", "C4", "C5", "C6"])
def test_chi_eff_families_normalise_with_every_correlation(aid, corr):
    from gwpop_search.models.declarative import _v2_chieff_logpdf

    spec = v2_root_model_spec()
    if corr is not None:
        spec = atom(spec, corr)
    if aid is not None:
        spec = apply_mutation(spec, V2_MUTATION_TABLE[V2_ATOM_IDS[aid]])
    overrides = {k: v for k, v in dict(
        chi_mu_q_slope=-0.4, chi_log_sigma_q_slope=-2.2, chi_mu_z_slope=0.3, chi_log_sigma_z_slope=1.0,
        chi_mu_log_m1_slope=0.2, chi_log_sigma_log_m1_slope=0.7, chi_fraction=0.2, chi_mu_2_frac=0.4,
        chi_log_sigma_2=-1.2, chi_fraction_low=0.05, chi_fraction_high=0.6, chi_fraction_m_t=40.0,
        chi_fraction_width=0.2, chi_eps=0.4, chi_nu=3.0).items() if k in spec.priors}
    hp = v2_physical_hyperparameters(spec, hyper(spec, **overrides))
    chi = np.linspace(-1.0, 1.0, 40001)
    for q, z, m1 in ((1.0, 0.0, 30.0), (0.3, 1.2, 8.0), (0.8, 0.4, 90.0)):
        lp = _v2_chieff_logpdf(spec, jnp.asarray(chi), hp, q=jnp.full(chi.shape, q), z=jnp.full(chi.shape, z),
                               log_m1=jnp.full(chi.shape, math.log(m1)))
        p = np.exp(np.asarray(lp))
        assert abs(_trapz(p, chi) - 1.0) < 2e-6, (aid, corr, q, z, m1, _trapz(p, chi))
    outside = _v2_chieff_logpdf(spec, jnp.asarray([-1.01, 1.01]), hp, q=jnp.ones(2), z=jnp.zeros(2),
                                log_m1=jnp.full(2, math.log(30.0)))
    assert np.all(np.isneginf(np.asarray(outside)))


def test_skew_normal_at_zero_skew_is_the_truncated_gaussian_even_off_the_interval():
    chi = jnp.linspace(-1.0, 1.0, 1001)
    for mu, sigma in ((0.05, 0.12), (0.99, 0.0068), (1.6, 0.05), (-2.5, 0.3)):
        a = np.asarray(log_eps_skewnorm(chi, mu, sigma, 0.0))
        b = np.asarray(truncated_normal_logpdf(chi, mu=mu, sigma=sigma, low=-1.0, high=1.0))
        finite = np.isfinite(b)
        assert np.array_equal(np.isfinite(a), finite)
        assert np.max(np.abs(a[finite] - b[finite])) < 1e-9


def test_mixture_component_means_are_ordered():
    spec = atom(v2_root_model_spec(), "S1")
    for mu, frac in ((-0.9, 0.0), (0.1, 0.5), (0.7, 1.0)):
        hp = v2_physical_hyperparameters(spec, hyper(spec, chi_mu=mu, chi_mu_2_frac=frac))
        assert mu <= float(hp["chi_mu_2"]) <= 1.0


# ---------------------------------------------------------------------------
# compiled model: support from the spec, Jacobian, required fields
# ---------------------------------------------------------------------------


def test_support_comes_from_the_spec_with_no_hidden_defaults():
    root = v2_root_model_spec()
    model = compile_model_spec(root)
    assert model.zmax == 1.9 and model.q_floor == 0.05
    assert model.cosmology.H0 == 67.74 and model.cosmology.Om0 == 0.3089
    assert model.required_fields == ("m1_detector", "q", "luminosity_distance", "chi_eff")
    assert model.to_config()["support"] == root.support
    with pytest.raises(ValueError, match="disagrees with the v2 model support"):
        compile_model_spec(root, zmax=2.5)
    with pytest.raises(ValueError, match="disagrees with the v2 model support"):
        compile_model_spec(root, q_floor=0.01)
    compile_model_spec(root, zmax=1.9)  # an agreeing value is accepted
    for key in root.support:
        support = {k: v for k, v in root.support.items() if k != key}
        with pytest.raises(ValueError, match="missing"):
            DEFAULT_COMPONENT_REGISTRY.validate_model(replace(root, support=support))
    iso = compile_model_spec(replace(root, support={**root.support, "sky": "isotropic"}))
    assert "ra" in iso.required_fields and "dec" in iso.required_fields


def test_v1_models_are_unchanged_and_reject_a_support_block():
    v1 = baseline_model_spec("gwtc5-v1")
    model = compile_model_spec(v1)
    assert model.zmax == 2.5 and model.q_floor == 0.05 and model.redshift_quadrature_order == 96
    assert "support" not in v1.to_dict()
    with pytest.raises(ValueError, match="only defined for v2"):
        DEFAULT_COMPONENT_REGISTRY.validate_model(replace(v1, support={"zmax": 1.9}))


def test_v2_prior_set_must_equal_the_hyperparameters():
    root = v2_root_model_spec()
    assert set(required_hyperparameters(root)) == set(root.priors)
    extra = dict(root.priors)
    extra["alpha"] = root.priors["alpha_1"]
    with pytest.raises(ValueError, match="unused=\\['alpha'\\]"):
        DEFAULT_COMPONENT_REGISTRY.validate_model(replace(root, priors=extra))
    missing = {k: v for k, v in root.priors.items() if k != "beta"}
    with pytest.raises(ValueError, match="missing=\\['beta'\\]"):
        DEFAULT_COMPONENT_REGISTRY.validate_model(replace(root, priors=missing))
    mixed = replace(root, chieff=DEFAULT_COMPONENT_REGISTRY.block("chieff", "truncated_gaussian"))
    with pytest.raises(ValueError, match="mixes a v1 family"):
        DEFAULT_COMPONENT_REGISTRY.validate_model(mixed)
    low = dict(root.priors)
    from gwpop_search.grammar import PriorConfig

    low["mlow_1"] = PriorConfig("uniform", {"low": 2.0, "high": 10.0})
    with pytest.raises(ValueError, match="mlow_1 prior"):
        DEFAULT_COMPONENT_REGISTRY.validate_model(replace(root, priors=low))


def test_detector_frame_density_is_the_source_density_times_the_jacobian():
    graph = enumerate_v2_depth1()
    rng = np.random.default_rng(11)
    n = 2000
    samples = {
        "m1_detector": rng.uniform(4.0, 200.0, n),
        "q": rng.uniform(0.05, 1.0, n),
        "luminosity_distance": rng.uniform(50.0, 15000.0, n),
        "chi_eff": rng.uniform(-0.9, 0.9, n),
    }
    for spec in graph.nodes:
        model = compile_model_spec(spec)
        hp = hyper(spec)
        det = np.asarray(model(samples, hp))
        z = np.asarray(model.cosmology.z_of_dL(jnp.asarray(samples["luminosity_distance"])))
        src = np.asarray(model.source_frame_logpdf(
            m1=samples["m1_detector"] / (1 + z), q=samples["q"], z=z, chi_eff=samples["chi_eff"],
            hyperparameters=hp))
        expect = src - np.log1p(z) - np.log(np.asarray(model.cosmology.ddL_dz(jnp.asarray(z))))
        expect = np.where(z <= 1.9, expect, -np.inf)
        assert np.array_equal(np.isfinite(det), np.isfinite(expect))
        assert np.isfinite(det).mean() > 0.3
        fin = np.isfinite(det)
        assert np.max(np.abs(det[fin] - expect[fin])) < 1e-12
        assert not np.any(np.isnan(det)) and not np.any(det == np.inf)


def test_every_depth1_model_is_finite_somewhere_and_jit_compiles():
    graph = enumerate_v2_depth1()
    rng = np.random.default_rng(5)
    samples = {
        "m1_detector": jnp.asarray(rng.uniform(6.0, 80.0, 500)),
        "q": jnp.asarray(rng.uniform(0.4, 1.0, 500)),
        "luminosity_distance": jnp.asarray(rng.uniform(300.0, 5000.0, 500)),
        "chi_eff": jnp.asarray(rng.uniform(-0.3, 0.3, 500)),
    }
    for spec in graph.nodes:
        model = compile_model_spec(spec)
        out = jax.jit(lambda hp, model=model: model(samples, hp))(hyper(spec))
        assert np.isfinite(np.asarray(out)).mean() > 0.9, spec.short_hash


# ---------------------------------------------------------------------------
# LVK linear chi_eff convention (GWTC-4 release)
# ---------------------------------------------------------------------------

GWTC4_BBH = Path("/hildafs/home/magana/tmp_ondemand_hildafs_phy220048p_symlink/share/"
                 "LVK_population_analyses/analyses_BBH")


@pytest.mark.skipif(not GWTC4_BBH.is_dir(), reason="LVK GWTC-4 analyses not mounted")
@pytest.mark.parametrize("fname,option,pivot", [
    ("BBHCorr_qchieffLinearCorrelationModel.h5", "q", 1.0),
    ("BBHCorr_zchieffLinearCorrelationModel.h5", "z", 0.5),
])
def test_linear_correlation_matches_the_lvk_mean_and_width_curves(fname, option, pivot):
    """mu(x) and sigma(x) of the LVK linear model: intercept at the pivot, ln width.

    The GWTC-4 q release pivots at q = 1 (as v2); the z release pivots at
    z = 0.5, whereas the v2 spec intercept is at z = 0 (the z_pivot option).
    """
    h5py = pytest.importorskip("h5py")
    from gwpop_search.models.declarative import _v2_chi_moments

    with h5py.File(GWTC4_BBH / fname, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f.attrs["hyperparameters"]]
        col = {n: i for i, n in enumerate(names)}
        x = f["posterior/hyperparameter_samples"][:100]
        pos = f["posterior/rates_on_grids/mu_chieff/positions"][0]
        mu_rel = f["posterior/rates_on_grids/mu_chieff/rates"][:100]
        sig_rel = f["posterior/rates_on_grids/sigma_chieff/rates"][:100]
    spec = atom(atom(v2_root_model_spec(), "C1" if option == "q" else "C3"), "C2" if option == "q" else "C4")
    spec = replace(spec, chieff=replace(spec.chieff, options={**spec.chieff.options, f"{option}_pivot": pivot}))
    for i in range(x.shape[0]):
        hp = {"chi_mu": x[i, col["mu_chieff_0"]], "chi_log_sigma": x[i, col["ln_sigma_chieff_0"]],
              f"chi_mu_{option}_slope": x[i, col["mu_chieff_1"]],
              f"chi_log_sigma_{option}_slope": x[i, col["ln_sigma_chieff_1"]]}
        grid = jnp.asarray(pos)
        kw = {"q": grid, "z": grid} if option == "q" else {"q": jnp.ones_like(grid), "z": grid}
        if option == "q":
            kw["z"] = jnp.zeros_like(grid)
        mu, log_sigma = _v2_chi_moments(spec, hp, log_m1=jnp.zeros_like(grid) + math.log(30.0), **kw)
        assert np.max(np.abs(np.asarray(mu) - mu_rel[i])) < 1e-12
        assert np.max(np.abs(np.exp(np.asarray(log_sigma)) / sig_rel[i] - 1.0)) < 1e-12
