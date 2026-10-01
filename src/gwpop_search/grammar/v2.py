"""The v2 GWTC-5 atom search: root R0, the 19 depth-1 atoms, depth-2 rule, checks.

Spec: the approved plan of 2026-09-30 (``scalable-stargazing-shamir.md``) and
the node table of ``report/dag_artifact_v2/build_v2.py``. Families and the
exact hyperparameter set of every model are in :mod:`.v2_structure`; the
densities are compiled by ``gwpop_search.models.declarative``.

Status: **DRAFT** until the operator freezes the v2 pre-registration. Every
prior the spec marks "set in the pre-registration draft" is listed in
:data:`V2_DRAFT_PRIORS` with its source and rationale.

Root R0 ("LVK Default BBH + Gaussian chi_eff"; GWTC-5 Table 5 priors):

* mass ``bp2p``: broken power law + peaks near 10 and 35 Msun with the Planck
  low-mass taper, alpha_1, alpha_2 ~ U(-4, 12), m_break ~ U(20, 50),
  mu_p10 ~ U(5, 20), mu_p35 ~ U(25, 60), sigma ~ U(0, 10),
  (lam_pl, lam_p10, lam_p35) ~ Dirichlet(1, 1, 1), mlow_1 ~ U(3, 10),
  delta_m_1 ~ U(0, 10), mmax = 300 (support);
* pairing ``tapered_powerlaw_q``: q^beta S(q m1; mlow_2, delta_m_2) / Z_q(m1),
  beta ~ U(-2, 7), mlow_2 | mlow_1 ~ U(3, mlow_1), delta_m_2 ~ U(0, 10);
* redshift ``powerlaw_1pz``: rate ~ (1+z)^kappa, kappa ~ U(-10, 10), z_max = 1.9;
* chi_eff ``linear_gaussian``: truncated Gaussian, mu ~ U(-1, 1), ln sigma ~ U(-5, 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .enumerate import ModelEdge, ModelGraph, enumerate_model_graph
from .mutations import ConditionalPriors, InapplicableMutation, MutationSpec, apply_mutation
from .registry import DEFAULT_COMPONENT_REGISTRY, ComponentRegistry
from .schema import ModelSpec, PriorConfig, structural_diff_axes
from .v2_structure import (
    CHIEFF_CORRELATION_OPTIONS,
    V2_MASS_FAMILIES,
    V2_CHIEFF_FAMILIES,
    mass_components,
)

V2_PROFILE = "gwtc5-v2"
V2_GRAPH_STATUS = "DRAFT"


def _u(low: float, high: float) -> PriorConfig:
    return PriorConfig("uniform", {"low": low, "high": high})


def _lu(low: float, high: float) -> PriorConfig:
    return PriorConfig("log_uniform", {"low": low, "high": high})


#: Model-wide support of every v2 model (part of every v2 model hash).
V2_SUPPORT: dict[str, object] = {
    # OD-6: z_max = 1.9 everywhere (selection and PE exports record 1.9).
    "zmax": 1.9,
    # Operator decision 2 (2026-09-30): the exact LVK floor, q nodes
    # linspace(0.001, 1, 500), so R0 reproduces the GWTC-5 Default release grids
    # (G2a, tests/test_v2_rates_on_grids.py). Replaces the OD-12 harness floor
    # 0.05 (the XPHM-SpinTaylor PE q prior minimum).
    "q_floor": 0.001,
    # OD-8: m2 >= 3 Msun (mlow_1 ~ U(3, 10), mlow_2 ~ U(3, mlow_1)); also the
    # lower edge of the LVK normalisation grid (minimum_mass = 3).
    "mmin": 3.0,
    # GWTC-5 fixed mmax = 300 Msun; also the upper edge of the grid
    # (maximum_mass = 300). OD-8 ceiling 590 Msun is respected.
    "mmax": 300.0,
    # LVK normalisation convention (gates/G2a_grids.md 0.2): geomspace m1 nodes,
    # Z_q linear in ln m1, q nodes linspace(q_floor, 1, n_q).
    "m1_grid": "geomspace",
    "n_m1": 1000,
    "n_q": 500,
    "redshift_quadrature_order": 96,
    # The v2 selection is sky-marginalised (S5 has no sky columns); every v2
    # population is isotropic, so the density is taken with respect to the
    # non-sky coordinates only.
    "sky": "marginalized",
    # Population cosmology D9 (BUILD_PLAN 2.6).
    "cosmology": {"H0": 67.74, "Om0": 0.3089},
}

_ROOT_PRIORS: dict[str, PriorConfig] = {
    "alpha_1": _u(-4.0, 12.0),
    "alpha_2": _u(-4.0, 12.0),
    "m_break": _u(20.0, 50.0),
    "mu_p10": _u(5.0, 20.0),
    "sigma_p10": _u(0.0, 10.0),
    "mu_p35": _u(25.0, 60.0),
    "sigma_p35": _u(0.0, 10.0),
    "lam_u_pl": _u(0.0, 1.0),
    "lam_u_p10": _u(0.0, 1.0),
    "lam_u_p35": _u(0.0, 1.0),
    "mlow_1": _u(3.0, 10.0),
    "delta_m_1": _u(0.0, 10.0),
    "mlow_2_frac": _u(0.0, 1.0),
    "delta_m_2": _u(0.0, 10.0),
    "beta": _u(-2.0, 7.0),
    "kappa": _u(-10.0, 10.0),
    "chi_mu": _u(-1.0, 1.0),
    "chi_log_sigma": _u(-5.0, 0.0),
}


def v2_root_model_spec(registry: ComponentRegistry = DEFAULT_COMPONENT_REGISTRY) -> ModelSpec:
    """Root R0 of the v2 search (the ``gwtc5-v2`` hyperprior profile)."""
    model = ModelSpec(
        mass=registry.block("mass", "bp2p"),
        pairing=registry.block("pairing", "tapered_powerlaw_q"),
        chieff=registry.block("chieff", "linear_gaussian"),
        redshift=registry.block("redshift", "powerlaw_1pz"),
        mixture=registry.block("mixture", "single"),
        priors=dict(_ROOT_PRIORS),
        support=dict(V2_SUPPORT),
    )
    registry.validate_model(model)
    return model


# ---------------------------------------------------------------------------
# The 19 depth-1 atoms
# ---------------------------------------------------------------------------

_SLOPES = tuple(CHIEFF_CORRELATION_OPTIONS.values())
_CORRELATION_SWITCHES = tuple(CHIEFF_CORRELATION_OPTIONS)
_SHAPE_EXTRA = (
    "chi_mu_2_frac", "chi_log_sigma_2", "chi_fraction", "chi_fraction_low",
    "chi_fraction_high", "chi_fraction_m_t", "chi_fraction_width", "chi_eps", "chi_nu",
)
_MIXTURE_COMPONENT_2 = {
    # component 2 mean mu_2 ~ U(mu_1, 1) (ordered means), ln sigma_2 as the root.
    "chi_mu_2_frac": _u(0.0, 1.0),
    "chi_log_sigma_2": _u(-5.0, 0.0),
}


#: Pairing slope of one mass component when the pairing is per component (P2
#: and the D5 alternative root A1); the P2 prior.
_BETA_COMPONENT_PRIOR = _u(-10.0, 13.0)


def _mass(mid, family, description, *, removals=(), updates=None, added=(), dropped=()):
    """A mass-family atom. ``added`` / ``dropped`` name the mass components it adds
    or removes: on a per-component pairing (root A1) their pairing slope
    ``beta_<c>`` is added (P2 prior) or removed with the component, so the atom
    stays one structural axis and the D5 edge is the same physical question."""
    conditional = ()
    if added or dropped:
        conditional = (ConditionalPriors(
            "pairing", "beta_dependence", "per_mass_component",
            prior_updates={f"beta_{c}": _BETA_COMPONENT_PRIOR for c in added},
            prior_removals=tuple(f"beta_{c}" for c in dropped),
        ),)
    return MutationSpec(
        mid, "mass", "change_family", family, requires_family="bp2p",
        prior_removals=tuple(removals), prior_updates=dict(updates or {}), description=description,
        conditional_priors=conditional,
    )


def _corr(mid, option, prior, description):
    return MutationSpec(
        mid, "chieff", "set_option", "linear", option=option,
        requires_any_family=V2_CHIEFF_FAMILIES,
        prior_updates={CHIEFF_CORRELATION_OPTIONS[option]: prior}, description=description,
    )


def _shape(mid, family, updates, description, *, family_options=None):
    return MutationSpec(
        mid, "chieff", "change_family", family, requires_family="linear_gaussian",
        prior_removals=_SHAPE_EXTRA, prior_updates=dict(updates),
        family_options=dict(family_options or {}), carry_options=_CORRELATION_SWITCHES,
        description=description,
    )


V2_MUTATIONS: tuple[MutationSpec, ...] = (
    _mass("v2.mass.drop_p35", "bp1p_low", "M1: drop the 35 Msun peak (BP1P)",
          removals=("mu_p35", "sigma_p35", "lam_u_p35"), dropped=("p35",)),
    _mass("v2.mass.drop_p10", "bp1p_high", "M2: drop the 10 Msun peak",
          removals=("mu_p10", "sigma_p10", "lam_u_p10"), dropped=("p10",)),
    _mass("v2.mass.add_p20", "bp3p", "M3: third peak near 20 Msun; mu_p10 narrowed to U(5, 14)",
          updates={"mu_p3": _u(14.0, 24.0), "sigma_p3": _u(0.0, 10.0), "lam_u_p3": _u(0.0, 1.0),
                   "mu_p10": _u(5.0, 14.0)}, added=("p3",)),
    _mass("v2.mass.add_p70", "bp3p", "M4: third peak near 60-90 Msun",
          updates={"mu_p3": _u(60.0, 90.0), "sigma_p3": _u(0.0, 10.0), "lam_u_p3": _u(0.0, 1.0)},
          added=("p3",)),
    _mass("v2.mass.no_break", "pl2p", "M5: no break (single power law + 2 peaks)",
          removals=("alpha_1", "alpha_2", "m_break"), updates={"alpha": _u(-4.0, 12.0)}),
    _corr("v2.chieff.mean_q", "mean_q", _u(-2.0, 2.0),
          "C1: chi_eff mean linear in q (intercept at q = 1)"),
    _corr("v2.chieff.log_sigma_q", "log_sigma_q", _u(-12.0, 4.0),
          "C2: chi_eff ln-width linear in q (intercept at q = 1)"),
    _corr("v2.chieff.mean_z", "mean_z", _u(-1.0, 1.0),
          "C3: chi_eff mean linear in z (intercept at z = 0)"),
    _corr("v2.chieff.log_sigma_z", "log_sigma_z", _u(-3.0, 5.0),
          "C4: chi_eff ln-width linear in z (intercept at z = 0)"),
    _corr("v2.chieff.mean_log_m1", "mean_log_m1", _u(-2.0 / 3.0, 2.0 / 3.0),
          "C5: chi_eff mean linear in ln(m1 / 30 Msun)"),
    _corr("v2.chieff.log_sigma_log_m1", "log_sigma_log_m1", _u(-4.0, 4.0),
          "C6: chi_eff ln-width linear in ln(m1 / 30 Msun)"),
    _shape("v2.chieff.mixture", "linear_gaussian_mixture",
           {"chi_fraction": _u(0.0, 1.0), **_MIXTURE_COMPONENT_2},
           "S1: two-component truncated-Gaussian chi_eff mixture (ordered means, constant fraction)"),
    _shape("v2.chieff.mixture_fraction_log_m1", "linear_gaussian_mixture",
           {"chi_fraction_low": _u(0.0, 1.0), "chi_fraction_high": _u(0.0, 1.0),
            "chi_fraction_m_t": _lu(10.0, 100.0), "chi_fraction_width": _lu(0.05, 1.0),
            **_MIXTURE_COMPONENT_2},
           "S2: chi_eff mixture with the component-2 fraction logistic in ln m1",
           family_options={"fraction_dependence": "logistic_log_m1"}),
    _shape("v2.chieff.skew_normal", "linear_skew_normal", {"chi_eps": _u(-1.0, 1.0)},
           "S3: epsilon-skew-normal chi_eff"),
    _shape("v2.chieff.student_t", "linear_student_t", {"chi_nu": _lu(1.0, 100.0)},
           "S4: truncated Student-t chi_eff"),
    MutationSpec(
        "v2.pairing.beta_logistic_log_m1", "pairing", "set_option", "logistic_log_m1",
        option="beta_dependence", requires_family="tapered_powerlaw_q", prior_removals=("beta",),
        # a step of the one slope: defined relative to a constant slope only, so
        # not applicable on root A1 (beta per component) -- D5 "not_applicable".
        requires_options={"beta_dependence": "constant"},
        prior_updates={"beta_low": _u(-2.0, 7.0), "beta_high": _u(-2.0, 7.0),
                       "beta_m_t": _lu(10.0, 100.0), "beta_width": _lu(0.05, 1.0)},
        description="P1: pairing slope beta steps logistically in ln m1",
    ),
    MutationSpec(
        "v2.pairing.beta_per_component", "pairing", "set_option", "per_mass_component",
        option="beta_dependence", requires_family="tapered_powerlaw_q", prior_removals=("beta",),
        # one slope per component of the parent's mass family (bp2p: pl, p10, p35)
        conditional_priors=tuple(
            ConditionalPriors("mass", None, family,
                              prior_updates={f"beta_{c}": _BETA_COMPONENT_PRIOR for c in ("pl",) + peaks})
            for family, (_, peaks) in V2_MASS_FAMILIES.items()
        ),
        description="P2: one pairing slope per mass component (LVK 'Extended')",
    ),
    MutationSpec(
        "v2.redshift.madau_dickinson", "redshift", "change_family", "madau_dickinson_psi",
        requires_family="powerlaw_1pz", prior_removals=("kappa",),
        # MD replaces a constant power law; on root A2 (kappa(m1)) the child would
        # drop kappa(m1) too, a different hypothesis -- D5 "not_applicable".
        requires_options={"kappa_dependence": "constant"},
        prior_updates={"md_gamma": _u(-10.0, 10.0), "md_kappa": _u(0.0, 10.0),
                       "md_z_peak": _u(0.0, 4.0)},
        description="Z1: Madau-Dickinson rate history (gwpopulation form)",
    ),
    MutationSpec(
        "v2.redshift.kappa_log_m1", "redshift", "set_option", "linear_log_m1",
        option="kappa_dependence", requires_family="powerlaw_1pz",
        prior_updates={"kappa_log_m1_slope": _u(-4.0, 4.0)},
        description="Z2: kappa linear in ln(m1 / 30 Msun)",
    ),
)
V2_MUTATION_TABLE: dict[str, MutationSpec] = {m.mutation_id: m for m in V2_MUTATIONS}

#: Spec atom id -> mutation id, in the spec's table order.
V2_ATOM_IDS: dict[str, str] = {
    "M1": "v2.mass.drop_p35",
    "M2": "v2.mass.drop_p10",
    "M3": "v2.mass.add_p20",
    "M4": "v2.mass.add_p70",
    "M5": "v2.mass.no_break",
    "C1": "v2.chieff.mean_q",
    "C2": "v2.chieff.log_sigma_q",
    "C3": "v2.chieff.mean_z",
    "C4": "v2.chieff.log_sigma_z",
    "C5": "v2.chieff.mean_log_m1",
    "C6": "v2.chieff.log_sigma_log_m1",
    "S1": "v2.chieff.mixture",
    "S2": "v2.chieff.mixture_fraction_log_m1",
    "S3": "v2.chieff.skew_normal",
    "S4": "v2.chieff.student_t",
    "P1": "v2.pairing.beta_logistic_log_m1",
    "P2": "v2.pairing.beta_per_component",
    "Z1": "v2.redshift.madau_dickinson",
    "Z2": "v2.redshift.kappa_log_m1",
}
V2_ATOM_ORDER = tuple(V2_ATOM_IDS)
V2_MUTATION_ATOM = {mid: aid for aid, mid in V2_ATOM_IDS.items()}
CHIEFF_ATOMS = ("C1", "C2", "C3", "C4", "C5", "C6", "S1", "S2", "S3", "S4")
MASS_ATOMS = ("M1", "M2", "M3", "M4", "M5")

#: Priors the spec leaves to the pre-registration draft (and the two support
#: choices that are interpretations), with sources. All DRAFT: the operator
#: approves them at freeze.
V2_DRAFT_PRIORS: dict[str, dict[str, str]] = {
    "C5.chi_mu_log_m1_slope": {
        "prior": "U(-2/3, 2/3) per e-fold of m1",
        "source": "GWTC-4 (arXiv:2508.18083) Table 10 q-slope delta_mu|q ~ U(-2, 2); Tong+22, Antonini+24",
        "rationale": "over ln m1 in [ln 5, ln 100] (~3 e-folds, where the events are) it spans the same |d mu| <= 2 "
                     "as the q slope over q in [0, 1]. Over the full population support [3, 300] Msun (4.6 e-folds) "
                     "it spans |d mu| <= 3.1 (the q slope spans 2.0 over q in [0.001, 1]); the alternative that "
                     "matches over the full support is U(-0.43, 0.43). OPERATOR CHOICE at freeze",
    },
    "C6.chi_log_sigma_log_m1_slope": {
        "prior": "U(-4, 4) per e-fold of m1",
        "source": "GWTC-4 Table 10 delta ln sigma|q ~ U(-12, 4); Tong+22, Antonini+24, Plunkett+26",
        "rationale": "|d ln sigma| <= 12 over ~3 e-folds of m1 (5-100 Msun, where the events are), the largest excursion "
                     "of the q-slope prior; symmetric because the sign is not predicted. Over the full support [3, 300] "
                     "Msun (4.6 e-folds) it spans |d ln sigma| <= 18.4 versus 12.0 for the q slope over q in [0.001, 1]; "
                     "the alternative that matches over the full support is U(-2.6, 2.6). OPERATOR CHOICE at freeze",
    },
    "S1.chi_mu_2_frac": {
        "prior": "U(0, 1): mu_2 = mu_1 + f (1 - mu_1), i.e. mu_2 | mu_1 ~ U(mu_1, 1)",
        "source": "spec 'ordered means'; v1 second component at chi_eff ~ 0.45; Hussain+26",
        "rationale": "orders the components (label-switching free) and keeps mu_2 inside [-1, 1]",
    },
    "S1.chi_log_sigma_2": {
        "prior": "U(-5, 0)",
        "source": "root ln sigma prior (GWTC-5 Table 5 style)",
        "rationale": "same width range as the bulk component",
    },
    "S2.chi_fraction_low/high": {
        "prior": "U(0, 1) each",
        "source": "spec (f_low -> f_high)",
        "rationale": "fraction of component 2 below / above the transition",
    },
    "S2.chi_fraction_m_t": {
        "prior": "LU(10, 100) Msun",
        "source": "Antonini+24 (transition ~45 Msun), Plunkett+26, Flanagan+26 mass regimes, Li+25",
        "rationale": "brackets the proposed transitions with a scale-free prior",
    },
    "S2.chi_fraction_width": {
        "prior": "LU(0.05, 1) in ln m1",
        "source": "draft choice",
        "rationale": "from a sharp step (5% in mass) to a transition spread over a factor e",
    },
    "P1.beta_m_t": {
        "prior": "LU(10, 100) Msun",
        "source": "Flanagan+26; marked-transition q steepening at 42.7 Msun",
        "rationale": "scale-free, brackets the proposed pairing transitions",
    },
    "P1.beta_width": {
        "prior": "LU(0.05, 1) in ln m1",
        "source": "draft choice (as S2)",
        "rationale": "sharp to broad transition",
    },
    "Z1.md_gamma/md_kappa/md_z_peak": {
        "prior": "gamma ~ U(-10, 10), kappa ~ U(0, 10), z_peak ~ U(0, 4)",
        "source": "Madau & Dickinson 2014 (2.7, 5.6, 1.9); GWTC-5 MD posterior (3.0, 4.2, 1.6)",
        "rationale": "gamma shares the root kappa prior so kappa_MD = 0 is the exact power-law null",
    },
    "Z2.kappa_log_m1_slope": {
        "prior": "U(-4, 4) per e-fold of m1",
        "source": "draft choice",
        "rationale": "kappa changes by up to ~4.4 between 10 and 30 Msun, several times the GWTC-5 kappa width",
    },
    "Z2.normalisation": {
        "prior": "p(z | m1) = dVc/dz (1+z)^(kappa(m1)-1) / N(kappa(m1)), normalised per m1 (implemented)",
        "source": "draft choice; OPEN OPERATOR DECISION",
        "rationale": "with the per-m1 normalisation BP2P is the mass spectrum integrated over the volume to z < 1.9. "
                     "The usual mass-dependent-evolution convention R(m1, z) = R(m1, 0) (1+z)^kappa(m1) (BP2P = the "
                     "local, z = 0, mass function) drops the N(kappa(m1)) division (the shape likelihood is invariant "
                     "to a Lambda-dependent constant). Both reduce exactly to R0 at slope 0 (SDDR exact either way) "
                     "but give different evidences for Z2 and for the whole A2 suite",
    },
    "taper.form": {
        "prior": "sharp cut: ln L -> -inf where sigma^2_lnL > 1 (sigma^2 = 1 kept); D3 sensitivity rerun at 4. "
                 "D2 cut bracketing: D2 must also hold at sigma^2 <= 0.9, from Z(0.9) = Z(1) P_post(sigma^2 <= "
                 "0.9) (exact for the sharp cut; the fractions are recorded by every evaluation). The near-cut "
                 "band mass sigma^2 > 0.95 is reported, not gating",
        "source": "operator decision 1 (2026-09-30): adopt the exact LVK GWTC-5 form. gwpopulation @b3a34f9 "
                  "hyperpe.py L185-189 (ln_l -= inf * (maximum_uncertainty < variance)), passed by "
                  "gwpopulation_pipe @88c2e2944b data_analysis.py L232-254 (--maximum-uncertainty); GWTC-5 "
                  "arXiv:2605.27226 Sec. III 'maximum variance of 1'; the Default release posteriors stop at "
                  "sigma^2 = 1 - 5.9e-6 with 0 of 8200 samples above (staging/v2/TAPER_FORM.md)",
        "rationale": "R0 ports the gwpopulation BP2P Default fit, whose guard is this cut. The Callister & Farr "
                     "S(x) = 1/(1 + x^-30) acts on N_eff^inj/(4 N_obs), not on sigma^2; it stays available as "
                     "VarianceTaper(kind='smooth') for diagnostics only. The LVK relaxed run uses variance 4, as "
                     "does v2's D3 (operator decision 2026-10-01). The DRAFT near-cut mass limit (0.10 at "
                     "sigma^2 > 0.95) is replaced by the direct tighter-cut measurement (operator decision "
                     "2026-10-01): the LVK Default posterior has 67% of its mass at sigma^2 > 0.95, so a mass "
                     "limit would block every edge, while ln BF(0.9) = ln BF(1) + ln P_child - ln P_parent "
                     "measures what the cut does to each edge",
    },
    "support.q_floor": {
        "prior": "0.001 (fixed support; truncate and renormalise p(q | m1) on [max(0.001, mlow_2/m1), 1]; "
                 "q nodes linspace(0.001, 1, 500))",
        "source": "operator decision 2 (2026-09-30): the exact LVK value (GWTC-5 rates_on_grids mass_ratio "
                  "positions start at 0.001); replaces the OD-12 harness floor 0.05",
        "rationale": "R0 then reproduces the GWTC-5 Default release dR/dm1 and dR/dq grids to < 1e-6 relative "
                     "(G2a, pinned by tests/test_v2_rates_on_grids.py). Below q = 0.05 there is no "
                     "XPHM-SpinTaylor PE prior support; the population mass there is small (LVK rate fraction "
                     "<= 1.5e-6 at the release draws tested, more for beta < 0) and found injections with "
                     "q < 0.05 now carry weight",
    },
    "A1.mass_atoms.beta_c": {
        "prior": "beta_p3 ~ U(-10, 13) added with M3/M4 on A1; beta_p35 / beta_p10 removed with M1 / M2",
        "source": "P2 per-component prior (LVK 'Extended')",
        "rationale": "on the per-component pairing root each mass component carries its own pairing slope, so "
                     "adding/dropping a component adds/drops its slope (same structural axis; the SDDR null "
                     "lam_u_c = 0 leaves beta_c unidentified). Also makes mass x P2 depth-2 pairs commute",
    },
    "C3/C4.z_pivot": {
        "prior": "0.0 (intercept at z = 0)",
        "source": "operator spec 2026-09-30",
        "rationale": "NOTE: the LVK GWTC-4 release (BBHCorr_zchieffLinearCorrelationModel) uses an intercept at z = 0.5",
    },
    "S4.form": {
        "prior": "location-scale Student-t truncated to [-1, 1], nu ~ LU(1, 100)",
        "source": "spec; LVK Student-t release not available on disk",
        "rationale": "form not verified against an LVK release",
    },
}


def mutations_for_profile(profile: str) -> tuple[MutationSpec, ...]:
    """The atom set that is enumerated under a registered hyperprior profile."""
    from .mutations import DEFAULT_MUTATIONS

    return V2_MUTATIONS if profile == V2_PROFILE else DEFAULT_MUTATIONS


def enumerate_v2_depth1(root: ModelSpec | None = None) -> ModelGraph:
    """Root + one child per atom (20 nodes, 19 edges for R0)."""
    root = v2_root_model_spec() if root is None else root
    return enumerate_model_graph(root, mutations=V2_MUTATIONS, max_depth=1, max_models=100)


# ---------------------------------------------------------------------------
# Depth 2: pre-declared conditional pairs
# ---------------------------------------------------------------------------

#: Spec priority pairs (plan: width(q) + mixture-fraction(m1); width(q) +
#: width(z); width(q) + width(m1)); then mass x spin, then the other chi_eff
#: pairs, then every other passing pair.
V2_DEPTH2_PRIORITY_PAIRS = (("C2", "S2"), ("C2", "C4"), ("C2", "C6"))
V2_DEPTH2_CAP = 12


class NotComposable(ValueError):
    """Two atoms cannot be applied together as one depth-2 model."""


def compose_atoms(root: ModelSpec, atom_a: str, atom_b: str, *,
                  table: Mapping[str, MutationSpec] = V2_MUTATION_TABLE) -> ModelSpec:
    """The model with both atoms applied to ``root``.

    Both orders are tried; an order is accepted only if the result differs
    from each single-atom child by exactly the other atom's axis (so a family
    change cannot silently discard the other atom). Both accepted orders must
    agree.
    """
    ma, mb = table[V2_ATOM_IDS[atom_a]], table[V2_ATOM_IDS[atom_b]]
    child_a, child_b = apply_mutation(root, ma), apply_mutation(root, mb)
    results = {}
    errors = []
    for first_child, second in ((child_a, mb), (child_b, ma)):
        try:
            model = apply_mutation(first_child, second)
        except (InapplicableMutation, ValueError) as exc:
            errors.append(str(exc))
            continue
        if (structural_diff_axes(child_a, model) == (mb.axis,)
                and structural_diff_axes(child_b, model) == (ma.axis,)):
            results[model.model_hash] = model
        else:
            errors.append(f"applying {second.mutation_id} discards the other atom")
    if not results:
        raise NotComposable(f"{atom_a} x {atom_b}: " + "; ".join(errors))
    if len(results) > 1:
        raise NotComposable(f"{atom_a} x {atom_b}: the two orders give different models")
    return next(iter(results.values()))


def _block_of(atom: str) -> str:
    return {"M": "mass", "C": "chieff", "S": "chieff", "P": "pairing", "Z": "redshift"}[atom[0]]


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    order = {aid: i for i, aid in enumerate(V2_ATOM_ORDER)}
    return (a, b) if order[a] <= order[b] else (b, a)


def depth2_priority(passing: Iterable[str]) -> list[tuple[int, tuple[str, str]]]:
    """All pairs of passing atoms in the pre-declared priority order ``(tier, pair)``."""
    requested = set(passing)
    unknown = requested - set(V2_ATOM_ORDER)
    if unknown:
        raise ValueError(f"unknown atom id(s) {sorted(unknown)}")
    passing = [a for a in V2_ATOM_ORDER if a in requested]
    seen: set[tuple[str, str]] = set()
    ranked: list[tuple[int, tuple[str, str]]] = []

    def add(tier, a, b):
        pair = _ordered_pair(a, b)
        if a != b and a in passing and b in passing and pair not in seen:
            seen.add(pair)
            ranked.append((tier, pair))

    for a, b in V2_DEPTH2_PRIORITY_PAIRS:
        add(0, a, b)
    for m in MASS_ATOMS:
        for s in CHIEFF_ATOMS:
            add(1, m, s)
    for i, a in enumerate(CHIEFF_ATOMS):
        for b in CHIEFF_ATOMS[i + 1:]:
            add(2, a, b)
    for i, a in enumerate(V2_ATOM_ORDER):
        for b in V2_ATOM_ORDER[i + 1:]:
            add(3, a, b)
    return ranked


@dataclass(frozen=True)
class Depth2Slot:
    atoms: tuple[str, str]
    tier: int
    model: ModelSpec

    def to_dict(self) -> dict[str, object]:
        return {"atoms": list(self.atoms), "tier": self.tier, "model_hash": self.model.model_hash}


@dataclass(frozen=True)
class Depth2Plan:
    passing: tuple[str, ...]
    cap: int
    selected: tuple[Depth2Slot, ...]
    not_composable: tuple[tuple[tuple[str, str], str], ...] = ()
    over_cap: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "rule": "pairs of depth-1 edges passing D1 + D2 only; priority tiers "
                    "0 = spec pairs, 1 = mass x chi_eff, 2 = chi_eff x chi_eff, 3 = other; cap",
            "passing": list(self.passing),
            "cap": self.cap,
            "selected": [slot.to_dict() for slot in self.selected],
            "not_composable": [{"atoms": list(p), "reason": r} for p, r in self.not_composable],
            "over_cap": [list(p) for p in self.over_cap],
        }


def plan_depth2(passing: Iterable[str], *, root: ModelSpec | None = None,
                cap: int = V2_DEPTH2_CAP) -> Depth2Plan:
    """Pre-declared depth-2 enumeration given the depth-1 atoms passing D1 + D2."""
    root = v2_root_model_spec() if root is None else root
    if cap < 0:
        raise ValueError("cap cannot be negative")
    requested = set(passing)
    ranked = depth2_priority(requested)  # validates the atom ids
    passing = tuple(a for a in V2_ATOM_ORDER if a in requested)
    selected: list[Depth2Slot] = []
    bad: list[tuple[tuple[str, str], str]] = []
    over: list[tuple[str, str]] = []
    hashes: set[str] = set()
    for tier, pair in ranked:
        try:
            model = compose_atoms(root, *pair)
        except NotComposable as exc:
            bad.append((pair, str(exc)))
            continue
        if model.model_hash in hashes:  # pragma: no cover - distinct atom pairs give distinct models
            continue
        if len(selected) >= cap:
            over.append(pair)
            continue
        hashes.add(model.model_hash)
        selected.append(Depth2Slot(pair, tier, model))
    return Depth2Plan(passing, cap, tuple(selected), tuple(bad), tuple(over))


def extend_graph_with_depth2(graph: ModelGraph, plan: Depth2Plan, *,
                             table: Mapping[str, MutationSpec] = V2_MUTATION_TABLE) -> ModelGraph:
    """Add the planned depth-2 nodes, with an edge from each depth-1 parent."""
    by_hash = graph.by_hash
    child_of = {}
    for edge in graph.edges:
        if edge.parent_hash == graph.root_hash and edge.depth == 1:
            child_of[V2_MUTATION_ATOM.get(edge.mutation_id, edge.mutation_id)] = edge.child_hash
    nodes = list(graph.nodes)
    depths = dict(graph.depths)
    edges = list(graph.edges)
    for slot in plan.selected:
        h = slot.model.model_hash
        if h not in by_hash:
            nodes.append(slot.model)
            depths[h] = 2
        a, b = slot.atoms
        for parent_atom, other in ((a, b), (b, a)):
            parent_hash = child_of[parent_atom]
            try:
                model = apply_mutation(by_hash[parent_hash], table[V2_ATOM_IDS[other]])
            except (InapplicableMutation, ValueError):
                continue
            if model.model_hash == h:
                edges.append(ModelEdge(parent_hash, h, V2_ATOM_IDS[other], 2))
    return ModelGraph(root_hash=graph.root_hash, nodes=tuple(nodes), edges=tuple(edges), depths=depths)


# ---------------------------------------------------------------------------
# D5 alternative roots
# ---------------------------------------------------------------------------

V2_ALT_ROOT_ATOMS = {"A1": "P2", "A2": "Z2"}


def v2_alternative_roots(root: ModelSpec | None = None) -> dict[str, ModelSpec]:
    """A1 = R0 + beta per mass component, A2 = R0 + kappa(m1)."""
    root = v2_root_model_spec() if root is None else root
    return {
        name: apply_mutation(root, V2_MUTATION_TABLE[V2_ATOM_IDS[atom]])
        for name, atom in V2_ALT_ROOT_ATOMS.items()
    }


#: Why an atom has no D5 edge on an alternative root (keyed by (root, atom)).
_D5_NOT_APPLICABLE_REASONS = {
    ("A1", "P2"): "already part of the root (beta per mass component)",
    ("A1", "P1"): "the logistic step of one pairing slope is defined relative to a constant slope; "
                  "A1 has one slope per mass component",
    ("A2", "Z2"): "already part of the root (kappa(m1))",
    ("A2", "Z1"): "Madau-Dickinson replaces a constant-kappa power law; on A2 the child would also drop "
                  "kappa(m1), a different hypothesis",
}
_D5_NOTES = {
    ("A1", "M1"): "the dropped p35 component's pairing slope beta_p35 is removed with it",
    ("A1", "M2"): "the dropped p10 component's pairing slope beta_p10 is removed with it",
    ("A1", "M3"): "the added component's pairing slope beta_p3 ~ U(-10, 13) (the P2 prior) is added with it",
    ("A1", "M4"): "the added component's pairing slope beta_p3 ~ U(-10, 13) (the P2 prior) is added with it",
}


def v2_d5_atom_semantics(root: ModelSpec | None = None) -> dict[str, dict[str, dict[str, object]]]:
    """Per alternative root and atom: is the D5 edge the same question, or not applicable.

    Computed by applying every atom to each alternative root (a grammar defect
    raises). ``same_edge`` rows carry the SDDR-free structural statement only;
    the claim table reads the not-applicable set from the D5 suite summary.
    """
    out: dict[str, dict[str, dict[str, object]]] = {}
    for name, alt in v2_alternative_roots(root).items():
        rows: dict[str, dict[str, object]] = {}
        for atom in V2_ATOM_ORDER:
            mutation = V2_MUTATION_TABLE[V2_ATOM_IDS[atom]]
            try:
                apply_mutation(alt, mutation)
            except InapplicableMutation as exc:
                rows[atom] = {
                    "status": "not_applicable",
                    "reason": _D5_NOT_APPLICABLE_REASONS.get((name, atom), str(exc)),
                }
                continue
            row: dict[str, object] = {"status": "same_edge"}
            if (name, atom) in _D5_NOTES:
                row["note"] = _D5_NOTES[(name, atom)]
            rows[atom] = row
        out[name] = rows
    return out


def enumerate_alt_root_graph(alt_root: ModelSpec, atoms: Iterable[str]) -> ModelGraph:
    """Depth-1 graph of the given atoms (e.g. the 10 chi_eff atoms + candidates) on an alt root."""
    mutations = [V2_MUTATION_TABLE[V2_ATOM_IDS[a]] for a in atoms]
    return enumerate_model_graph(alt_root, mutations=mutations, max_depth=1, max_models=100)


# ---------------------------------------------------------------------------
# Model-graph provenance checks (gate item G12, model part)
# ---------------------------------------------------------------------------


def v2_model_graph_checks(
    graph: ModelGraph,
    *,
    expected_zmax: float = 1.9,
    expected_q_floor: float = 0.001,
    mmin_low: float = 3.0,
    m1_ceiling: float = 590.0,
    export_zmax: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """G12 model checks over every node of a v2 graph.

    * every model's mmin (mlow_1 prior low and the support floor that bounds
      mlow_2) is >= ``mmin_low`` (OD-8; v1 had U(2, 10));
    * the m1 support ceiling (fixed mmax) is <= ``m1_ceiling`` (OD-8);
    * zmax equals ``expected_zmax`` (OD-6) and, when given, every export zmax;
    * q_floor equals ``expected_q_floor`` (operator decision 2: the LVK 0.001;
      see V2_DRAFT_PRIORS).
    """
    rows = []
    ok = True
    for node in graph.nodes:
        support = node.support
        # v2: mlow_1; a v1 graph (no support block) is checked on its mmin prior
        name = "mlow_1" if "mlow_1" in node.priors else "mmin"
        lo = float(dict(node.priors[name].parameters).get("low", float("nan"))) if name in node.priors \
            else float("nan")
        row = {
            "model_hash": node.model_hash,
            "mlow_1_prior_low": lo,
            "mlow_2_floor": float(support.get("mmin", float("nan"))),
            "m1_ceiling": float(support.get("mmax", float("nan"))),
            "zmax": float(support.get("zmax", float("nan"))),
            "q_floor": float(support.get("q_floor", float("nan"))),
        }
        row["pass"] = bool(
            lo >= mmin_low and row["mlow_2_floor"] >= mmin_low
            and row["m1_ceiling"] <= m1_ceiling
            and row["zmax"] == float(expected_zmax)
            and row["q_floor"] == float(expected_q_floor)
        )
        ok &= row["pass"]
        rows.append(row)
    exports = {}
    for name, value in dict(export_zmax or {}).items():
        exports[name] = {"zmax": float(value), "pass": float(value) == float(expected_zmax)}
        ok &= exports[name]["pass"]
    return {
        "criteria": {
            "mmin_low": mmin_low, "m1_ceiling": m1_ceiling,
            "zmax": expected_zmax, "q_floor": expected_q_floor,
        },
        "n_models": len(rows),
        "models": rows,
        "exports": exports,
        "pass": bool(ok),
    }


def v2_graph_payload(graph: ModelGraph, *, depth2: Depth2Plan | None = None) -> dict[str, object]:
    """Graph JSON (``load_model_graph`` compatible) plus v2 metadata."""
    payload = graph.to_dict()
    atom_of_edge = [V2_MUTATION_ATOM.get(edge.mutation_id) for edge in graph.edges]
    payload["metadata"] = {
        "status": V2_GRAPH_STATUS,
        "hyperprior_profile": V2_PROFILE,
        "spec": "scalable-stargazing-shamir.md (2026-09-30); report/dag_artifact_v2/build_v2.py",
        "support": dict(V2_SUPPORT),
        "atoms": {aid: {"mutation_id": mid, "description": V2_MUTATION_TABLE[mid].description}
                  for aid, mid in V2_ATOM_IDS.items()},
        "edge_atoms": atom_of_edge,
        "draft_priors": V2_DRAFT_PRIORS,
        "depth2": None if depth2 is None else depth2.to_dict(),
        "depth2_rule": {
            "cap": V2_DEPTH2_CAP,
            "priority_pairs": [list(p) for p in V2_DEPTH2_PRIORITY_PAIRS],
            "tiers": ["spec pairs", "mass x chi_eff", "chi_eff x chi_eff", "other"],
        },
        "alternative_roots": {name: {"atom": atom, "model_hash": spec.model_hash}
                              for (name, atom), spec in zip(V2_ALT_ROOT_ATOMS.items(),
                                                            v2_alternative_roots().values())},
        "d5_atom_semantics": v2_d5_atom_semantics(),
        "mass_components_root": list(mass_components(v2_root_model_spec())),
    }
    return payload
