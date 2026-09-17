"""Small deterministic synthetic catalogs used to validate the data/HBI boundary."""

from __future__ import annotations

import numpy as np

from .posterior import PosteriorCatalog
from .schema import CoordinateBasis
from .selection import Campaign, SelectionCatalog, SelectionMode

TOY_BASIS = CoordinateBasis(
    name="toy_source_m1_q_z_chieff",
    coordinates=("m1_source", "q", "z", "chi_eff"),
    frame="source_mass_redshift",
    spin_parameterization="chi_eff",
    density_measure="dm1_source dq dz dchi_eff",
    version="1",
)


def make_toy_posterior_catalog(seed: int = 7) -> PosteriorCatalog:
    rng = np.random.default_rng(seed)
    counts = np.asarray([5, 8, 6], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    n = int(offsets[-1])
    samples = {
        "m1_source": rng.uniform(15.0, 55.0, n),
        "q": rng.uniform(0.35, 1.0, n),
        "z": rng.uniform(0.03, 0.7, n),
        "chi_eff": rng.normal(0.05, 0.15, n),
    }
    return PosteriorCatalog(
        event_names=("GWTOY_A", "GWTOY_B", "GWTOY_C"),
        offsets=offsets,
        samples=samples,
        log_ref_density=np.zeros(n),
        basis=TOY_BASIS,
        metadata={"fixture": "toy-pe", "seed": seed},
    )


def make_toy_selection_catalog(seed: int = 11) -> SelectionCatalog:
    rng = np.random.default_rng(seed)
    n_o3, n_o4 = 9, 13
    n = n_o3 + n_o4
    samples = {
        "m1_source": rng.uniform(10.0, 70.0, n),
        "q": rng.uniform(0.2, 1.0, n),
        "z": rng.uniform(0.01, 1.0, n),
        "chi_eff": rng.uniform(-0.8, 0.8, n),
    }
    campaigns = (
        Campaign("O3", n_draw=1000, observing_time_yr=0.8),
        Campaign("O4", n_draw=1600, observing_time_yr=1.2),
    )
    return SelectionCatalog(
        samples=samples,
        log_draw_density=np.zeros(n),
        campaign_id=np.asarray(["O3"] * n_o3 + ["O4"] * n_o4),
        campaigns=campaigns,
        basis=TOY_BASIS,
        mode=SelectionMode.RAW_DRAW,
        metadata={"fixture": "toy-selection", "seed": seed},
    )
