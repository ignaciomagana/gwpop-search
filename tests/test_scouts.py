import numpy as np

from gwpop_search.grammar import DEFAULT_MUTATIONS
from gwpop_search.scouts import (
    HSGPAxis,
    HSGPDesign,
    basis_matrix,
    laplacian_frequencies,
    mutation_for_proposal,
    proposal_from_summary,
    squared_exponential_spectral_weights,
    weighted_linear_dependence,
)


def test_hsgp_basis_has_expected_shape_and_approximate_orthonormality():
    axis = HSGPAxis(
        "m1",
        lower=5.0,
        upper=90.0,
        modes=5,
        boundary_factor=1.5,
    )
    values = np.linspace(
        axis.center - axis.domain_half_width,
        axis.center + axis.domain_half_width,
        20_001,
    )
    phi = basis_matrix(values, axis)
    gram = np.trapezoid(
        phi[:, :, None] * phi[:, None, :],
        values,
        axis=0,
    )

    assert phi.shape == (values.size, 5)
    np.testing.assert_allclose(gram, np.eye(5), atol=3e-4)


def test_hsgp_tensor_basis_and_spectral_scale_are_deterministic():
    design = HSGPDesign(
        (
            HSGPAxis("m1", 5.0, 90.0, modes=3),
            HSGPAxis("q", 0.05, 1.0, modes=4),
        )
    )
    samples = {
        "m1": np.asarray([10.0, 30.0, 60.0]),
        "q": np.asarray([0.2, 0.7, 0.9]),
    }
    matrix = design.tensor_basis(samples)
    scale = design.tensor_spectral_scale(
        amplitudes={"m1": 1.0, "q": 1.0},
        length_scales={"m1": 10.0, "q": 0.2},
    )

    assert design.n_coefficients == 12
    assert matrix.shape == (3, 12)
    assert scale.shape == (12,)
    assert np.all(np.isfinite(matrix))
    assert np.all(scale > 0.0)
    np.testing.assert_array_equal(matrix, design.tensor_basis(samples))


def test_hsgp_frequencies_and_spectral_weights_are_positive_ordered():
    axis = HSGPAxis("z", 0.0, 2.5, modes=8)
    omega = laplacian_frequencies(axis)
    weights = squared_exponential_spectral_weights(
        axis,
        amplitude=1.2,
        length_scale=0.4,
    )

    assert np.all(np.diff(omega) > 0.0)
    assert np.all(weights > 0.0)
    assert weights[-1] < weights[0]


def test_injected_residual_dependency_proposes_existing_legal_mutation():
    rng = np.random.default_rng(123)
    q = rng.uniform(0.1, 1.0, 3000)
    residual = 0.8 * (q - 0.6) + rng.normal(0.0, 0.15, q.size)

    summary = weighted_linear_dependence(
        q,
        residual,
        target="chieff_width",
        covariate="q",
    )
    proposal = proposal_from_summary(summary, minimum_abs_z=4.0)

    assert summary.slope > 0.0
    assert summary.z_score > 4.0
    assert proposal is not None
    assert proposal.status == "proposed"
    assert proposal.mutation_id == "chieff.width.linear_q"
    mutation = mutation_for_proposal(proposal)
    assert mutation.mutation_id == proposal.mutation_id
    assert mutation in DEFAULT_MUTATIONS


def test_null_residual_does_not_automatically_propose_structure():
    rng = np.random.default_rng(987)
    z = rng.uniform(0.0, 1.5, 4000)
    residual = rng.normal(0.0, 1.0, z.size)

    summary = weighted_linear_dependence(
        z,
        residual,
        target="chieff_mean",
        covariate="z",
    )
    proposal = proposal_from_summary(summary, minimum_abs_z=4.0)

    assert abs(summary.z_score) < 4.0
    assert proposal is None


def test_unmapped_scout_target_never_creates_arbitrary_mutation():
    x = np.linspace(0.0, 1.0, 200)
    residual = 5.0 * x
    summary = weighted_linear_dependence(
        x,
        residual,
        target="made_up_target",
        covariate="x",
    )
    assert proposal_from_summary(summary, minimum_abs_z=1.0) is None
