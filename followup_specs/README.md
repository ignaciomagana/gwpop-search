# chi_eff follow-up model specs (exploratory, depth-2 follow-up)

Full declarative `ModelSpec` JSON files for three opt-in chi_eff alternatives,
each built on the frozen GWTC-5 root
(`frozen/gwtc5-bbh-v1a/model_graph.json`, root
`666c4ffc99f000e8c3783589866f0ab6bc120bf161fca5f47b5130983263ff67`,
profile `gwtc5-v1`). Every block option and every hyperprior is written out.
Mass, pairing, redshift and all non-chi_eff priors are identical to the root.

Regenerate (reads the frozen graph only):

    PYTHONPATH=src python scripts/write_followup_specs.py --output followup_specs

Validate:

    gwpop-search validate-model --spec followup_specs/<name>.json

| file | model_hash | mutation path from root |
|---|---|---|
| `chieff_student_t.json` | `5ba71a0ddf2a2f83aac1df628159a9eb51a7c22ea30f23075623c2729e5c9213` | `chieff.family.student_t` |
| `chieff_mixture_logistic_q_fraction.json` | `39017291f77f89e14c162f7523e2e0b2feec42e05551acb56f5beb3c430f9b25` | `chieff.family.gaussian_mixture` -> `chieff.fraction.logistic_q` |
| `chieff_width_logistic_q.json` | `2136018ba5c1ef75d50e23526f2ee33b8e4fd5c459c702fe7e706342e2935f57` | `chieff.width.logistic_q` |

`manifest.json` records the same hashes and paths.

## Models

1. **Truncated Student-t** (`chieff.family = truncated_student_t`):
   chi_eff ~ Student-t(loc=chi_mu, scale=chi_sigma, dof=chi_nu) truncated to
   [-1, 1], normalised exactly with the regularized incomplete beta function.
   Priors: chi_mu ~ U(-0.3, 0.3), chi_sigma ~ LU(0.03, 0.5) (inherited from the
   root), chi_nu ~ LU(1, 100). The Gaussian is the chi_nu -> infinity limit, so
   the root is *not* nested at a finite prior point (chi_nu = 100 differs from
   the Gaussian at O(1/nu) in the core and more in the far tails).

2. **q-dependent mixture weight** (`gaussian_mixture`,
   `fraction_dependence = logistic_q`, `q_pivot = 0.7`): weight of component 2
   is f(q) = sigmoid(logit_chi_fraction + chi_fraction_q_slope (q - 0.7)).
   Priors: chi_mu_1 ~ U(-0.5, 0.5) (bulk), chi_mu_2 ~ U(0, 1) (spinning
   component), chi_sigma_{1,2} ~ LU(0.02, 0.5), logit_chi_fraction ~ U(-8, 2),
   chi_fraction_q_slope ~ U(-10, 10). The components are not exchangeable, so no
   label canonicalisation is applied (identity parameterisation); the labels are
   fixed only by the asymmetric priors.

3. **Logistic width in q** (`truncated_gaussian`,
   `width_dependence = logistic_q`): log sigma = log chi_sigma +
   delta_log_chi_sigma * sigmoid((q_transition - q) / q_transition_width),
   i.e. broader at low q when delta > 0. Priors: delta_log_chi_sigma ~ U(-1, 4),
   q_transition ~ U(0.1, 1.0), q_transition_width ~ LU(0.01, 0.3).

These atoms live in `FOLLOWUP_MUTATIONS`, not `DEFAULT_MUTATIONS`; the frozen
graph and every existing model hash are unchanged
(`tests/test_followup_models.py`).
