"""Golden pin of the legacy default synthetic survey (computed at commit e3c5b01).

The survey-v2 options must leave the default survey (uniform detector box,
truth-centered observation) exactly as it was before they existed: same random
stream consumption, same arrays, same metadata and manifest payloads. The
values below were produced by the pre-v2 code (e3c5b01) with the pinned
environment; a failure means the legacy data a campaign would regenerate from
its seed has changed. The tolerance only absorbs last-bit differences between
platforms' transcendental functions, not any change of draws.
"""

import numpy as np

from gwpop_search.inference.synthetic import (
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.nulls.campaign import ExactNullCampaignConfig

SEED = 20260919
CONFIG = {
    "n_events": 6,
    "posterior_samples_per_event": 8,
    "n_injections": 400,
    "population_batch_size": 256,
    "redshift_sampling_grid": 1024,
}

# name: (size, sum, first three, last three)
GOLDEN_TRUTHS = {
    "m1_source": (6, 191.78287818339493, [9.36688684455998, 38.83258227280912, 33.52132622654441], [35.34668196045226, 35.487430526217786, 39.22797035281138]),
    "q": (6, 4.616112199811868, [0.9820554641021875, 0.7868501073237323, 0.5266320129915749], [0.7116921516025212, 0.656127964255325, 0.9527544995365275]),
    "z": (6, 6.81144672647487, [0.5709679730077335, 1.787008371830404, 1.2005213019181553], [0.7196718284096232, 1.4992175113108221, 1.0340597399981315]),
    "chi_eff": (6, -0.6191811049156268, [-0.14371692123069363, -0.0949686201919321, -0.08920549722764064], [-0.28478571640316497, 0.3350651901675966, -0.34156954002979206]),
    "luminosity_distance": (6, 48625.34199899218, [3423.3884615286775, 13886.177864981546, 8510.233167808148], [4534.892650146732, 11191.843404343557, 7078.80645018352]),
    "m1_detector": (6, 425.97373973794987, [14.715079239591198, 108.22673189411195, 73.76439243005872], [60.78469319514438, 88.6908078025497, 79.79203517649393]),
    "ra": (6, 17.590900651646994, [6.192732644615222, 0.03771947912450415, 1.5834016742573571], [5.25039773291414, 3.4402536071470253, 1.086395513588745]),
    "dec": (6, 1.5473912177632836, [-1.3691485608445149, 0.8488223198424109, 0.25614091942458034], [-0.720371162733856, 1.182136067471943, 1.3498116346027202]),
}
GOLDEN_PE = {
    "m1_detector": (48, 3429.463451980966, [15.240835099693978, 12.86124141918685, 16.25247880142303], [86.90181712712241, 64.6504624552058, 79.21348670319888]),
    "q": (48, 36.94746707173563, [0.9962154968462821, 0.9576889472426687, 0.8811578782757358], [0.989883272509831, 0.8484462221893906, 0.9713047133812801]),
    "luminosity_distance": (48, 392584.68181574676, [3537.5802636956655, 3516.3376215199905, 2915.7947082382443], [5047.192647579798, 7640.885035577565, 7362.208534222252]),
    "chi_eff": (48, -4.706403764194653, [-0.11874448161995146, -0.26942440436591153, -0.31716993118978193], [-0.4094647570987288, -0.34419924117141126, -0.20248337389299892]),
    "ra": (48, 140.72720521317595, [6.192732644615222, 6.192732644615222, 6.192732644615222], [1.086395513588745, 1.086395513588745, 1.086395513588745]),
    "dec": (48, 12.379129742106269, [-1.3691485608445149, -1.3691485608445149, -1.3691485608445149], [1.3498116346027202, 1.3498116346027202, 1.3498116346027202]),
}
GOLDEN_SELECTION = {
    "m1_detector": (309, 76085.11441059675, [313.0226800902434, 322.8099150245291, 203.626569449717], [256.81304027498504, 292.3609038229036, 199.57285110560343]),
    "q": (309, 175.19416562856617, [0.8571483679679924, 0.1798180211464943, 0.2922739012134739], [0.3496053518635301, 0.4617798095846124, 0.847110815098499]),
    "luminosity_distance": (309, 2812408.0327765024, [15906.047300539441, 8629.967510655784, 8821.187618089869], [18477.572381223283, 18481.8117246626, 6070.3639982818395]),
    "chi_eff": (309, -2.905464836608062, [0.39571925330340485, 0.11196611013533331, -0.9776106991557381], [-0.3836669654011191, -0.6820425082557462, -0.20522701735266158]),
    "ra": (309, 984.3601009174156, [3.230103571975827, 4.189118375859034, 0.9647653265680126], [2.6437249896532715, 4.828770908724688, 3.4357565811140303]),
    "dec": (309, -5.948970418012483, [0.7495375106379747, 0.1256819620801889, -0.9965962191431721], [0.19554806488956214, -0.41450039465850275, -0.2521943173897313]),
}
GOLDEN_LOG_BOX_DENSITY = -19.107070156303983
GOLDEN_N_SELECTED = 309

LEGACY_SURVEY_PAYLOAD = {
    "n_events": 48,
    "posterior_samples_per_event": 256,
    "n_injections": 20_000,
    "observing_time_yr": 1.0,
    "reference_chirp_mass": 20.0,
    "reference_horizon_mpc": 5_000.0,
    "m1_detector_min": 2.0,
    "m1_detector_max": 400.0,
    "pe_m1_fractional_sigma": 0.08,
    "pe_q_sigma": 0.06,
    "pe_d_l_fractional_sigma": 0.12,
    "pe_chi_eff_sigma": 0.12,
    "population_batch_size": 2048,
    "redshift_sampling_grid": 8192,
}


def _check(values, golden):
    size, total, head, tail = golden
    values = np.asarray(values, dtype=float)
    assert values.size == size
    np.testing.assert_allclose(values.sum(), total, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(values[:3], head, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(values[-3:], tail, rtol=1e-12, atol=0.0)


def test_legacy_default_survey_reproduces_the_pre_v2_draws():
    dataset = generate_baseline_synthetic_dataset(seed=SEED, config=SyntheticSurveyConfig(**CONFIG))
    assert getattr(dataset, "event_observations", None) is None
    for name, golden in GOLDEN_TRUTHS.items():
        _check(dataset.event_truths[name], golden)
    for name, golden in GOLDEN_PE.items():
        _check(dataset.posterior.samples[name], golden)
    for name, golden in GOLDEN_SELECTION.items():
        _check(dataset.selection.samples[name], golden)

    assert dataset.posterior.offsets.tolist() == list(range(0, 49, 8))
    np.testing.assert_allclose(dataset.posterior.log_ref_density, GOLDEN_LOG_BOX_DENSITY, rtol=1e-14)
    np.testing.assert_allclose(dataset.selection.log_draw_density, GOLDEN_LOG_BOX_DENSITY, rtol=1e-14)
    assert dataset.posterior.metadata == {
        "fixture": "phase3-synthetic-pe",
        "reference_prior": "uniform detector basis",
    }
    assert dataset.selection.n_selected == GOLDEN_N_SELECTED
    assert dataset.selection.metadata == {
        "fixture": "phase3-synthetic-selection",
        "n_detected": GOLDEN_N_SELECTED,
    }
    assert [campaign.to_dict() for campaign in dataset.selection.campaigns] == [
        {
            "campaign_id": "SYNTH",
            "n_draw": 400,
            "observing_time_yr": 1.0,
            "metadata": {"detection_rule": "chirp_mass_scaled_reach"},
        }
    ]


def test_legacy_default_manifest_payloads_are_unchanged():
    config = SyntheticSurveyConfig()
    assert config.to_dict() == LEGACY_SURVEY_PAYLOAD
    assert tuple(config.to_dict()) == tuple(LEGACY_SURVEY_PAYLOAD)
    assert config.dataset_identity_suffix() == ""
    null = ExactNullCampaignConfig().to_dict()
    assert null["survey"] == LEGACY_SURVEY_PAYLOAD
    assert tuple(null) == (
        "format_version",
        "n_nulls",
        "root_seed",
        "survey",
        "truth_hyperparameters",
        "data_mode",
        "min_resampling_ess",
        "max_gpu_hours_per_null",
    )
