import json

import pytest

from gwpop_search.scouts import (
    ScoutCampaignConfig,
    default_scout_campaign_config,
    load_scout_campaign_config,
    save_scout_campaign_config,
)


@pytest.mark.parametrize(
    "target,covariate",
    [
        ("q", "m1_source"),
        ("q", "z"),
        ("chi_eff", "m1_source"),
        ("chi_eff", "q"),
        ("chi_eff", "z"),
    ],
)
def test_default_scout_config_roundtrip(target, covariate, tmp_path):
    config = default_scout_campaign_config(target, covariate)
    path = tmp_path / "scout.json"
    save_scout_campaign_config(path, config)
    restored = load_scout_campaign_config(path)

    assert restored == config
    payload = json.loads(path.read_text())
    assert payload["format_version"] == "gwpop-search-hsgp-scout-campaign-1.0"
    assert payload["hsgp"]["target"] == target
    assert payload["hsgp"]["covariate"] == covariate
    assert payload["run"]["nuts"]["num_chains"] == 4


def test_default_scout_config_rejects_invalid_target_covariate_pair():
    with pytest.raises(ValueError, match="unsupported 'q' covariate"):
        default_scout_campaign_config("q", "q")


def test_scout_config_rejects_unknown_format():
    config = default_scout_campaign_config("chi_eff", "q")
    payload = config.to_dict()
    payload["format_version"] = "future-version"
    with pytest.raises(ValueError, match="unsupported HSGP scout"):
        ScoutCampaignConfig.from_dict(payload)
