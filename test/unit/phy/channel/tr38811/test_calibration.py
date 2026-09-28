import math

from sionna.phy.channel.tr38811.calibration import (
    LeoSBandLlsCalibration,
    ppm_to_hz,
)


def test_leo_sband_30deg_lls_manifest():
    c = LeoSBandLlsCalibration()
    m = c.manifest()
    assert m["carrier_frequency_hz"] == 2.0e9
    assert m["elevation_deg"] == 30.0
    assert m["subcarrier_spacings_hz"] == [15e3, 30e3]
    assert math.isclose(m["ue_speed_m_s"], 3.0 / 3.6)
    assert math.isclose(m["mean_delay_spread_s"], 10.0 ** -8.72)
    assert m["mean_k_factor_db"] == 20.80
    assert m["los_probability"] == 0.919
    assert m["residual_frequency_error_hz"] == 200.0


def test_tdl_baseline_uses_ntn_delay_spread_but_keeps_k_target_explicit():
    c = LeoSBandLlsCalibration()
    kw = c.tdl_baseline_kwargs()
    assert kw["model"] == "D"
    assert math.isclose(kw["delay_spread"], 10.0 ** -8.72)
    assert kw["carrier_frequency"] == 2.0e9
    assert kw["spec_version"] == "16.1"
    assert c.mean_k_factor_db == 20.8


def test_ppm_to_hz():
    assert ppm_to_hz(0.1, 2e9) == 200.0
