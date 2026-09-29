import math

from sionna.phy.channel.tr38811.lls import Tr38821LeoSBandLlsConfig


def test_tr38821_leo_sband_30deg_defaults():
    c = Tr38821LeoSBandLlsConfig()
    assert c.carrier_frequency_hz == 2.0e9
    assert c.elevation_deg == 30.0
    assert c.subcarrier_spacing_hz == 15.0e3
    assert math.isclose(c.ue_speed_m_s, 3.0 / 3.6)
    assert math.isclose(c.residual_frequency_error_hz, 200.0)
    assert math.isclose(c.mean_delay_spread_s, 10.0 ** -8.72)
    assert math.isclose(c.mean_k_factor_db, 20.8)
    assert c.tdl_model == "D"


def test_tr38821_builds_scalable_tdl_d():
    c = Tr38821LeoSBandLlsConfig()
    tdl = c.build_tdl(device="cpu")
    assert tdl.los
    assert math.isclose(float(tdl.delay_spread), c.mean_delay_spread_s, rel_tol=1e-6)
