from sionna.phy.channel.tr38811 import (
    nearest_elevation_bin,
    suburban_los_sband_profile,
)


def test_elevation_bin_half_up():
    assert nearest_elevation_bin(25.0) == 30
    assert nearest_elevation_bin(34.9) == 30
    assert nearest_elevation_bin(35.0) == 40


def test_suburban_sband_30deg_calibration_profile():
    p = suburban_los_sband_profile(30.0)
    assert p.elevation_deg == 30
    assert p.los_probability == 0.919
    assert p.mu_log10_ds_s == -8.72
    assert p.sigma_log10_ds == 0.79
    assert p.k_factor_mean_db == 20.80
    assert p.k_factor_sigma_db == 16.34
    assert p.r_tau == 3.50
    assert p.num_clusters == 3
