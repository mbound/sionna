import math
import torch

from sionna.phy.channel.tr38811 import (
    sample_suburban_los_sband_lsp,
    suburban_los_basic_pathloss_db,
)


def test_suburban_los_30deg_marginal_statistics():
    g = torch.Generator(device="cpu").manual_seed(1234)
    s = sample_suburban_los_sband_lsp(
        50000, 30.0, generator=g, dtype=torch.float64, device="cpu"
    )

    log_ds = torch.log10(s.delay_spread_s)
    assert abs(float(log_ds.mean()) - (-8.72)) < 0.015
    assert abs(float(log_ds.std(unbiased=True)) - 0.79) < 0.015

    assert abs(float(s.shadow_fading_db.mean())) < 0.03
    assert abs(float(s.shadow_fading_db.std(unbiased=True)) - 1.14) < 0.03

    assert abs(float(s.k_factor_db.mean()) - 20.8) < 0.15
    assert abs(float(s.k_factor_db.std(unbiased=True)) - 16.34) < 0.15


def test_suburban_los_basic_pathloss_matches_fspl_at_zero_sf():
    pl = suburban_los_basic_pathloss_db(
        992_778.3834972032, 2.0e9, 0.0, device="cpu"
    )
    assert math.isclose(float(pl), 158.4054293826943, rel_tol=0, abs_tol=1e-9)
