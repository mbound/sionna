import torch

from sionna.phy.channel.tr38811 import (
    doppler_hz,
    free_space_pathloss_db,
    slant_range_from_elevation,
)


def test_zenith_slant_range_equals_altitude():
    d = slant_range_from_elevation(90.0, 550_000.0)
    torch.testing.assert_close(d, torch.tensor(550_000.0, dtype=torch.float64))


def test_550km_30deg_slant_range():
    d = slant_range_from_elevation(30.0, 550_000.0)
    torch.testing.assert_close(
        d,
        torch.tensor(992_778.3834972032, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-6,
    )


def test_fspl_2ghz_30deg_550km():
    d = slant_range_from_elevation(30.0, 550_000.0)
    pl = free_space_pathloss_db(d, 2.0e9)
    torch.testing.assert_close(
        pl,
        torch.tensor(158.4054293826943, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-9,
    )


def test_doppler_sign_convention():
    f = doppler_hz(5647.640079761401, 2.0e9)
    torch.testing.assert_close(
        f,
        torch.tensor(-37676.99906420862, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-9,
    )
