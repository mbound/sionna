import math
import torch

from sionna.phy.channel.tr38811.lls import SuburbanLosTdl


def rms_delay(delays, powers):
    d = delays.detach().cpu().double().reshape(-1)
    p = powers.detach().cpu().double().reshape(-1)
    p = p / p.sum()
    mean = torch.sum(p * d)
    return float(torch.sqrt(torch.sum(p * (d - mean) ** 2)))


def test_suburban_30deg_tdl_uses_ntn_mean_ds_and_k():
    ch = SuburbanLosTdl(device="cpu")
    target_ds = 10.0 ** -8.72
    assert ch.ntn_profile.elevation_deg == 30
    assert math.isclose(float(ch.model_k_factor_db), 20.8, abs_tol=1e-5)
    assert math.isclose(
        rms_delay(ch.delays, ch.mean_powers),
        target_ds,
        rel_tol=1e-6,
        abs_tol=1e-15,
    )


def test_suburban_tdl_keeps_orbital_doppler_external():
    ch = SuburbanLosTdl(device="cpu")
    assert math.isclose(float(ch._min_speed), 3.0 / 3.6, rel_tol=1e-6)
    assert math.isclose(float(ch._max_speed), 3.0 / 3.6, rel_tol=1e-6)


def test_suburban_sband_range_guard():
    try:
        SuburbanLosTdl(carrier_frequency=800e6, device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("S-band wrapper must reject 800 MHz")
