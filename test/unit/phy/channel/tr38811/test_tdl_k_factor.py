import math
import torch

from sionna.phy.channel.tr38901 import TDL


def rms_delay(delays, powers):
    d = delays.detach().cpu().double().reshape(-1)
    p = powers.detach().cpu().double().reshape(-1)
    p = p / p.sum()
    mean = torch.sum(p * d)
    return float(torch.sqrt(torch.sum(p * (d - mean) ** 2)))


def test_tdl_d_default_k_factor_unchanged():
    ch = TDL(model="D", delay_spread=100e-9, carrier_frequency=2e9,
             device="cpu", spec_version="16.1")
    # Existing public property is the first-tap Rice factor.
    assert math.isclose(10.0 * math.log10(float(ch.k_factor)), 13.3,
                        abs_tol=1e-5)
    # TR 38.901 7.7.6 model K uses LOS / sum(all Rayleigh taps).
    assert math.isclose(float(ch.model_k_factor_db), 8.984647080037742,
                        abs_tol=1e-5)


def test_tdl_d_k_factor_override_matches_38901_776():
    ds = 100e-9
    ch = TDL(model="D", delay_spread=ds, carrier_frequency=2e9,
             k_factor_db=20.8, device="cpu", spec_version="16.1")
    assert math.isclose(float(ch.model_k_factor_db), 20.8, abs_tol=1e-5)
    assert math.isclose(rms_delay(ch.delays, ch.mean_powers), ds,
                        rel_tol=1e-6, abs_tol=1e-15)


def test_k_factor_override_rejects_nlos_and_fixed_profiles():
    try:
        TDL(model="A", delay_spread=100e-9, carrier_frequency=2e9,
            k_factor_db=10.0, device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("NLoS TDL-A must reject k_factor_db")

    try:
        TDL(model="D30", carrier_frequency=2e9,
            k_factor_db=10.0, device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("fixed TDL-D30 must reject k_factor_db")
