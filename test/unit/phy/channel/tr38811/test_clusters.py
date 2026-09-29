import math
import torch

from sionna.phy.channel.tr38811 import (
    los_delay_scaling_from_k_db,
    sample_suburban_los_clusters,
)


def test_los_delay_scaling_20p8db():
    k = torch.tensor(20.8, dtype=torch.float64)
    s = los_delay_scaling_from_k_db(k)
    expected = 0.7705 - 0.0433*20.8 + 0.0002*20.8**2 + 0.000017*20.8**3
    assert math.isclose(float(s), expected, abs_tol=1e-14)


def test_suburban_los_cluster_invariants():
    g = torch.Generator(device="cpu").manual_seed(7)
    ds = torch.full((1000,), 10.0**-8.72, dtype=torch.float64)
    k = torch.full((1000,), 20.8, dtype=torch.float64)
    s = sample_suburban_los_clusters(ds, k, generator=g)

    assert s.delays_s.shape == (1000, 3)
    torch.testing.assert_close(
        s.delays_s[:, 0], torch.zeros(1000, dtype=torch.float64)
    )
    assert bool(torch.all(s.delays_s[:, 1:] >= s.delays_s[:, :-1]))
    torch.testing.assert_close(
        s.diffuse_powers.sum(dim=-1),
        torch.ones(1000, dtype=torch.float64),
        rtol=1e-12, atol=1e-12,
    )
    torch.testing.assert_close(
        s.powers_with_specular.sum(dim=-1),
        torch.ones(1000, dtype=torch.float64),
        rtol=1e-12, atol=1e-12,
    )
    expected_spec = 10.0**(20.8/10.0) / (1.0 + 10.0**(20.8/10.0))
    assert math.isclose(float(s.specular_power[0]), expected_spec, rel_tol=1e-12)


def test_cluster_sampler_scalar_inputs():
    g = torch.Generator(device="cpu").manual_seed(1)
    s = sample_suburban_los_clusters(
        torch.tensor(10.0**-8.72, dtype=torch.float64),
        torch.tensor(20.8, dtype=torch.float64),
        generator=g,
    )
    assert s.delays_s.shape == (3,)
