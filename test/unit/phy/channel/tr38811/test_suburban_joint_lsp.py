import math
import torch

from sionna.phy.channel.tr38811 import (
    sample_suburban_los_sband_ul_joint_lsp,
)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a*b).mean() / (a.std(unbiased=False)*b.std(unbiased=False)))


def test_joint_lsp_spatial_ds_correlation():
    # Two UTs separated by the DS correlation distance (30 m).
    xy = torch.tensor([[0.0, 0.0], [30.0, 0.0]], dtype=torch.float64)
    g = torch.Generator(device="cpu").manual_seed(20260929)
    s = sample_suburban_los_sband_ul_joint_lsp(
        xy, 30000, generator=g, dtype=torch.float64, device="cpu"
    )
    # Native DS is log10(DS), so its target spatial correlation is exp(-1).
    rho = _corr(s.gaussian_native[:,0,0], s.gaussian_native[:,1,0])
    assert abs(rho - math.exp(-1.0)) < 0.025


def test_joint_lsp_shapes_and_angle_caps():
    xy = torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 1.0]], dtype=torch.float64)
    g = torch.Generator(device="cpu").manual_seed(4)
    s = sample_suburban_los_sband_ul_joint_lsp(
        xy, 100, generator=g, dtype=torch.float64, device="cpu"
    )
    assert s.gaussian_native.shape == (100, 3, 7)
    assert bool(torch.all(s.asd_deg <= 104.0))
    assert bool(torch.all(s.asa_deg <= 104.0))
    assert bool(torch.all(s.zsa_deg <= 52.0))
    assert bool(torch.all(s.zsd_deg <= 52.0))
