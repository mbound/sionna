import torch

from sionna.phy.channel.tr38811 import (
    sample_suburban_los_sband_ul_correlated_lsp,
    suburban_los_sband_ul_correlation_matrix,
)


def test_correlated_suburban_ul_lsp_matches_target_correlation():
    g = torch.Generator(device="cpu").manual_seed(20260929)
    s = sample_suburban_los_sband_ul_correlated_lsp(
        40000, generator=g, dtype=torch.float64, device="cpu"
    )
    x = s.gaussian_native
    x = (x - x.mean(dim=0)) / x.std(dim=0, unbiased=True)
    empirical = (x.T @ x) / (x.shape[0] - 1)
    target = suburban_los_sband_ul_correlation_matrix(
        dtype=torch.float64, device="cpu"
    )
    torch.testing.assert_close(empirical, target, rtol=0, atol=0.035)


def test_correlated_sampler_rejects_unported_elevation():
    try:
        sample_suburban_los_sband_ul_correlated_lsp(1, elevation_deg=40.0)
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError")
