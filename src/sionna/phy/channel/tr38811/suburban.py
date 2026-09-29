"""TR 38.811 Suburban LOS S-band calibration parameters.

This first porting checkpoint intentionally exposes parameter data without yet
claiming a complete TR 38.811 stochastic channel implementation.

The values are cross-checked against the MIT-licensed OpenNTN implementation
(Sub_Urban_LOS_S_band_*.json). They still need clause/table provenance locked
to the exact TR 38.811 revision used by our calibration manifest.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SuburbanLosSBandProfile:
    elevation_deg: int
    los_probability: float
    mu_log10_ds_s: float
    sigma_log10_ds: float
    mu_log10_asa_deg: float
    sigma_log10_asa: float
    mu_log10_zsa_deg: float
    sigma_log10_zsa: float
    shadow_fading_sigma_db: float
    k_factor_mean_db: float
    k_factor_sigma_db: float
    r_tau: float
    num_clusters: int


_TABLE = {
    10: (-8.16, 0.99,  0.05, 1.84, -1.78, 0.62, 1.79, 11.40,  6.26, 2.20, 3, 0.782),
    20: (-8.56, 0.96, -0.38, 1.94, -1.84, 0.81, 1.14, 19.45, 10.32, 3.36, 3, 0.869),
    30: (-8.72, 0.79, -0.56, 1.75, -1.67, 0.57, 1.14, 20.80, 16.34, 3.50, 3, 0.919),
    40: (-8.71, 0.81, -0.59, 1.82, -1.59, 0.86, 0.92, 21.20, 15.63, 2.81, 3, 0.929),
    50: (-8.72, 1.12, -0.58, 1.87, -1.55, 1.05, 1.42, 21.60, 14.22, 2.39, 3, 0.935),
    60: (-8.66, 1.23, -0.55, 1.92, -1.51, 1.23, 1.56, 19.75, 14.19, 2.73, 3, 0.940),
    70: (-8.38, 0.55, -0.28, 1.16, -1.27, 0.54, 0.85, 12.00,  5.70, 2.07, 2, 0.949),
    80: (-8.34, 0.63, -0.17, 1.09, -1.28, 0.67, 0.72, 12.85,  9.91, 2.04, 2, 0.952),
    90: (-8.34, 0.63, -0.17, 1.09, -1.28, 0.67, 0.72, 12.85,  9.91, 2.04, 2, 0.998),
}


def nearest_elevation_bin(elevation_deg: float) -> int:
    """Return the nearest TR-table elevation bin in 10-degree increments."""
    e = float(elevation_deg)
    if not 5.0 <= e <= 90.0:
        raise ValueError("elevation_deg must be between 5 and 90 degrees")
    b = int(math.floor((e + 5.0) / 10.0) * 10)
    return min(90, max(10, b))


def suburban_los_sband_profile(elevation_deg: float) -> SuburbanLosSBandProfile:
    """Return selected Suburban LOS S-band LSP parameters."""
    b = nearest_elevation_bin(elevation_deg)
    (
        mu_ds, sigma_ds, mu_asa, sigma_asa, mu_zsa, sigma_zsa,
        sigma_sf, mu_k, sigma_k, r_tau, n_clusters, los_p,
    ) = _TABLE[b]
    return SuburbanLosSBandProfile(
        elevation_deg=b,
        los_probability=los_p,
        mu_log10_ds_s=mu_ds,
        sigma_log10_ds=sigma_ds,
        mu_log10_asa_deg=mu_asa,
        sigma_log10_asa=sigma_asa,
        mu_log10_zsa_deg=mu_zsa,
        sigma_log10_zsa=sigma_zsa,
        shadow_fading_sigma_db=sigma_sf,
        k_factor_mean_db=mu_k,
        k_factor_sigma_db=sigma_k,
        r_tau=r_tau,
        num_clusters=n_clusters,
    )


@dataclass(frozen=True)
class SuburbanLosLspSample:
    """Marginal Suburban-LOS large-scale-parameter realization."""

    delay_spread_s: "torch.Tensor"
    asa_deg: "torch.Tensor"
    zsa_deg: "torch.Tensor"
    shadow_fading_db: "torch.Tensor"
    k_factor_db: "torch.Tensor"


def sample_suburban_los_sband_lsp(
    sample_shape,
    elevation_deg: float,
    *,
    generator=None,
    dtype=None,
    device=None,
) -> SuburbanLosLspSample:
    """Sample the *marginal* TR 38.811 Suburban-LOS S-band LSPs.

    This checkpoint validates the one-dimensional distributions before the full
    TR 38.811 cross-correlation machinery is ported. DS/ASA/ZSA are log-normal
    using the tabulated base-10 log means/stds; SF and K are normal in dB.
    """
    import torch

    p = suburban_los_sband_profile(elevation_deg)
    dtype = dtype or torch.float64
    shape = tuple(sample_shape) if not isinstance(sample_shape, int) else (sample_shape,)

    def z():
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    ten = lambda v: torch.tensor(float(v), dtype=dtype, device=device)
    ds = torch.pow(ten(10.0), ten(p.mu_log10_ds_s) + ten(p.sigma_log10_ds) * z())
    asa = torch.pow(ten(10.0), ten(p.mu_log10_asa_deg) + ten(p.sigma_log10_asa) * z())
    zsa = torch.pow(ten(10.0), ten(p.mu_log10_zsa_deg) + ten(p.sigma_log10_zsa) * z())
    sf = ten(p.shadow_fading_sigma_db) * z()
    k = ten(p.k_factor_mean_db) + ten(p.k_factor_sigma_db) * z()

    return SuburbanLosLspSample(
        delay_spread_s=ds,
        asa_deg=asa,
        zsa_deg=zsa,
        shadow_fading_db=sf,
        k_factor_db=k,
    )


def suburban_los_basic_pathloss_db(
    distance_m,
    carrier_frequency_hz: float,
    shadow_fading_db=0.0,
    *,
    dtype=None,
    device=None,
):
    """TR 38.811 LOS basic path loss: free-space loss plus shadow fading.

    Additional atmospheric/scintillation losses are deliberately separate.
    """
    import torch
    from .geometry import free_space_pathloss_db

    pl = free_space_pathloss_db(
        distance_m,
        carrier_frequency_hz,
        dtype=dtype,
        device=device,
    )
    sf = torch.as_tensor(shadow_fading_db, dtype=pl.dtype, device=pl.device)
    return pl + sf


# Cross-LSP correlation matrix for TR 38.811 Suburban LOS S-band UL.
# Order: DS, ASD, ASA, SF, K, ZSA, ZSD.
_SUBURBAN_LOS_S_UL_CORR = (
    ( 1.0,  0.4,  0.8, -0.4, -0.4,  0.0, -0.2),
    ( 0.4,  1.0,  0.0, -0.5,  0.0,  0.0,  0.5),
    ( 0.8,  0.0,  1.0, -0.5, -0.2,  0.4, -0.3),
    (-0.4, -0.5, -0.5,  1.0,  0.0, -0.8,  0.0),
    (-0.4,  0.0, -0.2,  0.0,  1.0,  0.0,  0.0),
    ( 0.0,  0.0,  0.4, -0.8,  0.0,  1.0,  0.0),
    (-0.2,  0.5, -0.3,  0.0,  0.0,  0.0,  1.0),
)

# The current correlation checkpoint is deliberately locked to the exact
# 30-degree UL calibration point used by the LEO S-band PUSCH work.
_SUBURBAN_LOS_S_UL_30_MEAN = (-8.72, -3.77, -0.56, 0.0, 20.80, -1.67, -1.28)
_SUBURBAN_LOS_S_UL_30_STD = (0.79, 1.72, 1.75, 1.14, 16.34, 0.57, 0.49)


@dataclass(frozen=True)
class SuburbanLosCorrelatedLspSample:
    """Correlated UL LSP realization in the native Gaussian/log domains."""

    gaussian_native: "torch.Tensor"
    delay_spread_s: "torch.Tensor"
    asd_deg: "torch.Tensor"
    asa_deg: "torch.Tensor"
    shadow_fading_db: "torch.Tensor"
    k_factor_db: "torch.Tensor"
    zsa_deg: "torch.Tensor"
    zsd_deg: "torch.Tensor"


def suburban_los_sband_ul_correlation_matrix(*, dtype=None, device=None):
    """Return the 7x7 cross-LSP correlation matrix."""
    import torch
    dtype = dtype or torch.float64
    return torch.tensor(_SUBURBAN_LOS_S_UL_CORR, dtype=dtype, device=device)


def sample_suburban_los_sband_ul_correlated_lsp(
    sample_shape,
    elevation_deg: float = 30.0,
    *,
    generator=None,
    dtype=None,
    device=None,
) -> SuburbanLosCorrelatedLspSample:
    """Sample correlated Suburban-LOS S-band UL LSPs at 30 degrees.

    The Gaussian vector order is DS, ASD, ASA, SF, K, ZSA, ZSD. DS/ASD/ASA/
    ZSA/ZSD are subsequently exponentiated base-10; SF and K remain in dB.

    This implements cross-LSP correlation at one location. Spatial correlation
    across multiple UT positions is a later checkpoint and is not implied here.
    """
    import torch

    if abs(float(elevation_deg) - 30.0) > 1e-12:
        raise NotImplementedError(
            "correlated UL checkpoint currently supports 30 degree elevation"
        )
    dtype = dtype or torch.float64
    shape = tuple(sample_shape) if not isinstance(sample_shape, int) else (sample_shape,)
    n = 1
    for dim in shape:
        n *= int(dim)

    corr = suburban_los_sband_ul_correlation_matrix(dtype=dtype, device=device)
    chol = torch.linalg.cholesky(corr)
    z = torch.randn((n, 7), dtype=dtype, device=device, generator=generator)
    zc = z @ chol.T

    mean = torch.tensor(
        _SUBURBAN_LOS_S_UL_30_MEAN, dtype=dtype, device=device
    )
    std = torch.tensor(
        _SUBURBAN_LOS_S_UL_30_STD, dtype=dtype, device=device
    )
    native = mean + zc * std
    native = native.reshape(*shape, 7)

    return SuburbanLosCorrelatedLspSample(
        gaussian_native=native,
        delay_spread_s=torch.pow(10.0, native[..., 0]),
        asd_deg=torch.pow(10.0, native[..., 1]),
        asa_deg=torch.pow(10.0, native[..., 2]),
        shadow_fading_db=native[..., 3],
        k_factor_db=native[..., 4],
        zsa_deg=torch.pow(10.0, native[..., 5]),
        zsd_deg=torch.pow(10.0, native[..., 6]),
    )


_SUBURBAN_LOS_S_UL_30_CORR_DISTANCE_M = (30.0, 18.0, 15.0, 37.0, 12.0, 15.0, 15.0)


def suburban_los_sband_ul_spatial_correlation_matrices(
    ut_xy_m,
    elevation_deg: float = 30.0,
    *,
    dtype=None,
    device=None,
):
    """Return one spatial-correlation matrix per UL LSP.

    Output order is DS, ASD, ASA, SF, K, ZSA, ZSD and output shape is
    [7, num_ut, num_ut]. Uses C_ij = exp(-d_ij / D_X).
    """
    import torch
    from sionna.phy.channel.tr38901.spatial_consistency import (
        spatial_consistency_correlation_matrix,
    )

    if abs(float(elevation_deg) - 30.0) > 1e-12:
        raise NotImplementedError(
            "spatial UL checkpoint currently supports 30 degree elevation"
        )
    dtype = dtype or torch.float64
    xy = torch.as_tensor(ut_xy_m, dtype=dtype, device=device)
    if xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError("ut_xy_m must have shape [num_ut, 2]")
    delta = xy[:, None, :] - xy[None, :, :]
    distance = torch.linalg.vector_norm(delta, dim=-1)

    precision = "double" if dtype == torch.float64 else "single"
    out = []
    for d_corr in _SUBURBAN_LOS_S_UL_30_CORR_DISTANCE_M:
        out.append(
            spatial_consistency_correlation_matrix(
                distance,
                d_corr,
                precision=precision,
                device=xy.device,
            )
        )
    return torch.stack(out, dim=0)
