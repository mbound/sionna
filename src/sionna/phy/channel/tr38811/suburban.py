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
