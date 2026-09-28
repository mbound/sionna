"""Deterministic NTN geometry helpers for TR 38.811 models."""

from typing import Optional
import torch
from sionna.phy import SPEED_OF_LIGHT

EARTH_RADIUS_M = 6_371_000.0


def _as_real_tensor(value, *, dtype, device):
    return torch.as_tensor(value, dtype=dtype, device=device)


def slant_range_from_elevation(
    elevation_deg,
    satellite_altitude_m,
    earth_radius_m: float = EARTH_RADIUS_M,
    *,
    dtype: Optional[torch.dtype] = None,
    device=None,
) -> torch.Tensor:
    """Return spherical-Earth satellite slant range in metres."""
    dtype = dtype or torch.float64
    e_deg = _as_real_tensor(elevation_deg, dtype=dtype, device=device)
    h = _as_real_tensor(satellite_altitude_m, dtype=dtype, device=device)
    if torch.any((e_deg <= 0.0) | (e_deg > 90.0)):
        raise ValueError("elevation_deg must be in (0, 90]")
    if torch.any(h <= 0.0):
        raise ValueError("satellite_altitude_m must be positive")

    e = torch.deg2rad(e_deg)
    r = _as_real_tensor(earth_radius_m, dtype=dtype, device=device)
    rs = r + h
    return torch.sqrt(rs * rs - (r * torch.cos(e)) ** 2) - r * torch.sin(e)


def free_space_pathloss_db(
    distance_m,
    carrier_frequency_hz,
    *,
    dtype: Optional[torch.dtype] = None,
    device=None,
) -> torch.Tensor:
    """Return free-space path loss in dB."""
    dtype = dtype or torch.float64
    d = _as_real_tensor(distance_m, dtype=dtype, device=device)
    fc = _as_real_tensor(carrier_frequency_hz, dtype=dtype, device=device)
    if torch.any(d <= 0.0) or torch.any(fc <= 0.0):
        raise ValueError("distance and carrier frequency must be positive")
    c = _as_real_tensor(SPEED_OF_LIGHT, dtype=dtype, device=device)
    return 20.0 * torch.log10(4.0 * torch.pi * d * fc / c)


def doppler_hz(
    range_rate_m_s,
    carrier_frequency_hz,
    *,
    dtype: Optional[torch.dtype] = None,
    device=None,
) -> torch.Tensor:
    """Return propagation Doppler in Hz.

    Positive range rate means increasing separation and therefore negative
    Doppler: f_D = -(f_c/c) dR/dt.
    """
    dtype = dtype or torch.float64
    rr = _as_real_tensor(range_rate_m_s, dtype=dtype, device=device)
    fc = _as_real_tensor(carrier_frequency_hz, dtype=dtype, device=device)
    c = _as_real_tensor(SPEED_OF_LIGHT, dtype=dtype, device=device)
    return -(fc / c) * rr
