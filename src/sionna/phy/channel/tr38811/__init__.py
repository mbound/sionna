"""3GPP TR 38.811 non-terrestrial channel-model building blocks.

This package is an incremental PyTorch/Sionna-2.x port. The first checkpoint
contains deterministic LEO geometry helpers and the Suburban LOS S-band
large-scale-parameter table needed for the TR 38.821 LEO-S-band calibration
work. Frequency-selective channel generation will be added incrementally.
"""

from .geometry import (
    EARTH_RADIUS_M,
    doppler_hz,
    free_space_pathloss_db,
    slant_range_from_elevation,
)
from .suburban import (
    SuburbanLosSBandProfile,
    nearest_elevation_bin,
    suburban_los_sband_profile,
)

__all__ = [
    "EARTH_RADIUS_M",
    "doppler_hz",
    "free_space_pathloss_db",
    "slant_range_from_elevation",
    "SuburbanLosSBandProfile",
    "nearest_elevation_bin",
    "suburban_los_sband_profile",
]
