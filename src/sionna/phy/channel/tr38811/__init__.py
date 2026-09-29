"""3GPP TR 38.811 non-terrestrial channel-model building blocks."""

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
    SuburbanLosLspSample,
    sample_suburban_los_sband_lsp,
    suburban_los_basic_pathloss_db,
)
from .lls import Tr38821LeoSBandLlsConfig, LeoSBandLlsCalibration

__all__ = [
    "EARTH_RADIUS_M",
    "doppler_hz",
    "free_space_pathloss_db",
    "slant_range_from_elevation",
    "SuburbanLosSBandProfile",
    "nearest_elevation_bin",
    "suburban_los_sband_profile",
    "suburban_los_basic_pathloss_db",
    "sample_suburban_los_sband_lsp",
    "SuburbanLosLspSample",
    "Tr38821LeoSBandLlsConfig",
    "LeoSBandLlsCalibration",
]
