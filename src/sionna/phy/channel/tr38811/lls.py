"""Helpers for the TR 38.821 NTN link-level calibration baseline.

The TR 38.821 LEO S-band baseline references the TR 38.901 TDL/CDL model with
delay/angular scaling factors and K factor taken from the mean Suburban LOS
parameters at 30 degree elevation. This module turns those assumptions into
explicit, reproducible Sionna configuration objects.

This is a calibration adapter, not a claim that TR 38.901 TDL-D alone is the
complete TR 38.811 NTN channel model.
"""

from dataclasses import dataclass
import math
from typing import Literal

from sionna.phy import SPEED_OF_LIGHT
from sionna.phy.channel.tr38901 import TDL

from .suburban import suburban_los_sband_profile


@dataclass(frozen=True)
class Tr38821LeoSBandLlsConfig:
    """TR 38.821 LEO S-band link-level baseline.

    Defaults follow the LEO S-band assumptions used in TR 38.821 section 6.1.2.
    """

    carrier_frequency_hz: float = 2.0e9
    elevation_deg: float = 30.0
    subcarrier_spacing_hz: float = 15.0e3
    ue_speed_m_s: float = 3.0 / 3.6
    residual_frequency_error_ppm: float = 0.1
    tdl_model: Literal["D", "A"] = "D"
    spec_version: str = "16.1"

    @property
    def suburban_profile(self):
        return suburban_los_sband_profile(self.elevation_deg)

    @property
    def mean_delay_spread_s(self) -> float:
        """Mean RMS delay spread implied by mu_log10(DS/s)."""
        return 10.0 ** self.suburban_profile.mu_log10_ds_s

    @property
    def mean_k_factor_db(self) -> float:
        return self.suburban_profile.k_factor_mean_db

    @property
    def mean_k_factor_linear(self) -> float:
        return 10.0 ** (self.mean_k_factor_db / 10.0)

    @property
    def residual_frequency_error_hz(self) -> float:
        return self.carrier_frequency_hz * self.residual_frequency_error_ppm * 1e-6

    @property
    def ue_max_doppler_hz(self) -> float:
        return self.carrier_frequency_hz * self.ue_speed_m_s / SPEED_OF_LIGHT

    @property
    def ue_speed_for_tdl_m_s(self) -> float:
        return self.ue_speed_m_s

    def build_tdl(self, *, device=None, precision=None) -> TDL:
        """Build the baseline Sionna TDL object.

        TDL-D is the LOS baseline explicitly referenced by the PRACH study and
        several TR 38.821 evaluations. TDL-A is retained as the documented
        NLOS/control alternative used in some NTN evaluations.

        For TDL-D, Sionna's scalable LoS model is explicitly re-weighted to
        the TR 38.811 Suburban mean K factor using its TR 38.901 Section 7.7.6
        K-factor override and RMS-delay re-normalization.
        """
        return TDL(
            model=self.tdl_model,
            delay_spread=self.mean_delay_spread_s,
            carrier_frequency=self.carrier_frequency_hz,
            min_speed=self.ue_speed_for_tdl_m_s,
            max_speed=self.ue_speed_for_tdl_m_s,
            k_factor_db=(self.mean_k_factor_db if self.tdl_model == "D" else None),
            precision=precision,
            device=device,
            spec_version=self.spec_version,
        )

    def as_dict(self) -> dict:
        p = self.suburban_profile
        return {
            "carrier_frequency_hz": self.carrier_frequency_hz,
            "elevation_deg": self.elevation_deg,
            "subcarrier_spacing_hz": self.subcarrier_spacing_hz,
            "ue_speed_m_s": self.ue_speed_m_s,
            "residual_frequency_error_ppm": self.residual_frequency_error_ppm,
            "residual_frequency_error_hz": self.residual_frequency_error_hz,
            "ue_max_doppler_hz": self.ue_max_doppler_hz,
            "tdl_model": self.tdl_model,
            "mean_delay_spread_s": self.mean_delay_spread_s,
            "mean_k_factor_db": self.mean_k_factor_db,
            "mean_k_factor_linear": self.mean_k_factor_linear,
            "los_probability": p.los_probability,
            "spec_version": self.spec_version,
        }


class LeoSBandLlsCalibration:
    """Backward-compatible calibration helper used by ntn-doppler-delay.

    It exposes explicit constructor kwargs plus a machine-readable manifest.
    """

    def __init__(self):
        self._config = Tr38821LeoSBandLlsConfig()

    @property
    def carrier_frequency_hz(self) -> float:
        return self._config.carrier_frequency_hz

    @property
    def elevation_deg(self) -> float:
        return self._config.elevation_deg

    @property
    def subcarrier_spacing_hz(self) -> float:
        return self._config.subcarrier_spacing_hz

    @property
    def ue_speed_m_s(self) -> float:
        return self._config.ue_speed_m_s

    @property
    def mean_delay_spread_s(self) -> float:
        return self._config.mean_delay_spread_s

    @property
    def mean_k_factor_db(self) -> float:
        return self._config.mean_k_factor_db

    @property
    def residual_frequency_error_hz(self) -> float:
        return self._config.residual_frequency_error_hz

    def tdl_baseline_kwargs(self, model: str = "D") -> dict:
        model = str(model).upper()
        if model not in ("D", "E"):
            raise ValueError("TR 38.821 LOS calibration adapter supports TDL-D/E")
        return {
            "model": model,
            "delay_spread": self.mean_delay_spread_s,
            "carrier_frequency": self.carrier_frequency_hz,
            "min_speed": self.ue_speed_m_s,
            "max_speed": self.ue_speed_m_s,
            "k_factor_db": self.mean_k_factor_db,
            "spec_version": "16.1",
        }

    def manifest(self) -> dict:
        out = self._config.as_dict()
        out.update(
            {
                "source": "3GPP TR 38.821 section 6.1.2",
                "channel_parameter_source": (
                    "3GPP TR 38.811 Suburban LOS, 30 degree elevation"
                ),
                "calibration_profile": "LEO S-band",
            }
        )
        return out
