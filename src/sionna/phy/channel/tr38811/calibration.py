"""TR 38.821 link-level calibration helpers for NTN S-band.

The baseline follows TR 38.821 V16.0.0, clause 6.1.2:
- 2 GHz S-band;
- 15 or 30 kHz SCS;
- LEO elevation 30 degrees;
- baseline TR 38.901 TDL/CDL model scaled with the mean suburban-LOS
  delay/angular spreads and mean K-factor from TR 38.811;
- 3 km/h UE speed;
- satellite Doppler/frequency drift applied separately from the fading-tap
  Doppler spectrum.

The selected 30-degree Suburban LOS large-scale parameters are supplied by
:mod:`sionna.phy.channel.tr38811.suburban`.
"""

from dataclasses import dataclass
import math

from .suburban import suburban_los_sband_profile


@dataclass(frozen=True)
class LeoSBandLlsCalibration:
    carrier_frequency_hz: float = 2.0e9
    elevation_deg: float = 30.0
    subcarrier_spacings_hz: tuple[float, float] = (15e3, 30e3)
    ue_speed_m_s: float = 3.0 / 3.6
    ue_crystal_accuracy_ppm: float = 10.0
    residual_frequency_error_ppm: float = 0.1

    @property
    def profile(self):
        return suburban_los_sband_profile(self.elevation_deg)

    @property
    def mean_delay_spread_s(self) -> float:
        """Median/nominal DS represented by the tabulated log10 mean."""
        return 10.0 ** self.profile.mu_log10_ds_s

    @property
    def mean_asa_deg(self) -> float:
        return 10.0 ** self.profile.mu_log10_asa_deg

    @property
    def mean_zsa_deg(self) -> float:
        return 10.0 ** self.profile.mu_log10_zsa_deg

    @property
    def mean_k_factor_db(self) -> float:
        return self.profile.k_factor_mean_db

    @property
    def max_residual_frequency_error_hz(self) -> float:
        return self.carrier_frequency_hz * self.residual_frequency_error_ppm * 1e-6

    def tdl_baseline_kwargs(self, model: str = "D") -> dict:
        """Return the direct Sionna TDL constructor subset for the LLS baseline.

        TR 38.821 references a baseline TDL/CDL model with the NTN mean delay
        spread and K-factor. Current Sionna TDL-D has a fixed profile K-factor;
        therefore this helper sets the delay spread and records the target
        K-factor separately. A dedicated K-factor override will only be added
        after locking its exact interpretation to the source RAN1 LLS TDocs.
        """
        return {
            "model": model,
            "delay_spread": self.mean_delay_spread_s,
            "carrier_frequency": self.carrier_frequency_hz,
            "min_speed": self.ue_speed_m_s,
            "max_speed": self.ue_speed_m_s,
            "spec_version": "16.1",
        }

    def manifest(self) -> dict:
        p = self.profile
        return {
            "reference": "3GPP TR 38.821 V16.0.0 clause 6.1.2",
            "channel_parameter_reference": "3GPP TR 38.811 / OpenNTN cross-check",
            "carrier_frequency_hz": self.carrier_frequency_hz,
            "elevation_deg": self.elevation_deg,
            "subcarrier_spacings_hz": list(self.subcarrier_spacings_hz),
            "ue_speed_m_s": self.ue_speed_m_s,
            "ue_crystal_accuracy_ppm": self.ue_crystal_accuracy_ppm,
            "residual_frequency_error_ppm": self.residual_frequency_error_ppm,
            "residual_frequency_error_hz": self.max_residual_frequency_error_hz,
            "mean_delay_spread_s": self.mean_delay_spread_s,
            "mean_asa_deg": self.mean_asa_deg,
            "mean_zsa_deg": self.mean_zsa_deg,
            "mean_k_factor_db": self.mean_k_factor_db,
            "los_probability": p.los_probability,
            "num_clusters_tr38811": p.num_clusters,
            "r_tau": p.r_tau,
        }


def ppm_to_hz(ppm: float, carrier_frequency_hz: float) -> float:
    return float(ppm) * 1e-6 * float(carrier_frequency_hz)


def speed_for_max_doppler_hz(doppler_hz: float, carrier_frequency_hz: float) -> float:
    """Equivalent terrestrial speed for a desired maximum Jakes Doppler."""
    c = 299_792_458.0
    return abs(float(doppler_hz)) * c / float(carrier_frequency_hz)
