"""Link-level NTN channel adapters built from TR 38.811 parameters.

These classes intentionally model the multipath/fading portion only. Orbital
common Doppler, Doppler drift and service-link propagation delay are applied by
an external geometry/orbit oracle so that true and UE-believed states can be
varied independently.
"""

from typing import Optional

from sionna.phy.channel.tr38901 import TDL

from .suburban import suburban_los_sband_profile


class SuburbanLosTdl(TDL):
    r"""TR 38.821-style NTN TDL using TR 38.811 Suburban LOS parameters.

    The requested elevation is mapped to the nearest 10-degree TR 38.811 table
    bin. By default, delay spread and model K-factor are the corresponding
    mean Suburban LOS values. TDL-D is the default base profile.

    Satellite common Doppler/frequency drift is deliberately not included in
    this class. The inherited TDL Doppler describes the UE/multipath Jakes
    component; orbital Doppler is applied separately in the waveform path.
    """

    def __init__(
        self,
        carrier_frequency: float = 2.0e9,
        elevation_deg: float = 30.0,
        model: str = "D",
        delay_spread_s: Optional[float] = None,
        k_factor_db: Optional[float] = None,
        ue_speed_m_s: float = 3.0 / 3.6,
        num_sinusoids: int = 20,
        num_rx_ant: int = 2,
        num_tx_ant: int = 1,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        spec_version: str = "16.1",
    ) -> None:
        if model not in ("D", "E"):
            raise ValueError("SuburbanLosTdl requires scalable LoS TDL-D or TDL-E")
        if not 1.5e9 <= float(carrier_frequency) <= 4.0e9:
            raise ValueError(
                "Suburban LOS S-band parameterization is limited to 1.5-4 GHz"
            )

        self._ntn_profile = suburban_los_sband_profile(elevation_deg)
        self._ntn_elevation_deg = float(elevation_deg)

        if delay_spread_s is None:
            delay_spread_s = 10.0 ** self._ntn_profile.mu_log10_ds_s
        if k_factor_db is None:
            k_factor_db = self._ntn_profile.k_factor_mean_db

        super().__init__(
            model=model,
            delay_spread=float(delay_spread_s),
            carrier_frequency=float(carrier_frequency),
            num_sinusoids=num_sinusoids,
            min_speed=float(ue_speed_m_s),
            max_speed=float(ue_speed_m_s),
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            k_factor_db=float(k_factor_db),
            precision=precision,
            device=device,
            spec_version=spec_version,
        )

    @property
    def ntn_profile(self):
        """Selected TR 38.811 Suburban LOS parameter profile."""
        return self._ntn_profile

    @property
    def elevation_deg(self) -> float:
        """Requested elevation angle in degrees."""
        return self._ntn_elevation_deg
