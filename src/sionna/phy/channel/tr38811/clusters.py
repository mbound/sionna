"""TR 38.811 cluster delay/power generation for the Suburban LOS checkpoint.

This ports steps 5 and 6 of the TR 38.901 stochastic procedure as used by
OpenNTN, with parameters supplied by TR 38.811. It deliberately stops before
angle/ray generation so each stage can be statistically calibrated separately.
"""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class LosClusterSample:
    """Cluster delays and power allocations.

    diffuse_powers sums to one across clusters before application of the
    Rician K factor. powers_with_specular has the LOS specular power combined
    into the first cluster and sums to one.
    """

    delays_s: torch.Tensor
    unscaled_delays_s: torch.Tensor
    diffuse_powers: torch.Tensor
    powers_with_specular: torch.Tensor
    specular_power: torch.Tensor
    diffuse_total_power: torch.Tensor


def los_delay_scaling_from_k_db(k_factor_db: torch.Tensor) -> torch.Tensor:
    """TR 38.901 LOS delay-scaling polynomial."""
    k = torch.as_tensor(k_factor_db)
    return (
        0.7705
        - 0.0433 * k
        + 0.0002 * k * k
        + 0.000017 * k * k * k
    )


def sample_suburban_los_clusters(
    delay_spread_s,
    k_factor_db,
    *,
    num_clusters: int = 3,
    r_tau: float = 3.5,
    cluster_shadowing_std_db: float = 3.0,
    generator=None,
) -> LosClusterSample:
    """Sample Suburban-LOS cluster delays and powers.

    Inputs may be scalars or equal-shaped tensors. A cluster dimension is
    appended to the broadcast input shape.

    The algorithm follows OpenNTN/TR 38.901:
    1. draw U in (1e-6, 1);
    2. tau'_n = -r_tau * DS * ln(U_n);
    3. subtract minimum and sort;
    4. apply LOS K-dependent delay scaling;
    5. draw per-cluster log-normal shadowing zeta;
    6. normalize diffuse powers;
    7. apply K/(K+1) specular split to the first cluster.
    """
    ds = torch.as_tensor(delay_spread_s)
    k_db = torch.as_tensor(k_factor_db, dtype=ds.dtype, device=ds.device)
    ds, k_db = torch.broadcast_tensors(ds, k_db)

    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if torch.any(ds <= 0):
        raise ValueError("delay_spread_s must be positive")
    if r_tau <= 1.0:
        raise ValueError("r_tau must be > 1")
    if cluster_shadowing_std_db < 0.0:
        raise ValueError("cluster_shadowing_std_db must be non-negative")

    shape = (*ds.shape, int(num_clusters))
    u = torch.rand(
        shape,
        dtype=ds.dtype,
        device=ds.device,
        generator=generator,
    )
    u = torch.clamp(u, min=1e-6, max=1.0)

    ds_e = ds.unsqueeze(-1)
    unscaled = -float(r_tau) * ds_e * torch.log(u)
    unscaled = unscaled - torch.amin(unscaled, dim=-1, keepdim=True)
    unscaled, _ = torch.sort(unscaled, dim=-1)

    scale = los_delay_scaling_from_k_db(k_db).unsqueeze(-1)
    if torch.any(scale <= 0):
        raise ValueError("LOS delay-scaling polynomial became non-positive")
    delays = unscaled / scale

    z = torch.randn(
        shape,
        dtype=ds.dtype,
        device=ds.device,
        generator=generator,
    ) * float(cluster_shadowing_std_db)

    p_unnorm = (
        torch.exp(
            -unscaled
            * (float(r_tau) - 1.0)
            / (float(r_tau) * ds_e)
        )
        * torch.pow(
            torch.tensor(10.0, dtype=ds.dtype, device=ds.device),
            -z / 10.0,
        )
    )
    diffuse = p_unnorm / torch.sum(p_unnorm, dim=-1, keepdim=True)

    k_linear = torch.pow(
        torch.tensor(10.0, dtype=ds.dtype, device=ds.device),
        k_db / 10.0,
    )
    diffuse_total = 1.0 / (k_linear + 1.0)
    specular = k_linear / (k_linear + 1.0)

    combined = diffuse * diffuse_total.unsqueeze(-1)
    combined = combined.clone()
    combined[..., 0] = combined[..., 0] + specular

    return LosClusterSample(
        delays_s=delays,
        unscaled_delays_s=unscaled,
        diffuse_powers=diffuse,
        powers_with_specular=combined,
        specular_power=specular,
        diffuse_total_power=diffuse_total,
    )
