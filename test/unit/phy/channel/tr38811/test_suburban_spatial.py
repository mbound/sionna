import math
import torch

from sionna.phy.channel.tr38811 import (
    suburban_los_sband_ul_spatial_correlation_matrices,
)


def test_suburban_ul_spatial_correlation_distances():
    xy = torch.tensor([[0.0, 0.0], [30.0, 0.0]], dtype=torch.float64)
    c = suburban_los_sband_ul_spatial_correlation_matrices(xy)
    assert c.shape == (7, 2, 2)
    assert math.isclose(float(c[0, 0, 1]), math.exp(-1.0), abs_tol=1e-12)
    assert math.isclose(float(c[4, 0, 1]), math.exp(-30.0/12.0), abs_tol=1e-12)
    torch.testing.assert_close(
        torch.diagonal(c, dim1=-2, dim2=-1),
        torch.ones((7, 2), dtype=torch.float64),
    )
