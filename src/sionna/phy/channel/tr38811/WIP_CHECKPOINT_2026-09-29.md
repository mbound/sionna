# TR 38.811 PyTorch port — WIP checkpoint 2026-09-29

Branch: `feature/ntn-tr38811-pytorch`

This checkpoint is intentionally committed before the port is complete.

## Implemented

- spherical-Earth NTN slant-range geometry;
- free-space path loss;
- Doppler from radial range rate;
- Suburban LOS S-band parameter tables;
- TR 38.821 LEO S-band calibration adapter;
- explicit K-factor override for calibrated TDL-D;
- marginal Suburban LOS LSP sampling;
- 30-degree UL cross-LSP correlation;
- per-LSP spatial-correlation matrices;
- joint cross-LSP + spatial sampling;
- LOS cluster-delay generation;
- LOS diffuse/specular cluster-power allocation.

## Current scope

The current narrow calibration target is:
- LEO S-band;
- 2 GHz;
- Suburban LOS;
- 30-degree elevation;
- UL-focused LSP/correlation checkpoint;
- TR 38.821 link-level baseline compatibility.

## Not yet implemented

- cluster/ray angles;
- ray-offset generation/coupling;
- XPR and polarization matrices;
- satellite/UE antenna response;
- final channel coefficient generator;
- complete LOS/NLOS scenario wrapper;
- Urban/Dense Urban and Ka-band extensions;
- complete cross-version TR 38.811 table provenance.

All unfinished pieces remain intentionally on this feature branch rather than
being merged into Sionna main.
