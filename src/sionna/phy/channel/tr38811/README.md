# TR 38.811 PyTorch port — work in progress

This directory is the Sionna 2.1/PyTorch implementation branch for the NTN
channel models needed by mbound/ntn-doppler-delay.

Current checkpoint implements spherical-Earth LEO slant range, free-space path
loss, Doppler from radial range rate, selected TR 38.811 Suburban LOS S-band
large-scale parameters, and CPU-safe unit tests.

The reference implementation used for cross-checking is the MIT-licensed
OpenNTN project: https://github.com/ant-uni-bremen/OpenNTN

OpenNTN targets Sionna 1.x/TensorFlow. This branch re-implements the required
functionality against Sionna 2.1/PyTorch rather than importing TensorFlow code.

Still to implement: complete LOS/NLOS path loss and clutter loss, correlated
LSP generation, clusters/rays, channel coefficients, spatial consistency,
frequency-selective time-varying CIR, antenna patterns, and the other TR 38.811
scenarios/bands.
