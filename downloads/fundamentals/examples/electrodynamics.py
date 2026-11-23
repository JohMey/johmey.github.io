"""Harmonic magnetic diffusion in a copper sheet with two driven faces."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.linear.matrices.dense_linear_algebra import solve_dense

mu, sigma = 4e-7 * np.pi, 5.8e7  # H/m, S/m
half_width = 1e-3  # m
x = np.linspace(-half_width, half_width, 101)
h = x[1] - x[0]
n = len(x) - 2
identity = np.eye(n)
minus_laplacian = 2 * identity - np.eye(n, k=1) - np.eye(n, k=-1)
boundary = np.zeros(n)
boundary[[0, -1]] = 1.0  # unit field at each face
fig, ax = plt.subplots(figsize=(6.6, 3.5), layout="constrained")

for frequency in (1e3, 1e4, 1e5):
    omega = 2 * np.pi * frequency
    alpha = omega * mu * sigma * h**2
    # Real and imaginary parts of the same complex diffusion equation.
    matrix = np.block([[minus_laplacian, -alpha * identity],
                       [alpha * identity, minus_laplacian]])
    u, v = np.split(solve_dense(matrix, np.r_[boundary, np.zeros(n)]), 2)
    field = np.r_[1, u + 1j * v, 1]
    delta = np.sqrt(2 / (omega * mu * sigma))
    exact = np.cosh((1+1j) * x / delta) / np.cosh((1+1j) * half_width / delta)
    assert np.max(np.abs(field - exact)) < 8e-4
    print(f"{frequency:g} Hz: centre amplitude {abs(field[n//2+1]):.4f}")
    ax.plot(1000 * x, abs(field), label=f"{frequency/1000:g} kHz")
ax.set(xlabel="Position across sheet (mm)", ylabel="Field amplitude / surface amplitude")
ax.legend()
ax.grid(alpha=0.2)
fig.savefig("electrodynamics.png", dpi=180)
