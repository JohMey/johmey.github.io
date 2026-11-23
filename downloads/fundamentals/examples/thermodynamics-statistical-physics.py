"""Finite periodic Ising chain: spin correlations from a transfer matrix."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.linear.matrices.dense_linear_algebra import symmetric_eigensystem

coupling_over_kT = 0.90
spin = np.array([1.0, -1.0])
transfer = np.exp(coupling_over_kT * np.outer(spin, spin))
eigenvalues, vectors = symmetric_eigensystem(transfer)
ratios = eigenvalues / eigenvalues[-1]  # common scale cancels from the correlation
spin_in_modes = vectors.T @ np.diag(spin) @ vectors
u = np.tanh(coupling_over_kT)
fig, ax = plt.subplots(figsize=(6.6, 3.5), layout="constrained")
for length in (10, 24):
    distances = np.arange(length + 1)
    values = []
    for r in distances:
        numerator = np.sum(spin_in_modes**2 * ratios[:, None]**r * ratios[None, :]**(length-r))
        values.append(numerator / np.sum(ratios**length))
    exact = (u**distances + u**(length-distances)) / (1 + u**length)
    assert np.allclose(values, exact)
    print(f"{length} sites: antipodal correlation {values[length//2]:.4f}")
    ax.plot(distances, values, "o-", markersize=3, label=f"Ring of {length} sites")
ax.plot(distances, u**distances, "--", label="Infinite chain")
ax.set(xlabel="Separation (lattice spacings)", ylabel="Spin correlation")
ax.grid(alpha=0.2)
ax.legend()
fig.savefig("thermodynamics-statistical-physics.png", dpi=180)
