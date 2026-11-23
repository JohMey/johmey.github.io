"""Rayleigh–Ritz bounds for an electron in a tilted infinite square well."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.linear.matrices.dense_linear_algebra import symmetric_eigenvalues

width_nm = 10.0
hbar, mass, electronvolt = 1.054571817e-34, 9.1093837e-31, 1.602176634e-19
energy_unit = 1e3 * np.pi**2 * hbar**2 / (2 * mass * (width_nm * 1e-9)**2 * electronvolt)
tilt = 12.0  # end-to-end potential rise in units of the untilted ground energy
basis = np.arange(1, 65)
i, j = basis[:, None], basis[None, :]
position = np.eye(len(basis)) / 2  # matrix of x/L in the sine basis
odd = (i - j) % 2 == 1
position[odd] = 2 / np.pi**2 * (1 / (i+j)[odd]**2 - 1 / (i-j)[odd]**2)
hamiltonian = np.diag(basis**2) + tilt * position
reference = symmetric_eigenvalues(hamiltonian)[:2] * energy_unit
sizes = np.arange(2, 13)
energies = np.array([symmetric_eigenvalues(hamiltonian[:n, :n])[:2] for n in sizes]) * energy_unit
errors = energies - reference
assert np.all(errors > 0) and np.all(np.diff(energies, axis=0) <= 1e-10)
print("64-state reference energies (meV):", reference.round(4))
fig, ax = plt.subplots(figsize=(6.6, 3.5), layout="constrained")
for index, label in enumerate(("Ground state", "First excited state")):
    ax.semilogy(sizes, errors[:, index], "o-", label=label)
ax.set(xlabel="Number of sine functions", ylabel="Excess above 64-state reference (meV)")
ax.grid(alpha=0.2)
ax.legend()
fig.savefig("quantum-mechanics.png", dpi=180)
