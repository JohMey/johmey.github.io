"""Local spectroscopy at an open chain end and in its interior."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.linear.matrices.dense_linear_algebra import symmetric_eigensystem

sites = 81
hopping = 1.0  # eV
broadening = 0.09  # eV, standard deviation of the energy resolution
hamiltonian = -hopping * (np.eye(sites, k=1) + np.eye(sites, k=-1))
levels, vectors = symmetric_eigensystem(hamiltonian)
energy = np.linspace(-2.6 * hopping, 2.6 * hopping, 1001)
kernel = np.exp(-0.5 * ((energy[:, None] - levels) / broadening)**2)
kernel /= np.sqrt(2 * np.pi) * broadening
fig, ax = plt.subplots(figsize=(6.6, 3.5), layout="constrained")
for site, label in [(0, "End site"), (sites // 2, "Centre site")]:
    weights = np.abs(vectors[site])**2
    density = kernel @ weights
    assert np.isclose(np.trapezoid(density, energy), 1, atol=1e-6)
    print(f"{label}: local density at zero energy = {density[len(energy)//2]:.3f} /eV")
    ax.plot(energy, density, label=label)
ax.set(xlabel="Energy (eV)", ylabel="Local density per site and spin (1/eV)")
ax.grid(alpha=0.2)
ax.legend()
fig.savefig("condensed-matter.png", dpi=180)
