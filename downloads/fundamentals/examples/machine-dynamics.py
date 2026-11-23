"""A force on the primary mass excites two modes of a tuned absorber."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.linear.matrices.dense_linear_algebra import symmetric_eigensystem

mass = np.array([5., 0.5])  # kg, primary and absorber
support, spring = 500., 50.  # N/m, support and connecting spring
stiffness = np.array([[support+spring, -spring], [-spring, spring]])
scale = np.diag(1/np.sqrt(mass))
omega2, shapes = symmetric_eigensystem(scale @ stiffness @ scale)
modes = scale @ shapes
residues = modes[0]**2      # applied force and measured motion both at mass 1
tuned = np.sqrt(spring/mass[1])
assert abs(np.sum(residues/(omega2 - tuned**2))) < 1e-12
assert np.isclose(np.sum(residues/omega2), 1/support)

frequency = np.linspace(1.0, 2.3, 600)  # Hz
omega = 2*np.pi*frequency[:, None]
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for damping, colour in [(0.005, "#19596a"), (0.04, "#a5442e")]:
    denominator = omega2 - omega**2 + 2j*damping*np.sqrt(omega2)*omega
    response = np.sum(residues / denominator, axis=1)
    ax.semilogy(frequency, 1000*abs(response), color=colour,
                label=f"Both modal damping ratios {100*damping:g}%")

ax.axvline(tuned/(2*np.pi), color="0.5", linestyle=":", label="Undamped cancellation")
ax.set(xlabel="Excitation frequency (Hz)", ylabel="Primary displacement / force (mm/N)")
ax.legend(fontsize=8)
fig.savefig("machine-dynamics.png", dpi=170)
