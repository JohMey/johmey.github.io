"""Slide a V-shaped power trace across an unchanged quadrature mesh."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics.fem.backends.spaces.line_lagrange import LineLagrangeSpace

cells = 8
line = Line(start=(0, 0, 0), end=(1, 0, 0))  # time interval, seconds
mesh = StructuredLineMesher(cells=cells).generate(line)
space = LineLagrangeSpace(mesh, order=1)
width = 1 / cells
fraction = np.linspace(0, 1, 121)
corner = (3 + fraction) * width  # position in fourth panel (s)
exact = (corner**2 + (1 - corner)**2) / 2  # J at 1 W/s

fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
for count, colour in [(1, "#176879"), (2, "#a4492b")]:
    energy = np.zeros_like(corner)
    for cell in range(cells):
        points, weights = space.quadrature(cell, order=count)
        power = np.abs(points[:, 0, None] - corner)  # W at 1 W/s
        energy += weights @ power
    error = energy - exact
    if count == 1:
        predicted = -width**2 * np.minimum(fraction, 1 - fraction)**2
        assert np.allclose(error, predicted)
    assert np.allclose(error[[0, -1]], 0)
    ax.plot(fraction, 1000 * error, color=colour, label=f"{count} Gauss point(s) per panel")
    print(f"{count} point(s): centre error {1000*error[len(error)//2]:+.4f} mJ")
ax.axhline(0, color="#555", lw=0.8)
ax.set(xlabel="Corner position within one panel (fraction)", ylabel="Computed minus exact energy (mJ)")
ax.legend()
fig.savefig("approximation-and-quadrature.png", dpi=160)
