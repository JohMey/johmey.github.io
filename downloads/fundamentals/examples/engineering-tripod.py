"""Resolve measured loads on a three-dimensional tripod into signed member forces."""
import numpy as np
import matplotlib.pyplot as plt
from onesim.numerics.nonlinear import LeastSquares

angles = np.deg2rad([0, 120, 240])
# Columns point from the loaded apex toward a foot; positive force is tension.
directions = np.array([np.cos(angles), np.sin(angles), -np.ones(3)]) / np.sqrt(2)
horizontal = np.linspace(0, 6, 61)  # kN
forces = []
for load in horizontal:
    external = np.array([load, 0.0, -6.0])
    fit = LeastSquares().solve(lambda force: directions @ force + external, [-2.0]*3)
    assert np.linalg.norm(directions @ fit.x + external) < 1e-6
    forces.append(fit.x)
forces = np.array(forces)
print("H=3 kN: signed member forces [kN]", np.round(forces[30], 3))
print("H=6 kN: signed member forces [kN]", np.round(forces[-1], 3))
fig, ax = plt.subplots(figsize=(6.4, 3.7), layout="constrained")
ax.plot(horizontal, forces[:, 0], label="Foot at 0°")
ax.plot(horizontal, forces[:, 1], "--", label="Feet at ±120°")
ax.axhline(0, color="0.5", lw=0.8)
ax.set(xlabel="Horizontal apex load (kN)", ylabel="Axial force (kN)")
ax.legend()
plt.show()
