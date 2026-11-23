"""Compare strain-rosette geometries with the same middle-gauge error."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.linear.matrices.dense_linear_algebra import solve_dense

true_strain = np.array([100., -60., 80.])  # ex, ey, engineering shear in µε
bias = 5.0  # µε added to the middle gauge
rotation = np.arange(0, 181)

def gauge_rows(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.column_stack((c**2, s**2, c*s))

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for spacing, colour in [(45, "#19596a"), (10, "#a5442e")]:
    layout = gauge_rows([0, spacing, 2*spacing])
    readings = layout @ true_strain
    assert np.allclose(solve_dense(layout, readings), true_strain)
    readings[1] += bias
    inferred = solve_dense(layout, readings)
    error = gauge_rows(rotation) @ (inferred - true_strain)
    ax.plot(rotation, error, color=colour, label=f"Gauge spacing {spacing}°")
    print(f"{spacing}°: error at 90° = {error[90]:.2f} µε")

ax.axhline(0, color="0.5", linewidth=0.7)
ax.set(xlabel="Direction of predicted normal strain (degrees)",
       ylabel="Prediction error (µε)", xlim=(0, 180))
ax.legend()
fig.savefig("engineering-mechanics.png", dpi=170)
