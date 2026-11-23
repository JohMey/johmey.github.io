"""A positive step drives two stable paths with opposite output signs."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.time.methods.backends.fixed_step import FixedStepRungeKutta

tau = np.array([2., 0.25])  # seconds, slow and fast paths
weights = np.array([2., -1.])
solver = FixedStepRungeKutta("rk4")
run = solver.integrate(lambda t, x: (1-x)/tau, np.zeros(2), 0., 8., step=0.02)
t = run.times
terms = run.states * weights
output = terms.sum(axis=1)
reference = (1-np.exp(-t[:, None]/tau)) @ weights
assert np.max(abs(output-reference)) < 1e-6
index = np.argmin(output)
print(f"First minimum on the grid: {output[index]:.3f} at {t[index]:.2f} s")

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.plot(t, terms[:, 0], "--", color="#51809c", label="Slow path")
ax.plot(t, terms[:, 1], ":", color="#a5442e", label="Fast path")
ax.plot(t, output, color="#19596a", label="Measured sum")
ax.axhline(0, color="0.5", linewidth=0.7)
ax.set(xlabel="Time after unit step (s)", ylabel="Normalised output")
ax.legend()
fig.savefig("control-engineering.png", dpi=170)
