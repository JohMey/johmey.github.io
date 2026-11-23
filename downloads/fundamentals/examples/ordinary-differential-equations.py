"""Measure artificial energy loss over 20 seconds in an undamped oscillator."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.time import ClassicalRungeKutta4TimeIntegrator

omega = 2.0  # rad/s
mass = 1.0  # kg
duration = 20.0  # s
method = ClassicalRungeKutta4TimeIntegrator()

def dynamics(state, time):
    position, velocity = state
    return (velocity, -omega**2 * position)

fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
losses = []
for n, colour in [(50, "#a4492b"), (100, "#176879"), (200, "#3868a0")]:
    h = duration / n
    state = (1.0, 0.0)  # metres, metres per second
    energies = []
    for j in range(n + 1):
        x, v = state
        energies.append(0.5 * mass * (v*v + omega**2*x*x))
        if j < n:
            state = method.advance(state, dynamics, h, time=j*h)
    fraction = np.array(energies) / energies[0]
    z = omega*h
    assert np.isclose(fraction[-1], (1 - z**6/72 + z**8/576)**n)
    losses.append(1 - fraction[-1])
    ax.plot(np.arange(n + 1)*h, 100*(1 - fraction), color=colour, label=f"h = {h:g} s")
print("Final percentage losses:", np.round(100*np.array(losses), 5))
ax.set(xlabel="Time (s)", ylabel="Artificial energy loss (%)")
ax.legend()
ax.grid(alpha=0.2)
fig.savefig("ordinary-differential-equations.png", dpi=160)
