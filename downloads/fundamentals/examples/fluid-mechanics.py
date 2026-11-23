"""Intersect a pump curve with a pipe-system head curve at two lifts."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.nonlinear.kernels.local_solvers.inversion.bracketed_inverse import bisect_increasing

speed = np.linspace(0.25, 1.2, 181)  # fraction of reference speed
shutoff = 20. * speed**2             # pump shutoff head in m
pump_loss, pipe_loss = 0.025, 0.075   # m per (L/s)²
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for lift, colour in [(0., "#19596a"), (5., "#a5442e")]:
    available = shutoff - lift
    delivery = np.zeros_like(speed)
    active = available > 0
    def total_loss(flow):
        return (pump_loss + pipe_loss) * flow**2
    delivery[active] = bisect_increasing(total_loss, available[active], 0., 30.)
    assert np.allclose(shutoff[active] - pump_loss*delivery[active]**2,
                       lift + pipe_loss*delivery[active]**2)
    ax.plot(speed, delivery, color=colour, label=f"Static lift {lift:g} m")

ax.axvline(0.5, color="0.45", linestyle=":", label="5 m lift threshold")
ax.set(xlabel="Pump speed / reference speed", ylabel="Forward delivery (L/s)")
ax.legend()
fig.savefig("fluid-mechanics.png", dpi=170)
