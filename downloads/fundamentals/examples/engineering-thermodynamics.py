"""Check for an outlet temperature cross in a two-stream heat exchanger."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.physics.systems.process import counterflow_effectiveness, parallel_flow_effectiveness

hot_in, cold_in = 80., 20.       # °C
hot_rate, cold_rate = 2., 1.     # kW/K
capacity_min = min(hot_rate, cold_rate)
ratio = capacity_min/max(hot_rate, cold_rate)
ntu = np.linspace(0, 4, 201)

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for name, model, style in [("Counterflow", counterflow_effectiveness, "-"),
                            ("Parallel flow", parallel_flow_effectiveness, "--")]:
    heat = model(ntu, ratio)*capacity_min*(hot_in-cold_in)  # kW
    hot_out = hot_in - heat/hot_rate
    cold_out = cold_in + heat/cold_rate
    assert np.allclose(hot_rate*(hot_in-hot_out), cold_rate*(cold_out-cold_in))
    ax.plot(ntu, hot_out, style, color="#a5442e", label=f"{name}, hot outlet")
    ax.plot(ntu, cold_out, style, color="#19596a", label=f"{name}, cold outlet")

ax.axvline(2*np.log(2), color="0.5", linestyle=":", label="Counterflow outlets equal")
ax.set(xlabel="Transfer units UA / minimum capacity rate", ylabel="Outlet temperature (°C)")
ax.legend(fontsize=8)
fig.savefig("engineering-thermodynamics.png", dpi=170)
