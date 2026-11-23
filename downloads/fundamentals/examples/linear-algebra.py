"""Find the reaction transmitted by a compliant link between two supports."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.linear.matrices.dense_linear_algebra import solve_dense

support = np.diag([2.0, 8.0])  # N/mm
load = np.array([10.0, 0.0])  # N
gap = np.array([1.0, -1.0])  # positive when the first support moves farther
free_gap = gap @ solve_dense(support, load)
compliance = gap @ solve_dense(support, gap)  # mm/N
link_stiffness = np.geomspace(0.02, 200, 150)  # N/mm
reaction = free_gap / (compliance + 1 / link_stiffness)
rigid_reaction = free_gap / compliance

# Check the reduced equation against the full two-displacement system.
for stiffness, force in zip(link_stiffness, reaction):
    displacement = solve_dense(support + stiffness * np.outer(gap, gap), load)
    assert np.isclose(stiffness * (gap @ displacement), force)
print(f"Half the rigid reaction at {1/compliance:.2f} N/mm; rigid limit {rigid_reaction:.2f} N")

fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
ax.semilogx(link_stiffness, reaction, color="#176879", label="Elastic link")
ax.axhline(rigid_reaction, ls="--", color="#a4492b", label="Rigid constraint")
ax.plot(1/compliance, rigid_reaction/2, "o", color="#273d30")
ax.set(xlabel="Link stiffness (N/mm)", ylabel="Link reaction (N)", ylim=(0, 9))
ax.grid(alpha=0.2)
ax.legend()
fig.savefig("linear-algebra.png", dpi=160)
