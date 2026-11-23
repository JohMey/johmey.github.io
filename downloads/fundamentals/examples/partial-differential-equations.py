"""Find the temperature jump at an imperfect contact between two slabs."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.linear.matrices.dense_linear_algebra import solve_dense

n = 12  # even number of equal, cell-centred control volumes
length = 0.12  # m
dx = length / n
k = np.repeat([12., 3.], n // 2)  # W/(m K)
left, right = 330., 300.  # K
centres = (np.arange(n) + 0.5) * dx
fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
for contact, colour in [(0.0, "#a4492b"), (0.0125, "#176879")]:  # m² K/W
    resistance = np.r_[dx/(2*k[0]), dx/(2*k[:-1]) + dx/(2*k[1:]), dx/(2*k[-1])]
    resistance[n//2] += contact
    g = 1 / resistance
    matrix = np.diag(g[:-1]+g[1:]) - np.diag(g[1:-1], 1) - np.diag(g[1:-1], -1)
    rhs = np.zeros(n)
    rhs[0], rhs[-1] = g[0]*left, g[-1]*right
    temperature = solve_dense(matrix, rhs)
    fluxes = -np.diff(np.r_[left, temperature, right]) / resistance
    q = fluxes[0]
    assert np.allclose(fluxes, (left-right)/(length/2/k[0]+contact+length/2/k[-1]))
    hot_face = temperature[n//2-1] - q*dx/(2*k[0])
    cold_face = temperature[n//2] + q*dx/(2*k[-1])
    assert np.isclose(hot_face-cold_face, q*contact)
    x = np.r_[0, centres[:n//2], length/2, length/2, centres[n//2:], length]
    y = np.r_[left, temperature[:n//2], hot_face, cold_face, temperature[n//2:], right]
    ax.plot(100*x, y, color=colour, label=f"Contact resistance {contact:g} m² K/W")
    print(f"Flux {q:.0f} W/m²; interface jump {hot_face-cold_face:.1f} K")
ax.set(xlabel="Distance through slabs (cm)", ylabel="Temperature (K)")
ax.legend()
ax.grid(alpha=0.2)
fig.savefig("partial-differential-equations.png", dpi=160)
