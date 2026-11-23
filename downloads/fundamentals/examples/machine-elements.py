"""Torque from one annular rubbing face under a fixed peak pressure."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.discretization.geometry.carriers.kernels.quadrature.rules.reference_quadrature import gauss_legendre

outer = 0.10             # outer radius, m
peak, mu = 8.0e5, 0.30   # allowable contact pressure (Pa), friction coefficient
nodes, weights = np.asarray(gauss_legendre(3)).T

def torque(inner):
    r = (inner + outer)/2 + (outer - inner)*nodes/2
    pressure = peak*inner/r  # uniform wear: p*r is constant
    integrand = 2*np.pi*mu*pressure*r**2
    return (outer - inner)/2 * (weights @ integrand)

ratio = np.linspace(0.05, 0.99, 240)
values = np.array([torque(x*outer) for x in ratio])
best = 1/np.sqrt(3)
assert abs(ratio[np.argmax(values)] - best) < 0.005
print(f"Best inner radius {1000*best*outer:.2f} mm; torque {torque(best*outer):.1f} N m")
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.plot(ratio, values, color="#19596a", label="Integrated contact moment")
ax.plot(best, torque(best*outer), "o", color="#a5442e", label="Analytic optimum")
ax.set(xlabel="Inner radius / outer radius", ylabel="Torque per contact face (N m)")
ax.legend()
fig.savefig("machine-elements.png", dpi=170)
