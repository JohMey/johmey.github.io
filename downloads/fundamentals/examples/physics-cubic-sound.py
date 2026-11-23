"""Sound velocities when a wave rotates through the (001) plane of a cubic solid."""
import numpy as np
import matplotlib.pyplot as plt
from onesim.numerics.linear.matrices.dense_linear_algebra import symmetric_eigenvalues

c11, c12, c44 = np.array([166, 64, 80]) * 1e9  # Pa
density = 2330.0  # kg/m^3

def speeds(angle):
    n = np.array([np.cos(angle), np.sin(angle), 0])
    gamma = (c12+c44) * np.outer(n, n)
    gamma += np.diag(c44 + (c11-c12-2*c44)*n**2)
    return np.sqrt(symmetric_eigenvalues(gamma)/density)

angle = np.linspace(0, np.pi/2, 181)
velocity = np.array([speeds(a) for a in angle]) / 1000
print("Wave speeds in km/s, sorted by magnitude")
for label, a in [("[100]", 0), ("[110]", np.pi/4)]:
    print(label, " ".join(f"{v/1000:.3f}" for v in speeds(a)))
for branch, label in enumerate(["Slow shear", "Fast shear", "Quasi-longitudinal"]):
    plt.plot(np.degrees(angle), velocity[:,branch], label=label)
plt.xlabel("Direction from [100] toward [010] (degrees)")
plt.ylabel("Sound speed (km/s)")
plt.legend()
plt.tight_layout()
plt.savefig("physics-cubic-sound.png", dpi=160)
