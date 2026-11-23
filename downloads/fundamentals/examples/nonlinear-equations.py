"""Follow both folds of a scalar equilibrium with a two-variable Newton correction."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.linear.matrices.dense_linear_algebra import solve_dense

step = 0.12  # arclength in dimensionless displacement-load coordinates
point = np.array([1.2, 1.2 - 1.2**3])
path = [point.copy()]
for _ in range(100):
    x, load = point
    tangent = np.array([-1.0, 3*x*x - 1])
    tangent /= np.linalg.norm(tangent)
    predicted = point + step * tangent
    point = predicted.copy()
    for _ in range(10):
        x, load = point
        residual = np.array([x**3 - x + load, tangent @ (point - predicted)])
        if np.linalg.norm(residual, np.inf) < 1e-12:
            break
        jacobian = np.array([[3*x*x - 1, 1], tangent])
        point -= solve_dense(jacobian, residual)
    else:
        raise RuntimeError("Reduce the continuation step")
    path.append(point.copy())
    if point[0] < -1.2:
        break

x, load = np.array(path).T
fold_x = np.array([-1, 1]) / np.sqrt(3)
assert x[-1] < -1.2 and np.max(np.abs(x**3 - x + load)) < 1e-11
assert np.count_nonzero(np.diff(np.sign(np.diff(load)))) == 2
print("Fold loads from the derivative:", fold_x - fold_x**3)
fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
ax.plot(load, x, ".-", color="#176879", label="Corrected continuation steps")
ax.scatter(fold_x-fold_x**3, fold_x, color="#a4492b", label="Analytic folds", zorder=3)
ax.set(xlabel="Load parameter (dimensionless)", ylabel="Displacement (dimensionless)")
ax.grid(alpha=0.2)
ax.legend()
fig.savefig("nonlinear-equations.png", dpi=160)
