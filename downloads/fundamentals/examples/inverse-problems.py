"""Choose a sampling time for a cooling rate when its amplitude is also unknown."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.linear.matrices.dense_linear_algebra import solve_dense

amplitude = 4.0  # K above a known ambient temperature
rate = 0.4  # per hour
noise = 0.08  # K, standard deviation of each independent measurement
times = np.linspace(0.15, 10.0, 400)  # first reading at 0 h; vary the second
standard_errors = []
for t in times:
    decay = np.exp(-rate*t)
    sensitivity = np.array([[1., 0.], [decay, -amplitude*t*decay]])
    covariance = solve_dense(sensitivity.T @ sensitivity / noise**2, np.eye(2))
    standard_errors.append(np.sqrt(covariance[1, 1]))
known_amplitude = noise * np.exp(rate*times) / (amplitude*times)
analytic = noise * np.sqrt(1 + np.exp(2*rate*times)) / (amplitude*times)
assert np.allclose(standard_errors, analytic)
best = np.argmin(standard_errors)
print(f"Optimal second reading on grid: {times[best]:.2f} h; rate SE {standard_errors[best]:.4f} /h")
fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
ax.semilogy(times, standard_errors, color="#176879", label="Estimate amplitude and rate")
ax.semilogy(times, known_amplitude, "--", color="#a4492b", label="Amplitude known exactly")
ax.scatter(times[best], standard_errors[best], color="#273d30", zorder=3)
ax.set(xlabel="Time of the second reading (h)", ylabel="Local rate standard error (1/h)")
ax.legend()
ax.grid(alpha=0.2)
fig.savefig("inverse-problems.png", dpi=160)
