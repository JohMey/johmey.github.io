"""Elastic collision of two carts with interchangeable spring bumpers."""
import matplotlib.pyplot as plt
import numpy as np
from onesim.numerics.time import ClassicalRungeKutta4TimeIntegrator

mass_left, mass_right = 0.20, 0.30  # kg
closing_speed = 0.50  # m/s
reduced_mass = mass_left * mass_right / (mass_left + mass_right)
stepper = ClassicalRungeKutta4TimeIntegrator()
fig, ax = plt.subplots(figsize=(6.6, 3.5), layout="constrained")

for stiffness in (120.0, 480.0):  # N/m
    omega = np.sqrt(stiffness / reduced_mass)
    times = np.linspace(0, np.pi / omega, 201)
    dt = times[1] - times[0]
    state = (0.0, closing_speed)  # compression and compression rate
    force = [0.0]
    def contact_rate(state, time):
        compression, speed = state
        return speed, -stiffness * compression / reduced_mass
    for time in times[:-1]:
        state = stepper.advance(state, contact_rate, dt, time=time)
        force.append(stiffness * state[0])
    impulse = np.trapezoid(force, times)
    assert np.isclose(impulse, 2 * reduced_mass * closing_speed, rtol=3e-5)
    print(f"{stiffness:g} N/m: {max(force):.3f} N peak, {impulse:.4f} N s")
    ax.plot(1000 * times, force, label=f"{stiffness:g} N/m")
ax.set(xlabel="Time since first contact (ms)", ylabel="Contact force (N)")
ax.grid(alpha=0.2)
ax.legend(title="Bumper stiffness")
fig.savefig("classical-mechanics.png", dpi=180)
