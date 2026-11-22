"""Export numerical and exact values for the cooling and oscillator labs."""

import argparse
import csv
import math
import sys


def rk4_step(rhs, state, step):
    """Advance an autonomous system with classical fourth-order Runge–Kutta."""
    k1 = rhs(state)
    k2 = rhs(tuple(y + step * k / 2 for y, k in zip(state, k1)))
    k3 = rhs(tuple(y + step * k / 2 for y, k in zip(state, k2)))
    k4 = rhs(tuple(y + step * k for y, k in zip(state, k3)))
    return tuple(
        y + step * (a + 2 * b + 2 * c + d) / 6
        for y, a, b, c, d in zip(state, k1, k2, k3, k4)
    )


def cooling():
    temperature = 400.0
    beta = 5e-10

    def rhs(state):
        return (-beta * state[0] ** 4,)

    def exact(time):
        return ((temperature ** -3 + 3 * beta * time) ** (-1 / 3),)

    return 30.0, (temperature,), rhs, exact, ("temperature_K",), tuple


def oscillator():
    damping = 0.1
    frequency = 2 * math.pi
    damped_frequency = math.sqrt(frequency**2 - damping**2)

    def rhs(state):
        position, velocity = state
        return velocity, -2 * damping * velocity - frequency**2 * position

    def exact(time):
        sine = math.sin(damped_frequency * time)
        cosine = math.cos(damped_frequency * time)
        envelope = math.exp(-damping * time)
        position = envelope * (cosine + damping / damped_frequency * sine)
        velocity = -envelope * frequency**2 / damped_frequency * sine
        return position, velocity

    def observables(state):
        position, velocity = state
        specific_energy = (velocity**2 + frequency**2 * position**2) / 2
        return position, specific_energy

    duration = 10 * 2 * math.pi / damped_frequency
    names = ("position_m", "specific_energy_J_per_kg")
    return duration, (1.0, 0.0), rhs, exact, names, observables


def main():
    models = {"cooling": cooling, "oscillator": oscillator}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=models)
    parser.add_argument("--step", type=float, default=0.1, help="Maximum time step in seconds")
    args = parser.parse_args()

    duration, state, rhs, exact, names, observe = models[args.model]()
    if not math.isfinite(args.step) or not duration / 1_000_000 <= args.step <= 1.0:
        parser.error("choose a step no larger than 1 s that takes at most 1,000,000 steps")
    count = math.ceil(duration / args.step)
    step = duration / count

    writer = csv.writer(sys.stdout, lineterminator="\n")
    numerical = [name + "_numerical" for name in names]
    analytical = [name + "_exact" for name in names]
    writer.writerow(["time_s", "step_s", *numerical, *analytical])
    for index in range(count + 1):
        time = index * step
        writer.writerow([time, step, *observe(state), *observe(exact(time))])
        if index < count:
            state = rk4_step(rhs, state, step)


if __name__ == "__main__":
    main()
