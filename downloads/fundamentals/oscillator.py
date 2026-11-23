"""Compare the free response of a released mass in three damping regimes."""
import argparse
import csv
import sys

import numpy as np
from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.solid_mechanics import StructuralOscillator
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue


def displacement_ratio(time, ratio, omega):
    """Closed-form displacement divided by its initial value at zero release speed."""
    tau = omega * time
    if abs(ratio - 1) < 1e-12:
        return (1 + tau) * np.exp(-tau)
    if ratio < 1:
        beta = np.sqrt(1 - ratio**2)
        return np.exp(-ratio * tau) * (np.cos(beta * tau) + ratio / beta * np.sin(beta * tau))
    # Evaluate the separated exponential modes to avoid overflowing cosh at late times.
    beta = np.sqrt(ratio**2 - 1)
    slow, fast = -1 / (ratio + beta), -(ratio + beta)
    return (-fast * np.exp(slow * tau) + slow * np.exp(fast * tau)) / (slow - fast)


def solve(ratio):
    mass, stiffness, release = 1.0, 4.0, 0.010
    domain = LumpedDomain()
    omega = np.sqrt(stiffness / mass)
    oscillator = StructuralOscillator(domain=domain.system, mass=mass, stiffness=stiffness,
                                      damping=2 * ratio * mass * omega)
    problem = Problem(name="Release from rest", geometry=domain, physics=(oscillator,),
                      initial_conditions=(InitialValue(oscillator.displacement, release),
                                          InitialValue(oscillator.velocity, 0.0)))
    simulation = Simulation(problem=problem, mesh=LumpedMesh(domain),
                            numerics=Numerics(spatial=LumpedParameters(), time_integrator=ClassicalRungeKutta4()))
    history = simulation.run(start_time=0, end_time=6.0, time_step=0.01)
    time, values = history.field(oscillator.displacement).time_series()
    return zip(time, 1000 * np.asarray(values), 1000 * release * displacement_ratio(time, ratio, omega))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=float, default=0.2, help="Damping divided by critical damping (0 to 4)")
    args = parser.parse_args()
    if not 0 <= args.ratio <= 4:
        parser.error("choose a damping ratio between 0 and 4 for the supplied step")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("Time (s)", "OneSimulation displacement (mm)", "Exact displacement (mm)"))
    writer.writerows(solve(args.ratio))
