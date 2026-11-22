"""Measure time-integration error in the conserved reaction A -> B."""

import argparse

import numpy as np
from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import ClassicalRungeKutta4, ForwardEuler
from onesim.physics.chemistry import ReactionIR, ReactionKinetics, ReactionNetworkIR, SpeciesIR
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=100, help="Equal time steps over five decay times")
parser.add_argument("--method", choices=("rk4", "euler"), default="rk4")
args = parser.parse_args()
if args.steps < 1:
    parser.error("choose a positive number of time steps")
if args.method == "euler" and args.steps < 5:
    parser.error("Euler would produce a negative concentration on the first step; use at least five steps")

rate = 0.8
network = ReactionNetworkIR(
    "decay", "1",
    species=(SpeciesIR("A", 0, 1e-27, composition={"H": 1}),
             SpeciesIR("B", 0, 1e-27, composition={"H": 1})),
    reactions=(ReactionIR("conversion", {"A": 1}, {"B": 1}, {"kind": "constant", "value": rate}),),
    strict_balance=True,
)
domain = LumpedDomain()
kinetics = ReactionKinetics(domain=domain.system, temperature=300.0, network=network)
a, b = kinetics.concentration("A"), kinetics.concentration("B")
problem = Problem(name="Isothermal first-order decay", geometry=domain, physics=(kinetics,),
                  initial_conditions=(InitialValue(a, 1.0), InitialValue(b, 0.0)))
integrator = ClassicalRungeKutta4() if args.method == "rk4" else ForwardEuler()
simulation = Simulation(problem=problem, mesh=LumpedMesh(domain),
                        numerics=Numerics(spatial=LumpedParameters(), time_integrator=integrator))
duration = 5 / rate
result = simulation.run(start_time=0.0, end_time=duration, time_step=duration / args.steps)
times, a_values = result.field(a).time_series()
_, b_values = result.field(b).time_series()
exact = np.exp(-rate * times)
print(f"Final A: {a_values[-1]:.10g} mol/m^3")
print(f"Maximum A error: {np.max(np.abs(a_values - exact)):.5e} mol/m^3")
print(f"Maximum A+B drift: {np.max(np.abs(a_values + b_values - 1)):.3e} mol/m^3")
