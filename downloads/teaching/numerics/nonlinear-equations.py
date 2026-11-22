"""Solve one implicit radiative-cooling step and evaluate its discrete residual."""

import argparse

import numpy as np
from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import DampedNewton, Newton
from onesim.numerics.time import BackwardEuler
from onesim.physics.heat_transfer import RadiationCooling, ThermalGreyDiffuseRadiationBoundary
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--step", type=float, default=1.0, help="Duration of the single backward Euler step, in s")
parser.add_argument("--solver", choices=("newton", "damped"), default="newton")
args = parser.parse_args()
if not np.isfinite(args.step) or args.step <= 0:
    parser.error("step must be positive and finite")

domain = LumpedDomain()
heat = RadiationCooling(
    domain=domain.system, heat_capacity=2, radiating_area=0.3,
    radiation=ThermalGreyDiffuseRadiationBoundary(
        condition_id="test radiation", emissivity=0.5, ambient_temperature=1.0, stefan_boltzmann=0.02,
    ),
)
problem = Problem(name="Implicit radiation step", geometry=domain, physics=(heat,),
                  initial_conditions=(InitialValue(heat.temperature, 2.0),))
solver_type = Newton if args.solver == "newton" else DampedNewton
solver = solver_type(linear_solver=SparseDirect(), jacobian="analytic", relative_tolerance=1e-12,
                     absolute_tolerance=1e-13)
simulation = Simulation(problem=problem, mesh=LumpedMesh(domain),
                        numerics=Numerics(spatial=LumpedParameters(), time_integrator=BackwardEuler(),
                                          nonlinear_solver=solver))
result = simulation.run(start_time=0.0, end_time=args.step, time_step=args.step)
temperature = np.asarray(result.field(heat.temperature).values).item()
residual = temperature - 2 + args.step * 0.0015 * (temperature**4 - 1)
print(f"End of one step: {temperature:.10f} K")
print(f"Backward Euler stage residual: {residual:.3e} K")
