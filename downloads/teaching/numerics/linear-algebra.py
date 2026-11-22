"""Compare CG and direct solutions on the same heated unit cube."""

import argparse

import numpy as np
from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import ConjugateGradient, SparseDirect
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction
from onesim.simulation import Problem, Simulation


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cells", type=int, default=5, help="Cells per axis")
parser.add_argument("--rtol", type=float, default=1e-10, help="Relative CG stopping tolerance")
args = parser.parse_args()
if args.cells < 2 or not 0 < args.rtol < 1:
    parser.error("choose at least two cells per axis and 0 < rtol < 1")

box = Box(length=1, width=1, height=1)
material = IsotropicSolid(density=1, specific_heat_capacity=1, thermal_conductivity=1,
                          youngs_modulus=100, poisson_ratio=0.3, thermal_expansion_coefficient=0)
heat = HeatConduction(domain=box.volume, material=material, heating=1.0)
problem = Problem(name="Temperature rise in a heated cube", geometry=box, physics=(heat,),
                  boundary_conditions=(FixedTemperature(physics=heat, boundary=box.boundaries.all, value=0.0),))
mesh = StructuredHexahedralMesher(cells=(args.cells,) * 3).generate(box)
space = FiniteElement(element=Lagrange(1))


def temperature(solver):
    simulation = Simulation(problem=problem, mesh=mesh, numerics=Numerics(spatial=space, linear_solver=solver))
    return np.asarray(simulation.solve_stationary().field(heat.temperature).values)


reference = temperature(SparseDirect())
approximation = temperature(ConjugateGradient(relative_tolerance=args.rtol, absolute_tolerance=1e-14,
                                               maximum_iterations=1000))
print(f"Direct peak temperature rise: {reference.max():.9g} K")
print(f"CG peak temperature rise:     {approximation.max():.9g} K")
print(f"Maximum CG/direct nodal difference: {np.max(np.abs(approximation - reference)):.3e} K")
