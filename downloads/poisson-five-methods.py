"""Compare five spatial discretizations for a manufactured Poisson solution."""

import argparse
from time import perf_counter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cells', type=int, default=16, help='Hexahedra per direction (default: 16)')
args = parser.parse_args()
if args.cells < 7:
    parser.error('--cells must be at least 7')

import numpy as np
from dataclasses import dataclass
from onesim.cad.d3 import Box
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import FixedPotential, PoissonEquation
from onesim.simulation import Problem, Simulation

@dataclass(frozen=True)
class Permittivity:
    unit = "F/m"
    def evaluate(self, **state):
        return 1.0

cube = Box(length=1, width=1, height=1)
exact = lambda x, y, z: np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z)

field = PoissonEquation(
    domain=cube.volume, permittivity=Permittivity(),
    charge_density=lambda x, y, z: 3 * np.pi**2 * exact(x, y, z))

problem = Problem(
    name="Charge in a grounded box", geometry=cube, physics=(field,),
    boundary_conditions=(
        FixedPotential(physics=field, boundary=cube.boundaries.all, value=0.0),))

mesh = StructuredHexahedralMesher(cells=(args.cells,) * 3).generate(cube)

from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.fvm import FiniteVolume, LeastSquaresGradient, CentralDifference
from onesim.numerics.fdm import (FiniteDifference, CentralLaplacian,
    PolynomialDifference, PolynomialLaplacian)
from onesim.numerics.fdm import CentralDifference as FDGradient

methods = {
    "FEM, linear elements":    FiniteElement(element=Lagrange(order=1)),
    "FEM, quadratic elements": FiniteElement(element=Lagrange(order=2)),
    "FVM":                     FiniteVolume(gradient=LeastSquaresGradient(),
                                            diffusion=CentralDifference()),
    "FDM, 2nd order":          FiniteDifference(gradient=FDGradient(),
                                                diffusion=CentralLaplacian()),
    "FDM, 4th order":          FiniteDifference(
                                   gradient=PolynomialDifference(accuracy_order=4),
                                   diffusion=PolynomialLaplacian(accuracy_order=4)),
}

centres = mesh.coordinates[mesh.cells].mean(axis=1)
for name, spatial in methods.items():
    start = perf_counter()
    result = Simulation(
        problem=problem, mesh=mesh,
        numerics=Numerics(spatial=spatial, linear_solver=SparseDirect()),
    ).solve_stationary()
    phi = result.field(field.potential).evaluate(centres)
    error = np.sqrt(np.mean((phi - exact(*centres.T))**2))
    print(f"{name:26s} RMS={error:.2e}, {perf_counter() - start:.2f} s")
