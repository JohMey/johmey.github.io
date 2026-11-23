"""Compare the 3D centreline of an elastic cantilever with the integrated beam equation."""
import argparse
import csv
import sys

import numpy as np
from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import CadHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import FixedDisplacement, LinearElasticity, Traction, TractionFree
from onesim.simulation import Problem, Simulation


def solve(height_mm, axial_cells):
    length, width, height = 0.12, 0.018, height_mm / 1000
    force, modulus = 30.0, 70e9
    beam = Box(length=length, width=width, height=height)
    faces = beam.boundaries
    material = IsotropicSolid(density=2700, specific_heat_capacity=900, thermal_conductivity=200,
                              youngs_modulus=modulus, poisson_ratio=0.30, thermal_expansion_coefficient=23e-6)
    solid = LinearElasticity(domain=beam.volume, material=material)
    problem = Problem(name="Cantilever with uniformly loaded end face", geometry=beam, physics=(solid,),
                      boundary_conditions=(
                          FixedDisplacement(boundary=faces.x_min, physics=solid, value=(0, 0, 0)),
                          Traction(boundary=faces.x_max, physics=solid, value=(0, 0, -force / (width * height))),
                          TractionFree(boundary=faces.all - faces.x_min - faces.x_max, physics=solid)))
    mesh = CadHexahedralMesher(cells=(axial_cells, 4, 4)).generate(beam)
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())
    field = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(solid.displacement)
    x = np.linspace(0, length, 49)
    centreline = np.column_stack((x, np.full_like(x, width / 2), np.full_like(x, height / 2)))
    downward = -field.evaluate(centreline)[:, 2]
    moment_of_area = width * height**3 / 12
    beam_prediction = force * x**2 * (3 * length - x) / (6 * modulus * moment_of_area)
    return zip(1000 * x, 1000 * downward, 1000 * beam_prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, default=12, help="Beam height in mm (6 to 48)")
    parser.add_argument("--cells", type=int, default=24, help="Number of hexahedra along the span")
    args = parser.parse_args()
    if not 6 <= args.height <= 48 or args.cells < 2:
        parser.error("use 6 to 48 mm and at least two axial cells")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("Position (mm)", "OneSimulation deflection (mm)", "Beam deflection (mm)"))
    writer.writerows(solve(args.height, args.cells))
