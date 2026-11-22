"""Measure nodal error for manufactured solutions on a square."""

import argparse

import numpy as np
from onesim.simulation.problem_definition.declarations.physics.equations.equation_solver import solve_coefficient_form
from onesim.simulation.problem_definition.declarations.physics.models.contracts.models.custom_models import (
    EquationFormula, EquationModel, EquationModelForm,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cells", type=int, default=24, help="Uniform intervals along each axis")
parser.add_argument("--mode", type=int, default=1, help="Number of half-waves along each axis")
args = parser.parse_args()
if args.mode < 1 or args.cells < 2 * args.mode:
    parser.error("choose a positive mode and at least two intervals per half-wave")

equation = EquationModel(
    equation_id="teaching.diffusion", name="Variable diffusion",
    form=EquationModelForm.COEFFICIENT_PDE, dependent_variables=("u",),
    domain_ids=("square",), dimension=2, coordinate_system="cartesian",
    formulas=(
        EquationFormula("diffusion", "1+x", "m^2/s"),
        EquationFormula("absorption", "2", "1/s"),
        EquationFormula("source", "(2*(1+x)*a**2+2)*sin(a*x)*sin(a*y)-a*cos(a*x)*sin(a*y)", "1/s"),
        EquationFormula("dirichlet", "0", "1"),
    ),
)
wave_number = args.mode * np.pi
solved = solve_coefficient_form(
    equation, resolution=args.cells, parameter_values={"a": wave_number},
    linear_method="cg", preconditioner="jacobi", relative_tolerance=1e-10,
    absolute_tolerance=1e-12, maximum_iterations=2000,
)
axis = np.asarray(solved["coordinates"])
mode_shape = np.sin(wave_number * axis)
exact = np.outer(mode_shape, mode_shape)
print(f"Interior unknowns: {solved['unknowns']}")
print(f"Maximum nodal error: {np.max(np.abs(np.asarray(solved['solution']) - exact)):.7e}")
print(f"Relative algebraic residual: {solved['solver']['relative_residual']:.5e}")
