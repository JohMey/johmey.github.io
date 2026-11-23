"""Compare a flux-difference solution with the continuous and discrete sine modes."""
import argparse
import csv
import sys

import numpy as np
from onesim.simulation.problem_definition.declarations.physics.equations.equation_solver import solve_coefficient_form
from onesim.simulation.problem_definition.declarations.physics.models.contracts.models.custom_models import (
    EquationFormula, EquationModel, EquationModelForm,
)


def solve(mode, cells):
    equation = EquationModel(
        equation_id="fundamentals.poisson_mode", name="Forced sine mode", form=EquationModelForm.COEFFICIENT_PDE,
        dependent_variables=("u",), domain_ids=("square",), dimension=2, coordinate_system="cartesian",
        formulas=(EquationFormula("diffusion", "1", "m^2/s"),
                  EquationFormula("source", "2*a**2*sin(a*x)*sin(a*y)", "1/s"),
                  EquationFormula("dirichlet", "0", "1")))
    a = mode * np.pi
    result = solve_coefficient_form(equation, resolution=cells, parameter_values={"a": a},
                                    linear_method="cg", preconditioner="jacobi", relative_tolerance=1e-11,
                                    absolute_tolerance=1e-13, maximum_iterations=2000)
    axis = np.asarray(result["coordinates"])
    exact = np.outer(np.sin(a * axis), np.sin(a * axis))
    discrete_eigenvalue = 8 * cells**2 * np.sin(a / (2 * cells))**2
    factor = 2 * a**2 / discrete_eigenvalue
    sampled_peak = np.max(np.abs(exact))
    measured_error = np.max(np.abs(np.asarray(result["solution"]) - exact))
    return cells, measured_error, abs(factor - 1) * sampled_peak


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, default=1, help="Half-waves across each side")
    parser.add_argument("--cells", type=int, nargs="+", default=[8, 12, 16, 24, 32, 48, 64, 96])
    args = parser.parse_args()
    if args.mode < 1 or min(args.cells) < 2 * args.mode:
        parser.error("use a positive mode and at least two cells per half-wave")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("Cells per side", "OneSimulation maximum nodal error", "Discrete-mode prediction"))
    writer.writerows(solve(args.mode, n) for n in args.cells)
