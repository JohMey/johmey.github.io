"""Potential on the midline of a grounded 1 m square with sinusoidal volume charge.

Run with a OneSimulation Python installation. Coordinates are metres and potential is volts.
"""
import argparse
import csv
import sys
import numpy as np
from onesim.simulation.problem_definition.declarations.physics.equations.equation_solver import solve_coefficient_form
from onesim.simulation.problem_definition.declarations.physics.models.contracts.models.custom_models import (
    EquationFormula, EquationModel, EquationModelForm,
)

EPSILON = 8.8541878128e-12


def midline(cells, charge):
    rho_over_eps = charge * 1e-9 / EPSILON
    model = EquationModel(
        equation_id="fundamentals.grounded_square", name="Grounded square potential",
        form=EquationModelForm.COEFFICIENT_PDE, dependent_variables=("u",),
        domain_ids=("square",), dimension=2, coordinate_system="cartesian",
        formulas=(EquationFormula("diffusion", "1", "1"),
                  EquationFormula("source", "s*sin(k*x)*sin(k*y)", "V/m^2"),
                  EquationFormula("dirichlet", "0", "V")))
    field = solve_coefficient_form(model, resolution=cells, parameter_values={"s": rho_over_eps, "k": np.pi},
                                   linear_method="cg", preconditioner="jacobi",
                                   relative_tolerance=1e-11, absolute_tolerance=1e-13, maximum_iterations=1000)
    axis = np.asarray(field["coordinates"])
    numerical = np.asarray(field["solution"])[cells // 2]
    exact = rho_over_eps * np.sin(np.pi * axis) / (2 * np.pi**2)
    return zip(axis, numerical, exact)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", type=int, default=12, help="Even number of grid cells per side")
    p.add_argument("--charge", type=float, default=2., help="Peak charge density in nC/m³")
    args = p.parse_args()
    if args.cells < 2 or args.cells % 2:
        p.error("--cells must be a positive even integer ≥ 2")
    out = csv.writer(sys.stdout, lineterminator="\n")
    out.writerow(["x_m", "onesimulation_V", "continuum_V"])
    out.writerows(midline(args.cells, args.charge))
