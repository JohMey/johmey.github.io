"""Compare a OneSimulation electrostatic solution with an independently derived reference.

Run field-laboratory.py --help for the three lab configurations. Each CSV row is a physical position and two potentials.
"""
import argparse
import csv
import sys
import numpy as np
from onesim.simulation.problem_definition.declarations.physics.equations.equation_solver import solve_coefficient_form
from onesim.simulation.problem_definition.declarations.physics.models.contracts.models.custom_models import (
    EquationFormula, EquationModel, EquationModelForm,
)


def solve(case, cells, contrast):
    dimension = 2 if case == "modes" else 1
    if case == "modes":
        diffusion, source, wall = "1", "k**2*(2*sin(k*x)+5*g*sin(2*k*x))*sin(k*y)", "0"
    elif case == "capacitor":
        diffusion, source, wall = "1+g*x", "0", "x"
    else:
        diffusion, source, wall = "1", "0", "1-x"
    # For the screening cell, u'' = g²u rather than the vacuum equation u'' = 0.
    terms = [EquationFormula("diffusion", diffusion, "1"), EquationFormula("source", source, "V/m^2"),
             EquationFormula("dirichlet", wall, "V")]
    if case == "screen":
        terms.append(EquationFormula("absorption", "g**2", "1/m^2"))
    definition = EquationModel(equation_id="fundamentals.field_lab_"+case, name=case+" field check",
                               form=EquationModelForm.COEFFICIENT_PDE, dimension=dimension,
                               coordinate_system="cartesian", dependent_variables=("u",),
                               domain_ids=("sample",), formulas=tuple(terms))
    problem = solve_coefficient_form(definition, resolution=cells, parameter_values={"g":contrast, "k":np.pi},
                                     linear_method="cg", preconditioner="jacobi", relative_tolerance=1e-11,
                                     absolute_tolerance=1e-13, maximum_iterations=4000)
    x = np.asarray(problem["coordinates"])
    u = np.asarray(problem["solution"])
    if case == "modes":
        u = u[:, cells//2]
        reference = np.sin(np.pi*x) + contrast*np.sin(2*np.pi*x)
    elif case == "capacitor":
        reference = x if contrast == 0 else np.log1p(contrast*x)/np.log1p(contrast)
    else:
        reference = 1-x if contrast == 0 else np.sinh(contrast*(1-x))/np.sinh(contrast)
    return zip(x,u,reference)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("modes", "capacitor", "screen"))
    parser.add_argument("--cells", type=int, default=16, help="Even number of cells per side, 4 to 128")
    parser.add_argument("--contrast", type=float, default=1., help="Second mode amplitude, material gradient, or reciprocal screening length")
    args=parser.parse_args()
    if args.cells<4 or args.cells>128 or args.cells%2:
        parser.error("choose an even --cells value from 4 to 128")
    if not np.isfinite(args.contrast) or abs(args.contrast)>30 or (args.case!="modes" and args.contrast<0):
        parser.error("choose the finite parameter range shown in the course")
    writer=csv.writer(sys.stdout,lineterminator="\n")
    writer.writerow(("position_m","OneSimulation_potential_V","continuous_reference_V"))
    writer.writerows(solve(args.case,args.cells,args.contrast))
