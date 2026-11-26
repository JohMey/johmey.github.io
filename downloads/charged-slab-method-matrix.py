"""OneSimulation example: the field of a uniformly charged slab through every method combination.
Requires the installed, activated OneSimulation runtime.

One physics declaration, Gauss's law for a dielectric slab carrying a uniform space charge between
two electrodes, is solved with every applicable combination of discretization family (finite
differences, finite volumes, continuous, Nitsche and discontinuous finite elements) and linear
solver. Each run is graded against the exact parabolic potential; combinations the library refuses
are reported as such.
"""

# # A uniformly charged slab: one model, every method
#
# A dielectric layer of thickness $d=1\,$mm and relative permittivity $\varepsilon_r=4$ carries a
# uniform space charge $\rho=2\cdot10^{-3}\,$C/m³. Its faces are electrodes at $\varphi(0)=0$ and
# $\varphi(d)=V=10\,$V. Gauss's law in terms of the potential,
#
# $$-\partial_x\big(\varepsilon\,\partial_x\varphi\big)=\rho,\qquad \varphi(0)=0,\quad
# \varphi(d)=V,\qquad E=-\partial_x\varphi,$$
#
# is declared once. Everything numerical is an argument of `Numerics`: the discretization family
# with its diffusion operator and boundary treatment, and the linear solver of the stationary
# system.
#
# **Reference.** Integrating twice gives the parabola and the linear field
#
# $$\varphi(x)=\frac{\rho}{2\varepsilon}\,x(d-x)+V\,\frac{x}{d},\qquad
# E(x)=-\frac{\rho}{2\varepsilon}\,(d-2x)-\frac{V}{d}.$$
#
# The displacement $D=\varepsilon E$ grows by $\rho d$ across the layer: the charge balance of the
# slab.
#
# **Grading.** Every run is evaluated at the cell centres $x_i$: the largest potential error, the
# largest field error, and the charge balance $\varepsilon\,[E(d)-E(0)]/(\rho d)-1$ from the field at
# the two electrodes.

import time

import numpy as np
from onesim.cad.d1 import Line
from onesim.materials import VacuumPermittivity, regionwise
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import (
    CentralDifference,
    CentralLaplacian,
    FiniteDifference,
    PolynomialDifference,
    PolynomialLaplacian,
)
from onesim.numerics.fem import (
    CIP,
    DiscontinuousLagrange,
    FiniteElement,
    Lagrange,
    Nitsche,
    SymmetricInteriorPenalty,
)
from onesim.numerics.fvm import CentralDifference as CellCentralDifference
from onesim.numerics.fvm import FiniteVolume, LeastSquaresGradient, TwoPointDiffusion
from onesim.numerics.linear import (
    GMRES,
    ILU,
    AlgebraicMultigrid,
    AutomaticLinearSolver,
    ConjugateGradient,
    FlexibleGMRES,
    RightGMRES,
    SparseDirect,
)
from onesim.physics.electrostatics import FixedPotential, PoissonEquation
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# Nothing in this cell names a numerical method.

depth, voltage, charge = 1e-3, 10.0, 2e-3
epsilon = 4 * float(VacuumPermittivity().evaluate())
cells, coarse_cells = 200, 100
layer = Line(start=(0.0, 0.0, 0.0), end=(depth, 0.0, 0.0))
gauss = PoissonEquation(
    domain=layer.line,
    permittivity=regionwise((), default=epsilon, unit="F/m"),
    charge_density=charge,
)
problem = Problem(
    name="Charged slab",
    geometry=layer,
    physics=(gauss,),
    boundary_conditions=(
        FixedPotential(physics=gauss, boundary=layer.boundaries.start, value=0.0),
        FixedPotential(physics=gauss, boundary=layer.boundaries.end, value=voltage),
    ),
)
meshes = {n: StructuredLineMesher(cells=n).generate(layer) for n in (cells, coarse_cells)}
centres = {n: mesh.cell_centers for n, mesh in meshes.items()}
electrodes = np.array([[0.0, 0.0, 0.0], [depth, 0.0, 0.0]])
print(
    f"{meshes[cells].cell_count} cells of width {1e6 * depth / cells:.1f} µm "
    f"(mesh check: {meshes[coarse_cells].cell_count} cells)"
)

# ## The exact solution
#
# The parabola and the field at every cell centre and on a fine plotting grid.

fine = np.linspace(0.0, depth, 801)
exact = {
    key: (
        charge / (2 * epsilon) * x * (depth - x) + voltage * x / depth,
        -charge / (2 * epsilon) * (depth - 2 * x) - voltage / depth,
    )
    for key, x in {**{n: c[:, 0] for n, c in centres.items()}, "fine": fine}.items()
}
print(
    f"peak potential {exact['fine'][0].max():.4f} V at x = {1e3 * fine[exact['fine'][0].argmax()]:.3f} mm, "
    f"field at the electrodes {exact['fine'][1][0]:.4e} and {exact['fine'][1][-1]:.4e} V/m"
)

# ## The method catalogue
#
# Finite differences with central, fourth- and sixth-order polynomial Laplacians (each with the
# first derivative of the same order for the field); finite volumes with the two-point flux and the
# corrected central flux; continuous Lagrange elements of degree 1 to 3 with strong electrode
# values, symmetric and nonsymmetric Nitsche terms and continuous interior penalty; discontinuous
# elements of degree 1 to 3 with the SIPG flux at two penalty levels.

spatial = {
    "FDM central, 2nd order": FiniteDifference(
        diffusion=CentralLaplacian(), gradient=CentralDifference()
    ),
    **{
        f"FDM polynomial, {order}th order": FiniteDifference(
            diffusion=PolynomialLaplacian(accuracy_order=order),
            gradient=PolynomialDifference(accuracy_order=order),
        )
        for order in (4, 6)
    },
    "FVM two-point flux": FiniteVolume(diffusion=TwoPointDiffusion()),
    "FVM corrected central flux": FiniteVolume(
        diffusion=CellCentralDifference(), gradient=LeastSquaresGradient()
    ),
    **{
        f"FEM CG P{degree}": FiniteElement(element=Lagrange(degree))
        for degree in (1, 2, 3)
    },
    "FEM CG P2 + symmetric Nitsche": FiniteElement(
        element=Lagrange(2), boundary_enforcement=Nitsche()
    ),
    "FEM CG P2 + nonsymmetric Nitsche": FiniteElement(
        element=Lagrange(2), boundary_enforcement=Nitsche(symmetry=-1.0)
    ),
    "FEM CG P2 + continuous interior penalty": FiniteElement(
        element=Lagrange(2), stabilization=CIP(penalty=0.01)
    ),
    **{
        f"FEM DG P{degree} SIPG": FiniteElement(
            element=DiscontinuousLagrange(degree), diffusion=SymmetricInteriorPenalty()
        )
        for degree in (1, 2, 3)
    },
    "FEM DG P2 SIPG, penalty 40": FiniteElement(
        element=DiscontinuousLagrange(2),
        diffusion=SymmetricInteriorPenalty(penalty=40.0),
    ),
}

# The stationary system is linear. Every scheme is solved with a sparse LU factorization, with
# Krylov solvers with and without preconditioning, and with the automatic choice. Conjugate
# gradients and the symmetric algebraic multigrid need a certified symmetric positive definite
# matrix. The one-sided boundary closures of the higher-order differences and the Nitsche terms do
# not provide one (the symmetric Nitsche matrix is definite only above a penalty threshold that is
# not certified); these pairs are marked "n/a" and listed under "Methods that do not apply".

linear_solvers = {
    "sparse LU": SparseDirect(),
    "GMRES": GMRES(),
    "GMRES + ILU": GMRES(preconditioner=ILU()),
    "GMRES + algebraic multigrid": GMRES(preconditioner=AlgebraicMultigrid()),
    "right GMRES": RightGMRES(),
    "flexible GMRES": FlexibleGMRES(),
    "conjugate gradients": ConjugateGradient(),
    "CG + algebraic multigrid": ConjugateGradient(preconditioner=AlgebraicMultigrid()),
    "automatic": AutomaticLinearSolver(),
}

# ## The test matrix
#
# **Space × linear solver**: every spatial scheme with every linear solver on 200 cells. The
# representative schemes are repeated on 100 cells for the mesh check.

symmetric_only = {"GMRES + algebraic multigrid", "conjugate gradients", "CG + algebraic multigrid"}
not_symmetric = {
    "FDM polynomial, 4th order",
    "FDM polynomial, 6th order",
    "FEM CG P2 + symmetric Nitsche",
    "FEM CG P2 + nonsymmetric Nitsche",
}
runs = [
    dict(matrix="space x linear solver", row=row, column=column, cells=cells,
         spatial=scheme, linear=solver)
    for row, scheme in spatial.items()
    for column, solver in linear_solvers.items()
    if not (row in not_symmetric and column in symmetric_only)
]

# ## A solver that must fail, and its stabilized counterpart
#
# The condition number of the stiffness matrix grows like $h^{-2}$ and with the element degree.
# Unpreconditioned GMRES then stalls within its iteration budget on the higher-order elements. Its
# stabilized counterpart adds exactly one ingredient, an incomplete LU preconditioner; it is shown
# behind a grey hatching. Unpreconditioned conjugate gradients can stall the same way on the
# cubic discontinuous elements; their counterpart adds the algebraic multigrid preconditioner. A
# counterpart is shown only where the declared solver failed.

stabilized_solver = {
    "GMRES": GMRES(preconditioner=ILU()),
    "right GMRES": RightGMRES(preconditioner=ILU()),
    "flexible GMRES": FlexibleGMRES(preconditioner=ILU()),
    "conjugate gradients": ConjugateGradient(preconditioner=AlgebraicMultigrid()),
}
runs += [
    {**run, "variant": "stabilized", "linear": stabilized_solver[run["column"]],
     "note_stabilized": f"{run['column']} + "
     + ("algebraic multigrid" if run["column"] == "conjugate gradients" else "ILU")
     + " preconditioner"}
    for run in list(runs)
    if run["column"] in stabilized_solver
]
representative = [
    "FDM central, 2nd order",
    "FDM polynomial, 4th order",
    "FVM two-point flux",
    "FEM CG P1",
    "FEM DG P1 SIPG",
]
runs += [
    {**run, "matrix": "mesh check", "cells": coarse_cells}
    for run in runs
    if run["column"] == "sparse LU" and run["row"] in representative
]
print(f"{len(runs)} runs")

# ## Run everything
#
# One loop, the same problem object every time. A combination the library refuses is recorded with
# its validation message; a Krylov solver that does not converge is recorded as failed.

for run in runs:
    simulation = Simulation(
        problem=problem,
        mesh=meshes[run["cells"]],
        numerics=Numerics(spatial=run["spatial"], linear_solver=run["linear"]),
    )
    started = time.perf_counter()
    check = simulation.validate()
    run.update(status="rejected", note="", errors=(np.nan,) * 3)
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = simulation.solve_stationary()
            run["potential"] = result.field(gauss.potential).evaluate(centres[run["cells"]]).ravel()
            run["field"] = result.field(gauss.electric_field).evaluate(centres[run["cells"]])[:, 0]
            boundary_field = result.field(gauss.electric_field).evaluate(electrodes)[:, 0]
            run["errors"] = (
                np.abs(run["potential"] - exact[run["cells"]][0]).max(),
                np.abs(run["field"] - exact[run["cells"]][1]).max(),
                epsilon * (boundary_field[1] - boundary_field[0]) / (charge * depth) - 1,
            )
            run["status"] = "ran" if np.isfinite(run["potential"]).all() else "failed"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "failed" if "converge" in message or "maximum iteration" in message else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    print(
        f"{run['matrix'][:10]:10s} {run['row'][:40]:40s} {run['column'][:28]:28s} "
        f"{run['status']:8s} {run['seconds']:6.2f} s  "
        + (
            f"max|Δφ| {run['errors'][0]:.2e} V  max|ΔE| {run['errors'][1]:.2e} V/m  "
            f"charge balance {run['errors'][2]:+.1e}"
            if run["status"] == "ran"
            else run["note"][:90]
        )
    )

# ## Space × linear solver
#
# Left: the largest potential error, which does not depend on the solver once it converges. Right:
# the wall time of each solve.

matrix = [
    r for r in runs if r["matrix"] == "space x linear solver" and r.get("variant") is None
]
grid = {(r["row"], r["column"]): r for r in matrix}
substitute = {
    (r["row"], r["column"]): r for r in runs if r.get("variant") == "stabilized"
}
hatch = {
    key
    for key, r in grid.items()
    if r["status"] != "ran" and substitute.get(key, {}).get("status") == "ran"
}
cell = {key: substitute[key] if key in hatch else r for key, r in grid.items()}
rows, columns = list(spatial), list(linear_solvers)
fig = plots.figure(1, 2, size=(19, 9))
for panel, measure, label, title in (
    (fig[0], "error", "max |φ - φ_exact| [V]", f"Potential error, {cells} cells"),
    (fig[1], "seconds", "wall time [s]", "Wall time of the stationary solve"),
):
    plots.matrix(
        [
            [
                (
                    (
                        max(cell[(row, c)]["errors"][0], 1e-16)
                        if measure == "error"
                        else cell[(row, c)]["seconds"]
                    )
                    if (row, c) in cell and cell[(row, c)]["status"] == "ran"
                    else np.nan
                )
                for c in columns
            ]
            for row in rows
        ],
        rows,
        columns,
        scale="log",
        label=label,
        text=[
            [
                (
                    (
                        (
                            f"{cell[(row, c)]['errors'][0]:.1e}"
                            if measure == "error"
                            else f"{cell[(row, c)]['seconds']:.2f} s"
                        )
                        if cell[(row, c)]["status"] == "ran"
                        else cell[(row, c)]["status"]
                    )
                    if (row, c) in cell
                    else "n/a"
                )
                for c in columns
            ]
            for row in rows
        ],
        hatched=[[(row, c) in hatch for c in columns] for row in rows],
        notes=(
            [
                "Hatched: the unpreconditioned Krylov solver stalls on the ill-conditioned matrix; "
                "colour and value show it with one preconditioner added (ILU for the GMRES "
                "family, algebraic multigrid for conjugate gradients)."
            ]
            if hatch and measure == "error"
            else None
        ),
        title=title,
        ax=panel,
    )
plots.show()

# Potential error, field error and charge balance of every scheme with the sparse LU solve.
# Continuous elements reproduce the parabola exactly at the nodes from degree 2 on (the field is
# linear); the cell-centred finite differences and volumes close at the electrodes with a half cell
# and are second order.

print(f"{'scheme':42s} {'max|Δφ| [V]':>12s} {'max|ΔE| [V/m]':>14s} {'charge bal.':>12s}")
for row in rows:
    r = grid[(row, "sparse LU")]
    if r["status"] == "ran":
        print(f"{row:42s} {r['errors'][0]:12.2e} {r['errors'][1]:14.2e} {r['errors'][2]:+12.1e}")

# ## Profiles of representative schemes
#
# Potential and field through the layer against the exact solution.

shown = [grid[(row, "sparse LU")] for row in representative if grid[(row, "sparse LU")]["status"] == "ran"]
fig = plots.figure(1, 2, size=(15, 5))
for panel, index, key, label in (
    (fig[0], 0, "potential", "potential φ [V]"),
    (fig[1], 1, "field", "field E [V/m]"),
):
    plots.line(
        y={r["row"]: (1e3 * centres[cells][:, 0], r[key]) for r in shown},
        reference={"exact": (1e3 * fine, exact["fine"][index])},
        xlabel="position x [mm]",
        ylabel=label,
        title=label.split(" [")[0].capitalize() + " across the layer",
        legend="best" if index == 0 else False,
        ax=panel,
    )
plots.show()

# ## Mesh check: 200 against 100 cells
#
# A second-order scheme reduces the potential error about fourfold; schemes that are exact for
# the parabola stay at round-off.

coarse = {r["row"]: r for r in runs if r["matrix"] == "mesh check"}
print(f"{'scheme':40s} {'err 100':>10s} {'err 200':>10s} {'ratio':>7s}")
for row, r in coarse.items():
    fine_run = grid[(row, "sparse LU")]
    if r["status"] == "ran" and fine_run["status"] == "ran":
        print(
            f"{row:40s} {r['errors'][0]:10.3e} {fine_run['errors'][0]:10.3e} "
            f"{r['errors'][0] / max(fine_run['errors'][0], 1e-300):7.2f}"
        )

# ## What the library refused, and what failed

for status in ("rejected", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(f"  {r['row'][:40]:40s} {r['column'][:28]:28s} {r['note'][:90]}")
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs):.0f} s in total"
)

# ## Methods that do not apply to this problem
#
# Gauss's law is stationary and linear: time integrators have nothing to advance and a nonlinear
# solver nothing to iterate. Conjugate gradients and the symmetric algebraic multigrid need a
# certified symmetric positive definite matrix: the one-sided closures of the fourth- and
# sixth-order differences make the matrix nonsymmetric, the nonsymmetric Nitsche term does too, and
# the symmetric Nitsche matrix is definite only above an uncertified penalty threshold. Conjugate
# gradients with an incomplete LU preconditioner are excluded because ILU is not symmetric. Convective fluxes, reconstructions, limiters and SUPG/GLS act on
# transport terms, which an electrostatic field does not have.
#
# ## Going further
#
# Make the charge density a function `lambda x: ...` or the permittivity regionwise and the matrix
# code stays as it is.
