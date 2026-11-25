"""OneSimulation example: the magnetic field of a current-carrying slab through every method combination.
Requires the installed, activated OneSimulation runtime.

One physics declaration, magnetostatics of a conducting slab with a uniform current between two
magnetically insulating walls, is solved on its line cross-section with every applicable
combination of discretization family (finite differences, finite volumes, continuous Nitsche and
discontinuous finite elements) and linear solver. Each run is graded against the closed-form
vector potential; combinations the library refuses are reported as such.
"""

# # A current-carrying slab: one model, every method
#
# A copper busbar of thickness $2a=20\,$mm and large height and length carries the current density
# $J_z=10^6\,$A/m² along $z$. Walls at $x=\pm b$, $b=30\,$mm, are magnetically insulating
# ($A_z=0$, so no flux leaves through them). Across the slab the fields depend on $x$ only, and
# Ampère's law reduces to one equation for the out-of-plane potential:
#
# $$-\partial_x\Big(\frac1\mu\,\partial_x A_z\Big)=J_z(x),\qquad A_z(\pm b)=0,\qquad
# \mathbf B=\nabla\times(A_z\mathbf e_z)=\big(0,\,-\partial_x A_z,\,0\big).$$
#
# `Magnetostatics` on a line declares exactly this: the line is the cross-section of the slab and
# the current comes from a `RegionCurrent` whose region is the conductor. Everything numerical is
# an argument of `Numerics`.
#
# **Reference.** Integrating twice with $\mu=\mu_0$ throughout gives
#
# $$A_z(x)=\mu_0J_z\begin{cases}\tfrac12(a^2-x^2)+a(b-a), & |x|\le a,\\ a\,(b-|x|), & a<|x|\le b,
# \end{cases}\qquad B_y(x)=\mu_0J_z\begin{cases}x, & |x|\le a,\\ a\,\operatorname{sgn}x, & |x|>a.
# \end{cases}$$
#
# Ampère's law in integral form checks the result: $B_y(b)-B_y(-b)=\mu_0 J_z\,2a$, the enclosed
# current per unit height.
#
# **Grading.** Every run is evaluated at the cell centres $x_i$: the largest error of $A_z$
# relative to its peak, the largest error of $B_y$, and the Ampère balance
# $[B_y(b)-B_y(-b)]/(2\mu_0J_za)-1$.

import time

import numpy as np
from onesim.cad.d1 import Line
from onesim.cad.d3 import Box
from onesim.materials import VacuumPermeability, regionwise
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
from onesim.numerics.fvm import FiniteVolume, TwoPointDiffusion
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
from onesim.physics.magnetostatics import MagneticInsulation, Magnetostatics, RegionCurrent
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# Nothing in this cell names a numerical method. The mesh places cell faces on the conductor
# surfaces $x=\pm a$.

half_width, wall, density = 0.01, 0.03, 1e6
mu0 = float(VacuumPermeability().evaluate())
cells, coarse_cells = 240, 120
section = Line(start=(-wall, 0.0, 0.0), end=(wall, 0.0, 0.0))
conductor = Box(origin=(-half_width, -1.0, -1.0), length=2 * half_width, width=2.0, height=2.0)
busbar = Magnetostatics(
    domain=section.line,
    permeability=regionwise((), default=mu0, unit="H/m"),
    current_sources=(RegionCurrent(region=conductor, density=(0.0, 0.0, density)),),
)
problem = Problem(
    name="Current-carrying slab",
    geometry=section,
    physics=(busbar,),
    boundary_conditions=(MagneticInsulation(physics=busbar, boundary=section.boundaries.all),),
)
meshes = {n: StructuredLineMesher(cells=n).generate(section) for n in (cells, coarse_cells)}
centres = {n: mesh.cell_centers for n, mesh in meshes.items()}
walls = np.array([[-wall, 0.0, 0.0], [wall, 0.0, 0.0]])
print(
    f"{meshes[cells].cell_count} cells of width {1e3 * 2 * wall / cells:.2f} mm "
    f"(mesh check: {meshes[coarse_cells].cell_count} cells)"
)

# ## The exact solution

fine = np.linspace(-wall, wall, 1201)
exact = {
    key: (
        mu0 * density * np.where(
            abs(x) <= half_width,
            0.5 * (half_width**2 - x**2) + half_width * (wall - half_width),
            half_width * (wall - abs(x)),
        ),
        mu0 * density * np.where(abs(x) <= half_width, x, half_width * np.sign(x)),
    )
    for key, x in {**{n: c[:, 0] for n, c in centres.items()}, "fine": fine}.items()
}
print(
    f"peak A_z {exact['fine'][0].max():.4e} Wb/m, field outside the conductor "
    f"{mu0 * density * half_width * 1e3:.4f} mT"
)

# ## The method catalogue
#
# Finite differences with central, fourth- and sixth-order Laplacians (each with the first
# derivative of the same order for $B$); finite volumes with the two-point flux; continuous
# Lagrange elements of degree 1 to 3 whose wall values enter through symmetric or nonsymmetric
# Nitsche terms, also with continuous interior penalty; discontinuous elements of degree 1 to 3
# with the SIPG flux at two penalty levels.

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
    **{
        f"FEM CG P{degree} + symmetric Nitsche": FiniteElement(
            element=Lagrange(degree), boundary_enforcement=Nitsche()
        )
        for degree in (1, 2, 3)
    },
    "FEM CG P2 + nonsymmetric Nitsche": FiniteElement(
        element=Lagrange(2), boundary_enforcement=Nitsche(symmetry=-1.0)
    ),
    "FEM CG P2 + Nitsche + continuous interior penalty": FiniteElement(
        element=Lagrange(2), boundary_enforcement=Nitsche(), stabilization=CIP(penalty=0.01)
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
# matrix; the one-sided closures of the higher-order differences and the Nitsche terms do not
# provide one, and these pairs are marked "n/a".

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
symmetric_only = {"GMRES + algebraic multigrid", "conjugate gradients", "CG + algebraic multigrid"}
not_symmetric = {
    "FDM polynomial, 4th order",
    "FDM polynomial, 6th order",
    *(row for row in spatial if "Nitsche" in row),
}

# ## The test matrix, and a solver that must fail
#
# **Space × linear solver** on 240 cells; representative schemes again on 120 cells. The condition
# number grows like $h^{-2}$ and with the element degree, so unpreconditioned GMRES may stall within
# its iteration budget. Its stabilized counterpart adds exactly one ingredient, an incomplete LU
# preconditioner, and is shown behind a grey hatching.

runs = [
    dict(matrix="space x linear solver", row=row, column=column, cells=cells,
         spatial=scheme, linear=solver)
    for row, scheme in spatial.items()
    for column, solver in linear_solvers.items()
    if not (row in not_symmetric and column in symmetric_only)
]
stabilized_solver = {
    "GMRES": GMRES(preconditioner=ILU()),
    "right GMRES": RightGMRES(preconditioner=ILU()),
    "flexible GMRES": FlexibleGMRES(preconditioner=ILU()),
}
runs += [
    {**run, "variant": "stabilized", "linear": stabilized_solver[run["column"]]}
    for run in list(runs)
    if run["column"] in stabilized_solver
]
representative = [
    "FDM central, 2nd order",
    "FDM polynomial, 4th order",
    "FVM two-point flux",
    "FEM CG P1 + symmetric Nitsche",
    "FEM DG P1 SIPG",
]
runs += [
    {**run, "matrix": "mesh check", "cells": coarse_cells}
    for run in runs
    if run["column"] == "sparse LU" and run["row"] in representative
]
print(f"{len(runs)} runs")

# ## Run everything

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
            run["potential"] = (
                result.field(busbar.out_of_plane_potential).evaluate(centres[run["cells"]]).ravel()
            )
            run["field"] = result.field(busbar.magnetic_flux_density).evaluate(
                centres[run["cells"]]
            )[:, 1]
            wall_field = result.field(busbar.magnetic_flux_density).evaluate(walls)[:, 1]
            run["errors"] = (
                np.abs(run["potential"] - exact[run["cells"]][0]).max() / exact["fine"][0].max(),
                np.abs(run["field"] - exact[run["cells"]][1]).max(),
                (wall_field[1] - wall_field[0]) / (2 * mu0 * density * half_width) - 1,
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
        f"{run['matrix'][:10]:10s} {run['row'][:46]:46s} {run['column'][:28]:28s} "
        f"{run.get('variant') or '':10s} {run['status']:8s} {run['seconds']:6.2f} s  "
        + (
            f"rel. max|ΔA| {run['errors'][0]:.2e}  max|ΔB| {run['errors'][1]:.2e} T  "
            f"Ampère {run['errors'][2]:+.1e}"
            if run["status"] == "ran"
            else run["note"][:90]
        )
    )

# ## Space × linear solver
#
# Left: the relative error of $A_z$; right: the wall time of each solve.

matrix = [
    r for r in runs if r["matrix"] == "space x linear solver" and r.get("variant") is None
]
grid = {(r["row"], r["column"]): r for r in matrix}
substitute = {(r["row"], r["column"]): r for r in runs if r.get("variant") == "stabilized"}
hatch = {
    key
    for key, r in grid.items()
    if r["status"] != "ran" and substitute.get(key, {}).get("status") == "ran"
}
cell = {key: substitute[key] if key in hatch else r for key, r in grid.items()}
rows, columns = list(spatial), list(linear_solvers)
fig = plots.figure(1, 2, size=(19, 9))
for panel, measure, label, title in (
    (fig[0], "error", "max |A - A_exact| / max A", f"Vector potential error, {cells} cells"),
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
                "Hatched: unpreconditioned GMRES stalls on the ill-conditioned matrix; colour and "
                "value show the same solver with an ILU preconditioner."
            ]
            if hatch and measure == "error"
            else None
        ),
        title=title,
        ax=panel,
    )
plots.show()

# Potential error, field error and Ampère balance of every scheme with the sparse LU solve. The
# potential is piecewise quadratic, so continuous and discontinuous elements of degree 2 and more
# reproduce it to round-off on a mesh that resolves the conductor surfaces.

print(f"{'scheme':48s} {'rel |ΔA|':>10s} {'max|ΔB| [T]':>12s} {'Ampère':>10s}")
for row in rows:
    r = grid[(row, "sparse LU")]
    if r["status"] == "ran":
        print(f"{row:48s} {r['errors'][0]:10.2e} {r['errors'][1]:12.2e} {r['errors'][2]:+10.1e}")

# ## Profiles of representative schemes

shown = [grid[(row, "sparse LU")] for row in representative if grid[(row, "sparse LU")]["status"] == "ran"]
fig = plots.figure(1, 2, size=(15, 5))
for panel, index, key, label, scale in (
    (fig[0], 0, "potential", "A_z [µWb/m]", 1e6),
    (fig[1], 1, "field", "B_y [mT]", 1e3),
):
    plots.line(
        y={r["row"]: (1e3 * centres[cells][:, 0], scale * r[key]) for r in shown},
        reference={"closed form": (1e3 * fine, scale * exact["fine"][index])},
        xlabel="position x [mm]",
        ylabel=label,
        title=("Out-of-plane vector potential" if index == 0 else "Flux density across the slab"),
        vlines=[-1e3 * half_width, 1e3 * half_width],
        legend="best" if index == 0 else False,
        ax=panel,
    )
plots.show()

# ## Mesh check: 240 against 120 cells

coarse = {r["row"]: r for r in runs if r["matrix"] == "mesh check"}
print(f"{'scheme':40s} {'err 120':>10s} {'err 240':>10s} {'ratio':>7s}")
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
        print(f"  {r['row'][:46]:46s} {r['column'][:28]:28s} {r.get('variant') or '':10s} {r['note'][:90]}")
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs):.0f} s in total"
)

# ## Methods that do not apply to this problem
#
# The slab problem is stationary and linear: time integrators and nonlinear solvers have nothing to
# do (a `BHCurve` material response would make it nonlinear). Conjugate gradients and the symmetric
# algebraic multigrid need a certified symmetric positive definite matrix, which the one-sided
# higher-order difference closures and the Nitsche wall terms do not provide. On a line the
# corrected central cell flux coincides with the two-point flux, so finite volumes appear once.
# The volume formulations of `Magnetostatics` (edge elements, finite integration) need a 3D
# region; the line model is their exact reduction for a slab.
#
# ## Going further
#
# Add a second `RegionCurrent` with the opposite current for a go-and-return busbar pair, or move
# the walls; the matrix code stays as it is.
