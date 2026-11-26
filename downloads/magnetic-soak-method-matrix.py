"""OneSimulation example: a magnetic field soaking into a conducting slab through every method combination.
Requires the installed, activated OneSimulation runtime.

One physics declaration, magnetic diffusion into a copper plate whose faces are suddenly exposed
to a uniform field, is solved with every applicable combination of discretization family (finite
differences, finite volumes, Nitsche-continuous and discontinuous finite elements), time
integrator, nonlinear solver and linear solver. Each run is graded against the exact Fourier
series; combinations the library refuses are reported as such.
"""

# # Magnetic diffusion into a conducting slab: one model, every method
#
# A copper plate of thickness $L=10\,$mm (conductivity $\sigma=5.96\cdot10^7\,$S/m,
# permeability $\mu_0$) is field-free at $t=0$. Then a uniform tangential field $B_0=0.1\,$T is
# switched on outside both faces. Eddy currents delay its entry: in the magnetoquasistatic limit
# the induction obeys a diffusion equation with the magnetic diffusivity $\eta=1/(\mu_0\sigma)$,
#
# $$\partial_t B_z=\eta\,\partial_x^2 B_z,\qquad B_z(0,t)=B_z(L,t)=B_0,\qquad B_z(x,0)=0,$$
#
# with $\eta\approx0.0134\,$m²/s. `MagneticDiffusion` declares this balance once. Everything
# numerical is an argument of `Numerics`.
#
# **Reference.** The odd sine modes of the initial deficit decay independently (Knoepfel,
# *Magnetic Fields*, §4.2; the same series as for heat conduction):
#
# $$B_z(x,t)=B_0\Big[1-\sum_{k\ \text{odd}}\frac{4}{k\pi}\,\sin\frac{k\pi x}{L}\,
# \exp\!\Big(-\frac{k^2\pi^2\eta t}{L^2}\Big)\Big].$$
#
# Early on each face sees a half-space and $B_z\approx B_0\,\mathrm{erfc}\big(x/(2\sqrt{\eta t})\big)$;
# the series contains that limit.
#
# **Grading.** At $t=0.4\,$ms (diffusion number $\eta t/L^2\approx0.053$) every run is evaluated
# at the cell centres $x_i$: the mean absolute error $\sum_i|B_h(x_i)-B(x_i)|\,\Delta x/L$ and the
# largest deviation; the wall time covers validation, binding and time stepping.

import time

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import CentralLaplacian, FiniteDifference, PolynomialLaplacian
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
from onesim.numerics.nonlinear import (
    JFNK,
    DampedNewton,
    FixedPoint,
    Newton,
    QuasiNewton,
    TrustRegionNewton,
)
from onesim.numerics.time import (
    BDF,
    AdaptiveBDF,
    AdaptiveRungeKutta45,
    BackwardEuler,
    ClassicalRungeKutta4,
    ForwardEuler,
    TimeMethod,
    TimeStepLimits,
)
from onesim.materials import VacuumPermeability
from onesim.physics.electromagnetics import MagneticDiffusion
from onesim.simulation import Problem, Simulation
from onesim.simulation.problem_definition.boundaries import PrescribedValueBoundary
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# Nothing in this cell names a numerical method.

thickness, field, end_time = 0.01, 0.1, 4e-4
cells, coarse_cells, time_cells = 200, 100, 50
conductivity, mu0 = 5.96e7, float(VacuumPermeability().evaluate())
diffusivity = 1 / (mu0 * conductivity)
plate = Line(start=(0.0, 0.0, 0.0), end=(thickness, 0.0, 0.0))
copper = MagneticDiffusion(domain=plate.line, permeability=mu0, conductivity=conductivity)
problem = Problem(
    name="Field soaking into a copper plate",
    geometry=plate,
    physics=(copper,),
    boundary_conditions=(
        PrescribedValueBoundary(
            physics=copper, boundary=plate.boundaries.all, value={"magnetic_induction": field}
        ),
    ),
    initial_conditions=(InitialValue(copper.magnetic_induction, 0.0),),
)
meshes = {
    n: StructuredLineMesher(cells=n).generate(plate) for n in (cells, coarse_cells, time_cells)
}
centres = {n: mesh.cell_centers for n, mesh in meshes.items()}
print(
    f"{meshes[cells].cell_count} cells of width {1e3 * thickness / cells:.3f} mm "
    f"(mesh check: {meshes[coarse_cells].cell_count} cells; time-integrator cut: "
    f"{meshes[time_cells].cell_count} cells), magnetic diffusivity {diffusivity:.4e} m²/s, "
    f"diffusion number {diffusivity * end_time / thickness**2:.4f}"
)

# ## The exact solution
#
# The series is summed with 2000 terms at every cell centre and on a fine plotting grid.

modes = np.arange(1, 4001, 2)[:, None]
fine = np.linspace(0.0, thickness, 801)
exact = {
    key: field
    * (
        1
        - (
            4 / (np.pi * modes)
            * np.sin(modes * np.pi * points / thickness)
            * np.exp(-((modes * np.pi / thickness) ** 2) * diffusivity * end_time)
        ).sum(axis=0)
    )
    for key, points in {**{n: c[:, 0] for n, c in centres.items()}, "fine": fine}.items()
}
print(
    f"mid-plane induction {np.interp(thickness / 2, fine, exact['fine']):.5f} T at t = {1e3 * end_time} ms"
)

# ## The method catalogue
#
# Finite differences with central, fourth- and sixth-order Laplacians; finite volumes with the
# two-point flux; continuous Lagrange elements of degree 1 to 3 whose face values enter through
# symmetric or nonsymmetric Nitsche terms, also with continuous interior penalty; discontinuous
# elements of degree 1 to 3 with the SIPG flux at two penalty levels. The induction balance has no
# gradient-derived output, so the difference schemes take no first derivative.

spatial = {
    "FDM central, 2nd order": FiniteDifference(diffusion=CentralLaplacian()),
    **{
        f"FDM polynomial, {order}th order": FiniteDifference(
            diffusion=PolynomialLaplacian(accuracy_order=order)
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

# Time integrators: the explicit one-step and multistep methods, implicit Runge-Kutta,
# Adams-Moulton and BDF methods, Rosenbrock methods and the adaptive integrators. Implicit stages
# take a Newton solver with a sparse direct solve; the linearly implicit Rosenbrock stages take the
# sparse direct solve alone. Explicit methods advance at their stable step (safety factor 0.5),
# implicit ones at $\Delta t=5\,$µs.

integrators = {
    "forward Euler": ForwardEuler(),
    "SSP-RK2": TimeMethod("ssprk2"),
    "SSP-RK3": TimeMethod("ssprk3"),
    "Heun": TimeMethod("heun"),
    "explicit midpoint": TimeMethod("explicit_midpoint"),
    "classical RK4": ClassicalRungeKutta4(),
    "Adams-Bashforth 2": TimeMethod("adams_bashforth2"),
    "ABM2 predictor-corrector": TimeMethod("predictor_corrector_abam2"),
    "Heun-Euler (embedded)": TimeMethod("embedded_heun_euler"),
    "leapfrog": TimeMethod("leapfrog"),
    "adaptive RK45": AdaptiveRungeKutta45(relative_tolerance=1e-5, absolute_tolerance=1e-7),
    "backward Euler": BackwardEuler(),
    "Crank-Nicolson": TimeMethod("crank_nicolson"),
    "theta = 0.75": TimeMethod("theta_075"),
    "implicit midpoint": TimeMethod("implicit_rk_midpoint"),
    "SDIRK2": TimeMethod("dirk_sdirk2"),
    "Adams-Moulton 2": TimeMethod("adams_moulton_2"),
    "Adams-Moulton 3": TimeMethod("adams_moulton_3"),
    "Adams-Moulton 4": TimeMethod("adams_moulton_4"),
    "BDF2 (variable step)": BDF(order=2),
    "BDF3 (variable step)": BDF(order=3),
    **{f"BDF{k} (Gear)": TimeMethod(f"bdf{k}_gear") for k in (2, 3, 4, 5, 6)},
    "Rosenbrock-Euler": TimeMethod("rosenbrock_euler"),
    "Rosenbrock-Wanner Euler": TimeMethod("rosenbrock_wanner_euler"),
    "adaptive BDF": AdaptiveBDF(relative_tolerance=1e-4, absolute_tolerance=1e-6),
}
implicit_step = 5e-6
newton = Newton(linear_solver=SparseDirect())

# ## Schemes that must fail, and their stabilized counterparts
#
# The diffusion operator has a real, negative spectrum reaching $-\lambda_{\max}\sim-12\alpha/h^2$.
# Adams-Moulton 3 and 4 are not A-stable: their stability interval on the negative axis ends at
# $\lambda\Delta t\approx6$ and $3$, while $\lambda_{\max}\Delta t$ is several hundred at
# $\Delta t=5\,$µs. Leapfrog has no damping: every decaying mode is amplified at any step. A
# fixed-point (Picard) iteration of an implicit step contracts only for $\lambda_{\max}\Delta t<1$.
# BDF5 and BDF6 are stable on the negative axis but not monotone: at large $\lambda\Delta t$ their
# first steps after the jump overshoot. Crank-Nicolson is A-stable but not L-stable: the stiffest modes of the initial jump at the
# faces are multiplied by almost $-1$ in every step, and on the high-order and Nitsche elements the
# resulting overshoot leaves $[0,B_0]$. Each of these must fail; its stabilized counterpart,
# shown behind a grey hatching, is the same method at a step it is stable or damping for, or
# leapfrog with the Robert-Asselin-Williams filter. A counterpart runs only where the declared
# scheme actually failed.

stabilized_time = {
    "leapfrog": (
        "leapfrog + Robert-Asselin-Williams filter (ν = 0.2, α = 0.53), safety factor 0.1",
        TimeMethod("leapfrog_raw"),
        0.1,
    ),
}
stable_step = {
    "Adams-Moulton 3": 1e-6,
    "Adams-Moulton 4": 5e-7,
    "Crank-Nicolson": 4e-7,
    "BDF5 (Gear)": 4e-7,
    "BDF6 (Gear)": 4e-7,
}
linear_solvers = {
    "sparse LU": SparseDirect(),
    "conjugate gradients": ConjugateGradient(),
    "CG + algebraic multigrid": ConjugateGradient(preconditioner=AlgebraicMultigrid()),
    "GMRES": GMRES(),
    "GMRES + algebraic multigrid": GMRES(preconditioner=AlgebraicMultigrid()),
    "GMRES + ILU": GMRES(preconditioner=ILU()),
    "right GMRES": RightGMRES(),
    "flexible GMRES": FlexibleGMRES(),
    "automatic": AutomaticLinearSolver(),
}
nonlinear_solvers = {
    **{
        (name, linear_name): kind(linear_solver=linear)
        for name, kind in (
            ("Newton", Newton),
            ("damped Newton", DampedNewton),
            ("trust-region Newton", TrustRegionNewton),
            ("quasi-Newton", QuasiNewton),
            ("JFNK", JFNK),
        )
        for linear_name, linear in linear_solvers.items()
    },
    ("fixed point", "none"): FixedPoint(maximum_iterations=200),
    ("fixed point + Aitken", "none"): FixedPoint(maximum_iterations=200, acceleration="aitken"),
}
stable_step.update({"fixed point": 2e-7, "fixed point + Aitken": 2e-7})

# ## The test matrix
#
# Three cuts through the space of combinations:
#
# 1. **Space × time integrator** — every spatial scheme on 200 cells with four implicit
#    integrators (backward Euler, Crank-Nicolson, BDF2, SDIRK2) at $\Delta t=5\,$µs.
# 2. **Time integrator × space** — every integrator on three spatial schemes (central differences,
#    two-point volumes and CG P1 with Nitsche faces). An explicit method on a diffusion operator
#    must take $\Delta t\propto h^2$, so this cut uses 50 cells; the spatial accuracy is the subject of the
#    first cut. Higher-degree and discontinuous elements are left out of this cut: degree and
#    interior penalty raise $\lambda_{\max}$, and an explicit step would have to be five to twenty
#    times smaller; they run with the four implicit integrators of the first cut.
# 3. **Nonlinear × linear solver** — backward Euler on central differences with 50 cells (the
#    fixed-point counterpart needs $\lambda_{\max}\Delta t<1$, which is affordable on this grid).
#
# The representative schemes are repeated on 100 cells for the mesh check.

space_integrators = {
    name: integrators[name]
    for name in ("backward Euler", "Crank-Nicolson", "BDF2 (Gear)", "SDIRK2")
}
time_spatial = {
    name: spatial[name]
    for name in (
        "FDM central, 2nd order",
        "FVM two-point flux",
        "FEM CG P1 + symmetric Nitsche",
    )
}
runs = [
    dict(
        matrix="space x time integrator",
        row=row,
        column=column,
        cells=cells,
        spatial=scheme,
        time=integrator,
        step=implicit_step,
    )
    for row, scheme in spatial.items()
    for column, integrator in space_integrators.items()
]
runs += [
    dict(
        matrix="time integrator x space",
        row=name,
        column=space,
        cells=time_cells,
        spatial=scheme,
        time=integrator,
        step=implicit_step if getattr(integrator, "implicit", False) else end_time,
    )
    for name, integrator in integrators.items()
    for space, scheme in time_spatial.items()
]
runs += [
    dict(
        matrix="nonlinear x linear solver",
        row=row,
        column=column,
        cells=time_cells,
        spatial=spatial["FDM central, 2nd order"],
        time=BackwardEuler(),
        nonlinear=solver,
        step=implicit_step,
    )
    for (row, column), solver in nonlinear_solvers.items()
]
representative = [
    ("FDM central, 2nd order", "BDF2 (Gear)"),
    ("FDM polynomial, 4th order", "BDF2 (Gear)"),
    ("FVM two-point flux", "BDF2 (Gear)"),
    ("FEM CG P1 + symmetric Nitsche", "BDF2 (Gear)"),
    ("FEM CG P2 + symmetric Nitsche", "BDF2 (Gear)"),
    ("FEM DG P1 SIPG", "BDF2 (Gear)"),
    ("FEM DG P2 SIPG", "BDF2 (Gear)"),
]
runs += [
    {**run, "matrix": "mesh check", "cells": coarse_cells}
    for run in runs
    if run["matrix"] == "space x time integrator"
    and (run["row"], run["column"]) in representative
]
print(
    f"{len(runs)} runs: "
    + ", ".join(
        f"{sum(r['matrix'] == m for r in runs)} {m}"
        for m in dict.fromkeys(r["matrix"] for r in runs)
    )
)

# ## Run everything
#
# One loop, the same problem object every time. A combination the library refuses is recorded with
# its validation message. A run whose induction leaves the interval $[0, B_0]$ violates the
# maximum principle of the diffusion equation and is recorded as failed, as is a solver that does not
# converge. A failed run with a stabilized counterpart appends that counterpart to the list, so the
# same loop runs it next.

for run in runs:
    implicit = getattr(run["time"], "implicit", False)
    rosenbrock = "rosenbrock" in getattr(run["time"], "method_id", "")
    simulation = Simulation(
        problem=problem,
        mesh=meshes[run["cells"]],
        numerics=Numerics(
            spatial=run["spatial"],
            time_integrator=run["time"],
            linear_solver=(
                SparseDirect()
                if rosenbrock or (not implicit and isinstance(run["spatial"], FiniteElement))
                else None
            ),
            nonlinear_solver=(
                run.get("nonlinear", newton) if implicit and not rosenbrock else None
            ),
            step_limits=TimeStepLimits(stability_safety_factor=run.get("safety", 0.5)),
        ),
    )
    started = time.perf_counter()
    check = simulation.validate()
    run.update(status="rejected", note="", error=np.nan, peak=np.nan, steps=0)
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = simulation.run(start_time=0.0, end_time=end_time, time_step=run["step"])
            run["profile"] = (
                result.field(copper.magnetic_induction).evaluate(centres[run["cells"]]).ravel()
            )
            run["error"] = np.abs(run["profile"] - exact[run["cells"]]).mean()
            run["peak"] = np.abs(run["profile"] - exact[run["cells"]]).max()
            run["steps"] = len(result.times) - 1
            inside = np.isfinite(run["profile"]).all() and (
                -0.01 * field <= run["profile"].min() and run["profile"].max() <= 1.01 * field
            )
            run["status"] = "ran" if inside else "failed"
            run["note"] = "" if inside else "induction leaves [0, B0] (unstable)"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "failed"
                if "unstable" in message
                or "finite" in message.replace("definite", "")
                or "converge" in message
                or "stagnat" in message
                else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    if run["status"] == "failed" and run.get("variant") is None and run["matrix"] != "mesh check":
        if run["row"] in stabilized_time:
            runs.append(
                {
                    **run,
                    "variant": "stabilized",
                    "time": stabilized_time[run["row"]][1],
                    "safety": stabilized_time[run["row"]][2],
                    "note_stabilized": stabilized_time[run["row"]][0],
                }
            )
        elif run["row"] in stable_step or run["column"] in stable_step:
            runs.append(
                {
                    **run,
                    "variant": "stabilized",
                    "step": stable_step.get(run["row"], stable_step.get(run["column"])),
                    "note_stabilized": f"Δt = {stable_step.get(run['row'], stable_step.get(run['column'])):.0e} s",
                }
            )
    print(
        f"{run['matrix'][:12]:12s} {run['row'][:34]:34s} {run['column'][:24]:24s} "
        f"{run.get('variant') or '':10s} {run['status']:8s} {run['seconds']:7.1f} s  "
        + (
            f"mean |B-B_exact| {run['error']:.2e} T  steps {run['steps']}"
            if run["status"] == "ran"
            else run["note"][:90]
        )
    )

# ## Space × time integrator
#
# The mean induction error of every spatial scheme with four implicit integrators. Backward Euler
# is first order in time; at $\Delta t=5\,$µs its time error dominates every spatial error, while
# Crank-Nicolson, BDF2 and SDIRK2 reveal the spatial ranking.

main = {
    (r["matrix"], r["row"], r["column"]): r
    for r in runs
    if r.get("variant") is None and r["matrix"] != "mesh check"
}
substitute = {
    (r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") == "stabilized"
}
hatch = {
    key
    for key, r in main.items()
    if r["status"] != "ran" and substitute.get(key, {}).get("status") == "ran"
}
cell = {key: substitute[key] if key in hatch else r for key, r in main.items()}

space_key = "space x time integrator"
rows, columns = list(spatial), list(space_integrators)
fig = plots.figure(1, 1, size=(12, 10))
plots.matrix(
    [
        [cell[(space_key, row, c)]["error"] for c in columns]
        for row in rows
    ],
    rows,
    columns,
    scale="log",
    label="mean |B - B_exact| at t = 0.4 ms [T]",
    text=[
        [
            (
                f"{cell[(space_key, row, c)]['error']:.2e}"
                if cell[(space_key, row, c)]["status"] == "ran"
                else cell[(space_key, row, c)]["status"]
            )
            for c in columns
        ]
        for row in rows
    ],
    hatched=[[(space_key, row, c) in hatch for c in columns] for row in rows],
    notes=(
        [
            "Hatched: Crank-Nicolson does not damp the stiff modes of the initial jump and "
            f"overshoots on these elements; colour and value show it at Δt = {stable_step['Crank-Nicolson']:.0e} s."
        ]
        if any((space_key, row, "Crank-Nicolson") in hatch for row in rows)
        else None
    ),
    title=f"Space x time integrator, {cells} cells, Δt = {implicit_step} s",
    ax=fig[0],
)
plots.show()

# All measures for the combinations that ran, sorted by the mean error.

matrix = [r for r in runs if r["matrix"] == space_key]
print(f"{'scheme':40s} {'integrator':16s} {'mean err':>9s} {'max err':>9s} {'time':>8s}")
for r in sorted((r for r in matrix if r["status"] == "ran"), key=lambda r: r["error"]):
    print(
        f"{r['row']:40s} {r['column']:16s} {r['error']:9.2e} {r['peak']:9.2e} {r['seconds']:7.1f}s"
    )

# ## Profiles of representative combinations
#
# Induction through the plate against the series, and a close-up of one face where the initial
# jump between $0$ and $B_0$ is smoothed out.

shown = [
    r
    for r in matrix
    if (r["row"], r["column"]) in representative and r["status"] == "ran"
]
fig = plots.figure(1, 2, size=(15, 5))
for panel, window, title in (
    (fig[0], None, f"Induction at t = {1e3 * end_time} ms"),
    (fig[1], (0.0, 1.5), "Close-up of a face"),
):
    plots.line(
        y={
            f"{r['row']}, {r['column']}": (1e3 * centres[cells][:, 0], r["profile"])
            for r in shown
        },
        reference={"Fourier series": (1e3 * fine, exact["fine"])},
        xlabel="position x [mm]",
        ylabel="induction B_z [T]",
        title=title,
        xlimits=window,
        ylimits=None if window is None else (0.04, 0.1),
        legend="upper right" if window is None else False,
        ax=panel,
    )
plots.show()

# ## Time integrator × space
#
# Explicit integrators at their stable step, implicit integrators at $\Delta t=5\,$µs, on 50
# cells.

time_key = "time integrator x space"
time_rows, time_columns = list(integrators), list(time_spatial)
time_hatched = [[(time_key, row, c) in hatch for c in time_columns] for row in time_rows]
time_notes = (
    [
        "Hatched: the declared integrator is unstable at this step (not A-stable, or undamped); "
        "colour and value show its stabilized counterpart:"
    ]
    + [
        f"  {row}: same method at Δt = {stable_step[row]:.0e} s instead of {implicit_step:.0e} s."
        for row in stable_step
        if any((time_key, row, c) in hatch for c in time_columns)
    ]
    + [
        f"  {row}: {description}."
        for row, (description, _, _) in stabilized_time.items()
        if any((time_key, row, c) in hatch for c in time_columns)
    ]
)
fig = plots.figure(1, 2, size=(17, 12))
for panel, measure, label, title in (
    (fig[0], "error", "mean |B - B_exact| [T]", "Mean induction error"),
    (fig[1], "seconds", "wall time [s]", "Wall time and accepted steps"),
):
    plots.matrix(
        [
            [
                cell[(time_key, row, c)][measure]
                if cell[(time_key, row, c)]["status"] == "ran"
                else np.nan
                for c in time_columns
            ]
            for row in time_rows
        ],
        time_rows,
        time_columns,
        scale="log",
        label=label,
        text=[
            [
                (
                    (
                        f"{cell[(time_key, row, c)]['error']:.2e}"
                        if measure == "error"
                        else f"{cell[(time_key, row, c)]['seconds']:.1f} s, "
                        f"{cell[(time_key, row, c)]['steps']} steps"
                    )
                    if cell[(time_key, row, c)]["status"] == "ran"
                    else cell[(time_key, row, c)]["status"]
                )
                for c in time_columns
            ]
            for row in time_rows
        ],
        hatched=time_hatched,
        notes=time_notes if measure == "error" else None,
        title=title,
        ax=panel,
    )
plots.show()

# ## Nonlinear × linear solver
#
# Backward Euler on central differences with 50 cells. The induction equation is linear, so every
# converging Newton variant reproduces the same discrete solution in one iteration per step; the
# wall time differs. The step matrix $I+\Delta t\,K$ of the symmetric central stencil is
# symmetric positive definite, but this owner does not declare it so: conjugate gradients and the
# symmetric algebraic multigrid are refused. That is a gap of the library, shown as "rejected".

solver_key = "nonlinear x linear solver"
solving = [r for r in runs if r["matrix"] == solver_key and r.get("variant") is None]
solver_rows = list(dict.fromkeys(r["row"] for r in solving))
solver_columns = [*linear_solvers, "none"]
fig = plots.figure(1, 1, size=(15, 6))
plots.matrix(
    [
        [
            (
                cell[(solver_key, row, c)]["seconds"]
                if (solver_key, row, c) in cell and cell[(solver_key, row, c)]["status"] == "ran"
                else np.nan
            )
            for c in solver_columns
        ]
        for row in solver_rows
    ],
    solver_rows,
    solver_columns,
    scale="log",
    label="wall time [s]",
    text=[
        [
            (
                (
                    f"{cell[(solver_key, row, c)]['seconds']:.1f} s\n"
                    f"{cell[(solver_key, row, c)]['error']:.2e}"
                )
                if cell[(solver_key, row, c)]["status"] == "ran"
                else cell[(solver_key, row, c)]["status"]
            )
            if (solver_key, row, c) in cell
            else ""
            for c in solver_columns
        ]
        for row in solver_rows
    ],
    hatched=[[(solver_key, row, c) in hatch for c in solver_columns] for row in solver_rows],
    notes=[
        "Hatched: the fixed-point (Picard) iteration cannot contract at Δt = 5 µs; colour and "
        "value show the same iteration at the step it converges for:"
    ]
    + [
        f"  {row}: Δt = {stable_step[row]:.0e} s."
        for row in ("fixed point", "fixed point + Aitken")
        if any((solver_key, row, c) in hatch for c in solver_columns)
    ],
    title="Backward Euler: wall time and mean induction error per solver pair",
    ax=fig[0],
)
plots.show()

# ## Mesh check: 200 against 100 cells
#
# BDF2 at $\Delta t=5\,$µs on half as many cells. Second-order schemes should reduce
# the spatial error about fourfold; once it falls below the time error of the step, the ratio
# levels off.

coarse = {(r["row"], r["column"]): r for r in runs if r["matrix"] == "mesh check"}
grid = {(r["row"], r["column"]): r for r in matrix}
print(f"{'scheme':40s} {'err 100':>10s} {'err 200':>10s} {'ratio':>6s}")
for key, r in coarse.items():
    if r["status"] == "ran" and grid[key]["status"] == "ran":
        print(
            f"{key[0]:40s} {r['error']:10.3e} {grid[key]['error']:10.3e} "
            f"{r['error'] / grid[key]['error']:6.2f}"
        )

# ## What the library refused, and what failed
#
# A refused combination is caught by validation or binding before any time step; a failed one
# started and left the physical range or did not converge.

for status in ("rejected", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(
            f"  {r['matrix'][:12]:12s} {r['row'][:34]:34s} {r['column'][:24]:24s} "
            f"{r.get('variant') or '':10s} {r['note'][:90]}"
        )
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.0f} min in total"
)

# ## Methods that do not apply to this problem
#
# Riemann solvers, reconstructions and slope limiters act on convective fluxes; magnetic diffusion
# in a plate at rest has none, so finite volumes take only a diffusive face flux. SUPG, GLS and
# graph viscosity stabilize convection and have nothing to act on here. Staggered Verlet needs a
# second-order system and IMEX splits a stiff from a non-stiff part, which a single diffusion
# operator does not have. Continuous elements on this balance take their face values through
# Nitsche terms; strongly imposed values are not available for the line balance owner.
#
# ## Going further
#
# Replace a `PrescribedValueBoundary` by an `OutwardFluxBoundary` (a prescribed tangential electric
# field) and the matrix code stays as it is; add a column by adding one object to a dictionary.
