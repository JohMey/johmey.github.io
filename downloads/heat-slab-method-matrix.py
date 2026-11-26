"""OneSimulation example: transient heat conduction in a slab through every method combination.
Requires the installed, activated OneSimulation runtime.

One physics declaration, Fourier conduction in a steel slab whose faces are held at two
temperatures, is solved with every applicable combination of discretization family (finite
differences, finite volumes, continuous, discontinuous and Nitsche finite elements), time
integrator, nonlinear solver and linear solver. Each run is graded against the exact Fourier
series; combinations the library refuses are reported as such.
"""

# # Heat conduction in a slab: one model, every method
#
# A steel slab of thickness $L=20\,$mm starts at $T_0=300\,$K. At $t=0$ its left face is raised to
# $T_1=400\,$K and held there; the right face stays at $T_0$. Fourier's law gives
#
# $$\rho c_p\,\partial_t T=\partial_x\big(k\,\partial_x T\big),\qquad T(0,t)=T_1,\quad T(L,t)=T_0,
# \quad T(x,0)=T_0,$$
#
# with $k=45\,$W/(m K), $\rho=7850\,$kg/m³ and $c_p=460\,$J/(kg K), so the diffusivity is
# $\alpha=k/(\rho c_p)\approx1.25\cdot10^{-5}\,$m²/s. The heat equation is declared once.
# Everything numerical is an argument of `Numerics`: the discretization family, its diffusion
# operator and boundary treatment, the time integrator and, for implicit steps, the nonlinear and
# the linear solver.
#
# **Reference.** Separating the steady line $T_s(x)=T_1+(T_0-T_1)\,x/L$ leaves a sine series
# (Carslaw and Jaeger, *Conduction of Heat in Solids*, §3.3):
#
# $$T(x,t)=T_s(x)+\sum_{n\ge1}\frac{2(T_0-T_1)}{n\pi}\,\sin\frac{n\pi x}{L}\,
# \exp\!\Big(-\frac{n^2\pi^2\alpha t}{L^2}\Big).$$
#
# **Grading.** At $t=3\,$s (Fourier number $\alpha t/L^2\approx0.093$) every run is evaluated at
# the cell centres $x_i$: the mean absolute error $\sum_i|T_h(x_i)-T(x_i)|\,\Delta x/L$ and the largest
# deviation; the wall time covers validation, binding and time
# stepping.

import time

import numpy as np
from onesim.cad.d1 import Line
from onesim.materials import IsotropicSolid
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
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# Nothing in this cell names a numerical method.

thickness, hot, cold, end_time = 0.02, 400.0, 300.0, 3.0
cells, coarse_cells, time_cells = 200, 100, 50
steel = IsotropicSolid(
    density=7850.0,
    specific_heat_capacity=460.0,
    thermal_conductivity=45.0,
    youngs_modulus=2.1e11,
    poisson_ratio=0.3,
    thermal_expansion_coefficient=1.2e-5,
)
diffusivity = steel.thermal_conductivity / (steel.density * steel.specific_heat_capacity)
slab = Line(start=(0.0, 0.0, 0.0), end=(thickness, 0.0, 0.0))
heat = HeatConduction(domain=slab.line, material=steel)
problem = Problem(
    name="Heated slab",
    geometry=slab,
    physics=(heat,),
    boundary_conditions=(
        FixedTemperature(physics=heat, boundary=slab.boundaries.start, value=hot),
        FixedTemperature(physics=heat, boundary=slab.boundaries.end, value=cold),
    ),
    initial_conditions=(InitialValue(heat.temperature, cold),),
)
meshes = {
    n: StructuredLineMesher(cells=n).generate(slab) for n in (cells, coarse_cells, time_cells)
}
centres = {n: mesh.cell_centers for n, mesh in meshes.items()}
print(
    f"{meshes[cells].cell_count} cells of width {1e3 * thickness / cells:.2f} mm "
    f"(mesh check: {meshes[coarse_cells].cell_count} cells; time-integrator cut: "
    f"{meshes[time_cells].cell_count} cells), diffusivity {diffusivity:.4e} m²/s, "
    f"Fourier number {diffusivity * end_time / thickness**2:.4f}"
)

# ## The exact solution
#
# The series is summed with 2000 terms at every cell centre and on a fine plotting grid.

modes = np.arange(1, 2001)[:, None]
fine = np.linspace(0.0, thickness, 801)
exact = {
    key: hot
    + (cold - hot) * points / thickness
    + (
        2 * (cold - hot) / (np.pi * modes)
        * np.sin(modes * np.pi * points / thickness)
        * np.exp(-((modes * np.pi / thickness) ** 2) * diffusivity * end_time)
    ).sum(axis=0)
    for key, points in {**{n: c[:, 0] for n, c in centres.items()}, "fine": fine}.items()
}
print(
    f"mid-plane temperature {np.interp(thickness / 2, fine, exact['fine']):.3f} K at t = {end_time} s"
)

# ## The method catalogue
#
# Finite differences with central, fourth- and sixth-order polynomial Laplacians; finite volumes
# with the two-point flux and with the corrected central flux of the native cell balance;
# continuous Lagrange elements of degree 1 to 3 with strong boundary values, symmetric and
# nonsymmetric Nitsche boundary terms and continuous interior penalty; discontinuous elements of
# degree 1 to 3 with the symmetric interior-penalty (SIPG) flux at two penalty levels. The
# finite-difference heat flux needs a first derivative, so each stencil is paired with the
# gradient of the same order.

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

# Time integrators: the explicit one-step and multistep methods, implicit Runge-Kutta,
# Adams-Moulton and BDF methods, Rosenbrock methods and the adaptive integrators. Implicit stages
# take a Newton solver with a sparse direct solve; the linearly implicit Rosenbrock stages take the
# sparse direct solve alone. Explicit methods advance at their stable step (safety factor 0.5),
# implicit ones at $\Delta t=0.05\,$s.

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
    "adaptive RK45": AdaptiveRungeKutta45(relative_tolerance=1e-5, absolute_tolerance=1e-6),
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
    "adaptive BDF": AdaptiveBDF(relative_tolerance=1e-4, absolute_tolerance=1e-3),
}
implicit_step = 0.05
newton = Newton(linear_solver=SparseDirect())

# ## Schemes that must fail, and their stabilized counterparts
#
# The diffusion operator has a real, negative spectrum reaching $-\lambda_{\max}\sim-12\alpha/h^2$.
# Adams-Moulton 3 and 4 are not A-stable: their stability interval on the negative axis ends at
# $\lambda\Delta t\approx6$ and $3$, while $\lambda_{\max}\Delta t$ is several hundred at
# $\Delta t=0.05\,$s. Leapfrog has no damping: every decaying mode is amplified at any step. A
# fixed-point (Picard) iteration of an implicit step contracts only for $\lambda_{\max}\Delta t<1$.
# BDF5 and BDF6 are stable on the negative axis but not monotone: at large $\lambda\Delta t$ their
# first steps after the jump overshoot. Crank-Nicolson is A-stable but not L-stable: the stiffest modes of the initial jump at the hot
# face are multiplied by almost $-1$ in every step, and on the high-order and Nitsche elements the
# resulting overshoot leaves $[T_0,T_1]$. Each of these must fail; its stabilized counterpart,
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
    "Adams-Moulton 3": 2e-3,
    "Adams-Moulton 4": 1e-3,
    "Crank-Nicolson": 2e-3,
    "BDF5 (Gear)": 2e-3,
    "BDF6 (Gear)": 2e-3,
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
stable_step.update({"fixed point": 1e-3, "fixed point + Aitken": 1e-3})

# ## The test matrix
#
# Three cuts through the space of combinations:
#
# 1. **Space × time integrator** — every spatial scheme on 200 cells with four implicit
#    integrators (backward Euler, Crank-Nicolson, BDF2, SDIRK2) at $\Delta t=0.05\,$s.
# 2. **Time integrator × space** — every integrator on three spatial schemes (central differences,
#    two-point volumes and CG P1). An explicit method on a diffusion operator must take
#    $\Delta t\propto h^2$, so this cut uses 50 cells; the spatial accuracy is the subject of the
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
    for name in ("FDM central, 2nd order", "FVM two-point flux", "FEM CG P1")
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
    ("FEM CG P1", "BDF2 (Gear)"),
    ("FEM CG P2", "BDF2 (Gear)"),
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
# its validation message. A run whose temperature leaves the interval $[T_0, T_1]$ violates the
# maximum principle of the heat equation and is recorded as failed, as is a solver that does not
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
            run["profile"] = result.field(heat.temperature).evaluate(centres[run["cells"]]).ravel()
            run["error"] = np.abs(run["profile"] - exact[run["cells"]]).mean()
            run["peak"] = np.abs(run["profile"] - exact[run["cells"]]).max()
            run["steps"] = len(result.times) - 1
            inside = np.isfinite(run["profile"]).all() and (
                cold - 1.0 <= run["profile"].min() and run["profile"].max() <= hot + 1.0
            )
            run["status"] = "ran" if inside else "failed"
            run["note"] = "" if inside else "temperature leaves [T0, T1] (unstable)"
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
            f"mean |T-T_exact| {run['error']:.2e} K  steps {run['steps']}"
            if run["status"] == "ran"
            else run["note"][:90]
        )
    )

# ## Space × time integrator
#
# The mean temperature error of every spatial scheme with four implicit integrators. Backward Euler
# is first order in time; at $\Delta t=0.05\,$s its time error dominates every spatial error, while
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
    label="mean |T - T_exact| at t = 3 s [K]",
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
# Temperature through the slab against the series, and a close-up of the hot face where the
# initial jump between $T_0$ and $T_1$ is smoothed out.

shown = [
    r
    for r in matrix
    if (r["row"], r["column"]) in representative and r["status"] == "ran"
]
fig = plots.figure(1, 2, size=(15, 5))
for panel, window, title in (
    (fig[0], None, f"Temperature at t = {end_time} s"),
    (fig[1], (0.0, 3.0), "Close-up of the hot face"),
):
    plots.line(
        y={
            f"{r['row']}, {r['column']}": (1e3 * centres[cells][:, 0], r["profile"])
            for r in shown
        },
        reference={"Fourier series": (1e3 * fine, exact["fine"])},
        xlabel="position x [mm]",
        ylabel="temperature [K]",
        title=title,
        xlimits=window,
        ylimits=None if window is None else (340.0, 400.0),
        legend="upper right" if window is None else False,
        ax=panel,
    )
plots.show()

# ## Time integrator × space
#
# Explicit integrators at their stable step, implicit integrators at $\Delta t=0.05\,$s, on 50
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
        f"  {row}: same method at Δt = {stable_step[row]:.0e} s instead of {implicit_step} s."
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
    (fig[0], "error", "mean |T - T_exact| [K]", "Mean temperature error"),
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
# Backward Euler on central differences with 50 cells. The heat equation is linear, so every
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
        "Hatched: the fixed-point (Picard) iteration cannot contract at Δt = 0.05 s; colour and "
        "value show the same iteration at the step it converges for:"
    ]
    + [
        f"  {row}: Δt = {stable_step[row]:.0e} s."
        for row in ("fixed point", "fixed point + Aitken")
        if any((solver_key, row, c) in hatch for c in solver_columns)
    ],
    title="Backward Euler: wall time and mean temperature error per solver pair",
    ax=fig[0],
)
plots.show()

# ## Mesh check: 200 against 100 cells
#
# BDF2 at $\Delta t=0.05\,$s on half as many cells. Second-order schemes should reduce
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
# Riemann solvers, reconstructions and slope limiters act on convective fluxes; pure conduction has
# none, so finite volumes take only a diffusive face flux. SUPG, GLS and graph viscosity stabilize
# convection and have nothing to act on here. Staggered Verlet needs a second-order system and IMEX
# splits a stiff from a non-stiff part, which a single diffusion operator does not have.
#
# ## Going further
#
# Replace a `FixedTemperature` by `HeatFlux` or `ConvectiveHeatFlux` and the matrix code stays as
# it is; add a column by adding one object to a dictionary.
