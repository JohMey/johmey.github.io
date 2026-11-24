"""OneSimulation example: Sod's shock tube through every method combination OneSimulation offers.
Requires the installed, activated OneSimulation runtime.

One physics declaration, the Euler equations with Sod's initial state, is solved on 400 cells with
every applicable combination of discretization family, reconstruction, limiter, Riemann solver,
time integrator, nonlinear solver and linear solver. Each run is graded against the exact Riemann
solution; combinations the library refuses are reported as such.
"""

# # Sod's shock tube: one model, every method
#
# A tube of ideal gas ($\gamma=1.4$) is divided by a diaphragm at $x=0.5$. Both halves are at rest:
#
# $$(\rho,u,p)_L=(1,\,0,\,1),\qquad (\rho,u,p)_R=(0.125,\,0,\,0.1).$$
#
# When the diaphragm is removed, a rarefaction fan runs left, a contact discontinuity and a shock
# run right. The one-dimensional Euler equations
#
# $$\partial_t\begin{pmatrix}\rho\\ \rho u\\ E\end{pmatrix}
# +\partial_x\begin{pmatrix}\rho u\\ \rho u^2+p\\ u(E+p)\end{pmatrix}=0,
# \qquad p=(\gamma-1)\Big(E-\tfrac12\rho u^2\Big)$$
#
# are declared once. Everything numerical is an argument of `Numerics`: the discretization family
# (finite volumes, finite differences, discontinuous and continuous finite elements), the
# reconstruction and its limiter, the Riemann solver, the time integrator and, for implicit steps,
# the nonlinear and the linear solver.
#
# **Reference.** The exact solution of the Riemann problem (Toro, *Riemann Solvers and Numerical
# Methods for Fluid Dynamics*, ch. 4) comes from OneSimulation's exact Euler sampler: the star
# pressure matches an ideal-gas Hugoniot curve on the shock side and an isentrope on the
# rarefaction side, and the self-similar solution is sampled at $\xi=(x-0.5)/t$.
#
# **Grading.** At $t=0.2$ every run is evaluated at the 400 cell centres $x_i$:
# $L_1(q)=\sum_i |q_h(x_i)-q(x_i)|\,\Delta x$ for density, velocity and pressure; the shock position
# is where the density crosses the middle of its jump; the wall time covers validation, binding and
# time stepping.

import time

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import (
    BoundaryFluxDifference,
    CentralDifference,
    FiniteDifference,
    GlobalLaxFriedrichsSplit,
    PolynomialDifference,
    WENO5Flux,
    WENOFluxDifference,
)
from onesim.numerics.fem import (
    DGPolynomialLimiter,
    DiscontinuousLagrange,
    FiniteElement,
    InvariantDomainGraphViscosity,
    Lagrange,
    RowSumLumping,
)
from onesim.numerics.fluxes import HLL, HLLC, HLLE, Rusanov
from onesim.numerics.fvm import (
    MUSCL,
    QUICK,
    BarthJespersenLimiter,
    ConvexInvariantDomain,
    Eno2Reconstructor,
    Eno3Reconstructor,
    FiniteVolume,
    MinmodLimiter,
    MonotonizedCentralLimiter,
    SuperbeeLimiter,
    TvbLimiter,
    VanLeerLimiter,
    VenkatakrishnanLimiter,
    Weno3Reconstructor,
    Weno5Reconstructor,
)
from onesim.numerics.linear import GMRES, ILU, FlexibleGMRES, RightGMRES, SparseDirect
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
from onesim.physics.hydrodynamics import (
    CompressibleEuler,
    EulerInvariantDomain,
    EulerPressureWave,
    IdealGasEquationOfState,
    NormalEulerPrimitive,
    TammannHugoniot,
    TammannRarefaction,
    TransmissiveBoundary,
    exact_euler_pressure_trace,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# Nothing in this cell names a numerical method.

gamma, end_time, cells, coarse_cells = 1.4, 0.2, 400, 200
tube = Line(start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0))
gas = CompressibleEuler(
    domain=tube.line, equation_of_state=IdealGasEquationOfState(gamma=gamma)
)
sod = Problem(
    name="Sod shock tube",
    geometry=tube,
    physics=(gas,),
    boundary_conditions=(
        TransmissiveBoundary(boundary=tube.boundaries.all, physics=gas),
    ),
    initial_conditions=(
        InitialValue(gas.density, lambda x: np.where(x < 0.5, 1.0, 0.125)),
        InitialValue(gas.momentum, 0.0),
        InitialValue(gas.total_energy, lambda x: np.where(x < 0.5, 2.5, 0.25)),
    ),
)
meshes = {n: StructuredLineMesher(cells=n).generate(tube) for n in (cells, coarse_cells)}
centres = {n: mesh.cell_centers[:, 0] for n, mesh in meshes.items()}
print(
    f"{meshes[cells].cell_count} cells of width {1 / cells:.5f} "
    f"(coarse check: {meshes[coarse_cells].cell_count} cells)"
)

# ## The exact solution
#
# The two initial states with their ideal-gas shock and rarefaction curves; the exact sampler returns
# the state on every ray $\xi$ and the speeds of the outermost waves.

waves = [
    EulerPressureWave(
        NormalEulerPrimitive(*state), TammannHugoniot(gamma), TammannRarefaction(gamma)
    )
    for state in ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1))
]
fine = np.linspace(0.0, 1.0, 2001)
exact = {
    key: np.array(
        [
            [
                getattr(
                    exact_euler_pressure_trace(
                        *waves, similarity=(x - 0.5) / end_time
                    ).state,
                    name,
                )
                for x in points
            ]
            for name in ("density", "velocity", "pressure")
        ]
    )
    for key, points in {**centres, "fine": fine}.items()
}
fan_head, shock_speed = exact_euler_pressure_trace(*waves).signal_bounds
exact_shock = 0.5 + shock_speed * end_time
shock_level = 0.5 * (
    exact_euler_pressure_trace(*waves, similarity=shock_speed - 1e-9).state.density
    + 0.125
)
print(
    f"star pressure {exact['fine'][2][1000]:.5f}, contact velocity {exact['fine'][1][1000]:.5f}, "
    f"shock speed {shock_speed:.5f}, shock at x = {exact_shock:.5f} for t = {end_time}"
)

# ## The method catalogue
#
# Four Riemann solvers, seven slope limiters plus the unlimited slope, five higher-order
# reconstructions, discontinuous elements of degree 0 to 3 with and without limiting, continuous
# elements with and without graph viscosity, and finite-difference stencils. Each is an object; the
# matrix is built from them with plain dictionaries.
#
# Limited discontinuous Galerkin is the expensive part of the matrix (about five minutes per P1 run
# on 400 cells, longer for P2 and P3). It runs with all four Riemann solvers for P1 and with HLLC for
# P2 and P3; the limiter is minmod on the slopes plus the Euler invariant domain (positive density
# and pressure at every node). A plain minmod limiter gives the same profiles to three digits on
# this problem.

riemann = {"Rusanov": Rusanov(), "HLL": HLL(), "HLLE": HLLE(), "HLLC": HLLC()}
limiters = {
    "minmod": MinmodLimiter(),
    "MC": MonotonizedCentralLimiter(),
    "van Leer": VanLeerLimiter(),
    "superbee": SuperbeeLimiter(),
    "Barth-Jespersen": BarthJespersenLimiter(),
    "Venkatakrishnan": VenkatakrishnanLimiter(),
    "TVB (M = 50)": TvbLimiter(
        base_limiter=MinmodLimiter(), tvb_constant=50.0, mesh_width=1 / cells
    ),
}
reconstructions = {
    "ENO2": Eno2Reconstructor(),
    "ENO3": Eno3Reconstructor(),
    "WENO3": Weno3Reconstructor(),
    "WENO5": Weno5Reconstructor(),
    "QUICK": QUICK(),
}
dg_limiter = DGPolynomialLimiter(
    slope_limiter=MinmodLimiter(), invariant_domain=EulerInvariantDomain()
)
graph_viscosity = InvariantDomainGraphViscosity(
    mass=RowSumLumping(), invariant_domain=EulerInvariantDomain()
)
spatial = {}
for flux_name, flux in riemann.items():
    spatial[("FVM first order", flux_name)] = FiniteVolume(convection=flux)
    spatial[("FVM MUSCL, unlimited", flux_name)] = FiniteVolume(
        convection=flux, reconstruction=MUSCL()
    )
    for limiter_name, limiter in limiters.items():
        spatial[(f"FVM MUSCL + {limiter_name}", flux_name)] = FiniteVolume(
            convection=flux, reconstruction=MUSCL(), limiter=limiter
        )
    spatial[("FVM MUSCL + minmod + positivity", flux_name)] = FiniteVolume(
        convection=flux,
        reconstruction=MUSCL(),
        limiter=MinmodLimiter(),
        positivity=ConvexInvariantDomain(),
    )
    for reconstruction_name, reconstruction in reconstructions.items():
        spatial[(f"FVM {reconstruction_name}", flux_name)] = FiniteVolume(
            convection=flux, reconstruction=reconstruction
        )
    spatial[("DG P0", flux_name)] = FiniteElement(
        element=DiscontinuousLagrange(0), convection=flux
    )
    for degree in (1, 2, 3):
        spatial[(f"DG P{degree}, unlimited", flux_name)] = FiniteElement(
            element=DiscontinuousLagrange(degree), convection=flux
        )
        if degree == 1 or flux_name == "HLLC":
            spatial[(f"DG P{degree} + minmod + invariant domain", flux_name)] = (
                FiniteElement(
                    element=DiscontinuousLagrange(degree),
                    convection=flux,
                    limiter=dg_limiter,
                )
            )
    for degree in (1, 2):
        spatial[(f"CG P{degree} + graph viscosity", flux_name)] = FiniteElement(
            element=Lagrange(degree), convection=flux, stabilization=graph_viscosity
        )
        spatial[(f"CG P{degree} Galerkin", flux_name)] = FiniteElement(
            element=Lagrange(degree), convection=flux
        )
    spatial[("FDM central, 2nd order", flux_name)] = FiniteDifference(
        gradient=BoundaryFluxDifference(derivative=CentralDifference(), boundary_flux=flux)
    )
    spatial[("FDM polynomial, 4th order", flux_name)] = FiniteDifference(
        gradient=BoundaryFluxDifference(
            derivative=PolynomialDifference(accuracy_order=4), boundary_flux=flux
        )
    )
spatial[("FDM WENO5 flux splitting", "Lax-Friedrichs splitting")] = FiniteDifference(
    gradient=WENOFluxDifference(
        flux_splitting=GlobalLaxFriedrichsSplit(), reconstruction=WENO5Flux()
    )
)

# Time integrators: explicit one-step and multistep methods, implicit Runge-Kutta, Adams-Moulton
# and BDF methods, Rosenbrock methods and the adaptive integrators. Implicit stages take a Newton
# solver with a coloured finite-difference Jacobian and a sparse direct solve; the linearly implicit
# Rosenbrock stages take the sparse direct solve alone. The nonlinear and linear solvers are then
# varied on their own.

integrators = {
    "forward Euler": ForwardEuler(),
    "SSP-RK2": TimeMethod("ssprk2"),
    "classical RK4": ClassicalRungeKutta4(),
    "Adams-Bashforth 2": TimeMethod("adams_bashforth2"),
    "ABM2 predictor-corrector": TimeMethod("predictor_corrector_abam2"),
    "Heun-Euler (embedded)": TimeMethod("embedded_heun_euler"),
    "leapfrog": TimeMethod("leapfrog"),
    "adaptive RK45": AdaptiveRungeKutta45(),
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
    "adaptive BDF": AdaptiveBDF(),
}
newton = Newton(linear_solver=SparseDirect(), jacobian="colored_finite_difference")
linear_solvers = {
    "sparse LU": SparseDirect(),
    "GMRES": GMRES(),
    "GMRES + ILU": GMRES(preconditioner=ILU()),
    "right GMRES": RightGMRES(),
    "flexible GMRES": FlexibleGMRES(),
}
nonlinear_solvers = {
    **{
        (name, linear_name): kind(
            linear_solver=linear, jacobian="colored_finite_difference"
        )
        for name, kind in (
            ("Newton", Newton),
            ("damped Newton", DampedNewton),
            ("trust-region Newton", TrustRegionNewton),
            ("quasi-Newton", QuasiNewton),
        )
        for linear_name, linear in linear_solvers.items()
    },
    **{
        ("JFNK", linear_name): JFNK(linear_solver=linear)
        for linear_name, linear in linear_solvers.items()
        if linear_name != "flexible GMRES"
    },
    ("fixed point", "none"): FixedPoint(),
    ("fixed point + Aitken", "none"): FixedPoint(acceleration="aitken"),
}

# ## The test matrix
#
# Three cuts through the space of combinations:
#
# 1. **Space × Riemann solver** — every spatial scheme with every Riemann solver, advanced with
#    SSP-RK2 at the stable step (safety factor 0.5).
# 2. **Time integrator × space** — every integrator with HLLC fluxes. Explicit integrators run on
#    first-order, MUSCL-minmod and WENO5 volumes at their stable step; implicit integrators run on the
#    first-order and the MUSCL-minmod volumes with $\Delta t=2\cdot10^{-3}$, a CFL number of about
#    1.4 that no explicit method could take.
# 3. **Nonlinear × linear solver** — backward Euler on MUSCL-minmod-HLLC with
#    $\Delta t=2\cdot10^{-3}$. One pair is left out for cost: Jacobian-free Newton-Krylov with
#    flexible GMRES runs, but took 800 s already on 40 cells, about 40 times as long as with GMRES.
#
# A few representative combinations are repeated on 200 cells for the mesh check.

time_spatial = {
    name: spatial[(name, "HLLC")]
    for name in ("FVM first order", "FVM MUSCL + minmod", "FVM WENO5")
}
runs = [
    dict(
        matrix="space x Riemann solver",
        row=row,
        column=column,
        cells=cells,
        spatial=scheme,
        time=TimeMethod("ssprk2"),
        nonlinear=None,
        step=end_time,
    )
    for (row, column), scheme in spatial.items()
]
runs += [
    dict(
        matrix="time integrator x space",
        row=name,
        column=space,
        cells=cells,
        spatial=scheme,
        time=integrator,
        nonlinear=(
            newton
            if getattr(integrator, "implicit", False)
            and "rosenbrock" not in getattr(integrator, "method_id", "")
            else None
        ),
        linear=(
            SparseDirect()
            if "rosenbrock" in getattr(integrator, "method_id", "")
            else None
        ),
        step=2e-3 if getattr(integrator, "implicit", False) else end_time,
    )
    for name, integrator in integrators.items()
    for space, scheme in time_spatial.items()
    if space != "FVM WENO5" or not getattr(integrator, "implicit", False)
]
runs += [
    dict(
        matrix="nonlinear x linear solver",
        row=row,
        column=column,
        cells=cells,
        spatial=time_spatial["FVM MUSCL + minmod"],
        time=BackwardEuler(),
        nonlinear=solver,
        step=2e-3,
    )
    for (row, column), solver in nonlinear_solvers.items()
]
representative = [
    ("FVM first order", "Rusanov"),
    ("FVM first order", "HLLC"),
    ("FVM MUSCL + minmod", "HLLC"),
    ("FVM MUSCL + superbee", "HLLC"),
    ("FVM WENO5", "HLLC"),
    ("FDM WENO5 flux splitting", "Lax-Friedrichs splitting"),
    ("DG P2 + minmod + invariant domain", "HLLC"),
    ("CG P1 + graph viscosity", "HLLC"),
]
runs += [
    {**run, "matrix": "mesh check", "cells": coarse_cells}
    for run in runs
    if run["matrix"] == "space x Riemann solver"
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
# its validation message; a run that stops because the solution left the physical state space
# (negative density or pressure) is recorded as failed.

for run in runs:
    simulation = Simulation(
        problem=sod,
        mesh=meshes[run["cells"]],
        numerics=Numerics(
            spatial=run["spatial"],
            linear_solver=(
                SparseDirect()
                if isinstance(run["spatial"], FiniteElement)
                else run.get("linear")
            ),
            time_integrator=run["time"],
            nonlinear_solver=run["nonlinear"],
            step_limits=TimeStepLimits(stability_safety_factor=0.5),
        ),
    )
    started = time.perf_counter()
    check = simulation.validate()
    run.update(status="rejected", note="", errors=(np.nan,) * 3, shock=np.nan, steps=0)
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = simulation.run(
                start_time=0.0, end_time=end_time, time_step=run["step"]
            )
            points = np.column_stack(
                (centres[run["cells"]], np.zeros((run["cells"], 2)))
            )
            run["profiles"] = np.array(
                [
                    result.field(field).evaluate(points).reshape(-1)
                    for field in (gas.density, gas.velocity, gas.pressure)
                ]
            )
            run["errors"] = tuple(
                np.abs(run["profiles"] - exact[run["cells"]]).sum(axis=1) / run["cells"]
            )
            crossing = np.flatnonzero(run["profiles"][0] > shock_level)[-1]
            run["shock"] = np.interp(
                shock_level,
                run["profiles"][0][[crossing + 1, crossing]],
                centres[run["cells"]][[crossing + 1, crossing]],
            )
            run["steps"] = len(result.times) - 1
            run["status"] = "ran" if np.isfinite(run["profiles"]).all() else "failed"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "failed"
                if "non-positive" in message or "finite" in message
                else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    print(
        f"{run['matrix'][:12]:12s} {run['row'][:34]:34s} {run['column'][:24]:24s} "
        f"{run['status']:8s} {run['seconds']:7.1f} s  "
        + (
            f"L1 rho {run['errors'][0]:.2e}  shock {run['shock'] - exact_shock:+.1e}"
            if run["status"] == "ran"
            else run["note"][:90]
        )
    )

# ## Space × Riemann solver
#
# The density error of every spatial scheme with every Riemann solver. Empty cells name why a
# combination did not produce a solution.

matrix = [r for r in runs if r["matrix"] == "space x Riemann solver"]
rows, columns = list(dict.fromkeys(r["row"] for r in matrix)), [
    *riemann,
    "Lax-Friedrichs splitting",
]
grid = {(r["row"], r["column"]): r for r in matrix}
fig = plots.figure(1, 1, size=(13, 17))
plots.matrix(
    [
        [grid[(row, c)]["errors"][0] if (row, c) in grid else np.nan for c in columns]
        for row in rows
    ],
    rows,
    columns,
    scale="log",
    label="L1 density error at t = 0.2",
    text=[
        [
            (
                f"{grid[(row, c)]['errors'][0]:.2e}"
                if (row, c) in grid and grid[(row, c)]["status"] == "ran"
                else grid[(row, c)]["status"] if (row, c) in grid else "–"
            )
            for c in columns
        ]
        for row in rows
    ],
    title=f"Space x Riemann solver, {cells} cells, SSP-RK2",
    ax=fig[0],
)
plots.show()

# All four measures for the combinations that ran, sorted by the density error.

print(
    f"{'scheme':36s} {'Riemann solver':24s} {'L1 rho':>9s} {'L1 u':>9s} {'L1 p':>9s} "
    f"{'shock err':>10s} {'steps':>6s} {'time':>8s}"
)
for r in sorted(
    (r for r in matrix if r["status"] == "ran"), key=lambda r: r["errors"][0]
):
    print(
        f"{r['row']:36s} {r['column']:24s} {r['errors'][0]:9.2e} {r['errors'][1]:9.2e} "
        f"{r['errors'][2]:9.2e} {r['shock'] - exact_shock:+10.1e} {r['steps']:6d} {r['seconds']:7.1f}s"
    )

# ## Profiles of representative combinations
#
# Density, velocity and pressure at $t=0.2$ against the exact solution, and a close-up of the
# contact and the shock.

shown = [
    r
    for r in matrix
    if (r["row"], r["column"]) in representative and r["status"] == "ran"
]
fig = plots.figure(2, 2, size=(15, 9))
for panel, index, name, window in (
    (fig[0], 0, "density", None),
    (fig[1], 1, "velocity", None),
    (fig[2], 2, "pressure", None),
    (fig[3], 0, "density, contact and shock", (0.6, 0.95)),
):
    window_y = None if window is None else (0.1, 0.45)
    plots.line(
        y={
            f"{r['row']}, {r['column']}": (centres[cells], r["profiles"][index])
            for r in shown
        },
        reference={"exact Riemann solution": (fine, exact["fine"][index])},
        xlabel="position x",
        ylabel=name.split(",")[0],
        title=f"{name.capitalize()} at t = {end_time}",
        xlimits=window,
        ylimits=window_y,
        legend="upper right" if index == 0 and window is None else False,
        ax=panel,
    )
plots.show()

# ## Time integrator × space
#
# Explicit integrators at their stable step, implicit integrators at $\Delta t=2\cdot10^{-3}$. A dash
# marks a combination outside this cut (implicit WENO5).

timing = [r for r in runs if r["matrix"] == "time integrator x space"]
time_rows, time_columns = list(dict.fromkeys(r["row"] for r in timing)), list(time_spatial)
time_grid = {(r["row"], r["column"]): r for r in timing}
fig = plots.figure(1, 2, size=(17, 11))
plots.matrix(
    [
        [
            time_grid[(row, c)]["errors"][0] if (row, c) in time_grid else np.nan
            for c in time_columns
        ]
        for row in time_rows
    ],
    time_rows,
    time_columns,
    scale="log",
    label="L1 density error",
    text=[
        [
            (
                f"{time_grid[(row, c)]['errors'][0]:.2e}"
                if time_grid[(row, c)]["status"] == "ran"
                else time_grid[(row, c)]["status"]
            )
            if (row, c) in time_grid
            else "–"
            for c in time_columns
        ]
        for row in time_rows
    ],
    title="L1 density error",
    ax=fig[0],
)
plots.matrix(
    [
        [
            (
                time_grid[(row, c)]["seconds"]
                if (row, c) in time_grid and time_grid[(row, c)]["status"] == "ran"
                else np.nan
            )
            for c in time_columns
        ]
        for row in time_rows
    ],
    time_rows,
    time_columns,
    scale="log",
    label="wall time [s]",
    text=[
        [
            (
                f"{time_grid[(row, c)]['seconds']:.0f} s, {time_grid[(row, c)]['steps']} steps"
                if time_grid[(row, c)]["status"] == "ran"
                else time_grid[(row, c)]["status"]
            )
            if (row, c) in time_grid
            else "–"
            for c in time_columns
        ]
        for row in time_rows
    ],
    title="Wall time and accepted steps",
    ax=fig[1],
)
plots.show()

# ## Nonlinear × linear solver
#
# Backward Euler on MUSCL-minmod-HLLC. The discrete solution should not depend on the solver as long
# as each one converges; the wall time does.

solving = [r for r in runs if r["matrix"] == "nonlinear x linear solver"]
solver_rows = list(dict.fromkeys(r["row"] for r in solving))
solver_columns = [*linear_solvers, "none"]
solver_grid = {(r["row"], r["column"]): r for r in solving}
fig = plots.figure(1, 1, size=(13, 6))
plots.matrix(
    [
        [
            (
                solver_grid[(row, c)]["seconds"]
                if (row, c) in solver_grid and solver_grid[(row, c)]["status"] == "ran"
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
                    f"{solver_grid[(row, c)]['seconds']:.0f} s\n"
                    f"{solver_grid[(row, c)]['errors'][0]:.3e}"
                )
                if solver_grid[(row, c)]["status"] == "ran"
                else solver_grid[(row, c)]["status"]
            )
            if (row, c) in solver_grid
            else ""
            for c in solver_columns
        ]
        for row in solver_rows
    ],
    title="Backward Euler: wall time and L1 density error per solver pair",
    ax=fig[0],
)
plots.show()

# ## Mesh check: 400 against 200 cells
#
# The representative combinations on half as many cells. For a discontinuous solution the $L_1$
# error falls roughly like $\Delta x^{1/2}$ to $\Delta x^{1}$; the shock position converges to its
# exact value.

coarse = {(r["row"], r["column"]): r for r in runs if r["matrix"] == "mesh check"}
print(
    f"{'scheme':36s} {'Riemann solver':24s} {'L1 rho 200':>11s} {'L1 rho 400':>11s} "
    f"{'ratio':>6s} {'shock err 200':>14s} {'shock err 400':>14s}"
)
for key, r in coarse.items():
    fine_run = grid[key]
    if r["status"] == "ran" and fine_run["status"] == "ran":
        print(
            f"{key[0]:36s} {key[1]:24s} {r['errors'][0]:11.3e} {fine_run['errors'][0]:11.3e} "
            f"{r['errors'][0] / fine_run['errors'][0]:6.2f} {r['shock'] - exact_shock:+14.2e} "
            f"{fine_run['shock'] - exact_shock:+14.2e}"
        )

# ## What the library refused, and what failed
#
# A refused combination is caught by validation or binding before any time step; a failed one
# started and produced a non-physical state.

for status in ("rejected", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(f"  {r['matrix'][:12]:12s} {r['row'][:34]:34s} {r['column'][:24]:24s} {r['note'][:100]}")
print(
    "\nsummary: "
    + ", ".join(
        f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed")
    )
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.0f} min in total"
)

# ## Going further
#
# Swap the Riemann problem for Lax's or the 123 problem by changing the two initial states; the
# matrix code stays as it is. Add a column by adding one object to a dictionary: a new limiter, a
# different flux, another integrator.
