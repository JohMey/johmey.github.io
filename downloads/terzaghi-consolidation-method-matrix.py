"""OneSimulation example: Terzaghi's one-dimensional consolidation through every method combination.
Requires the installed, activated OneSimulation runtime.

One physics declaration, a saturated clay layer under a suddenly applied load, is solved on 200
cells with every applicable combination of discretization family, boundary enforcement, time
integrator, nonlinear solver and linear solver. Each run is graded against Terzaghi's Fourier
series for the excess pore pressure and the degree of consolidation.
"""

# # Terzaghi consolidation: one model, every method
#
# A saturated clay layer of thickness $H=1$ m drains through its top ($z=0$) into a sand layer and
# rests on impermeable rock ($z=H$). At $t=0$ a load $\sigma=100$ kPa is applied; the pore water
# first carries it all, $p(z,0)=\sigma$, and then flows out through the top. OneSimulation's
# `TerzaghiConsolidation` model couples three balances:
#
# $$S\,\partial_t p-\frac{k}{\mu}\,\partial_z^2 p=0,\qquad
# \partial_t\varepsilon=-m_v\,\partial_t p,\qquad
# \partial_t e=-(1+e)\,\partial_t\varepsilon,$$
#
# for the excess pore pressure $p$, the vertical strain $\varepsilon$ and the void ratio $e$, with the
# constrained storage $S$, the intrinsic permeability $k$, the water viscosity $\mu$ and the
# constrained compressibility $m_v$. The pressure obeys a diffusion equation with the coefficient of
# consolidation $c_v=k/(\mu S)$. The model is declared once; everything numerical is an argument of
# `Numerics`.
#
# **Reference.** Terzaghi's solution (K. Terzaghi, *Theoretical Soil Mechanics*, 1943) in the time
# factor $T_v=c_v t/H^2$ is
#
# $$p(z,t)=\sum_{m=0}^{\infty}\frac{2\sigma}{M}\sin\Big(\frac{Mz}{H}\Big)e^{-M^2T_v},\qquad
# M=\frac{\pi}{2}(2m+1),$$
#
# and the degree of consolidation, the settlement relative to its final value, is
# $U(T_v)=1-\sum_m \frac{2}{M^2}e^{-M^2T_v}$. Strain and void ratio follow exactly:
# $\varepsilon=m_v(\sigma-p)$ and $e=(1+e_0)e^{-\varepsilon}-1$.
#
# **Grading.** At $T_v=0.2$ (about 50 % consolidation) every run is evaluated at the 200 cell
# centres: the $L_2$ error of the pore pressure relative to the load, the error of the degree of
# consolidation from the computed settlement $\int_0^H\varepsilon\,dz$, and the largest deviation of
# the void ratio from its exact relation to the computed strain. The wall time covers validation,
# binding and time stepping.

import time

import numpy as np
from onesim.cad.d1 import Line
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
    AutomaticLinearSolver,
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
    SpectralEnclosureStability,
    TimeMethod,
)
from onesim.physics.geotechnics import (
    DrainedBoundary,
    ImpermeableBoundary,
    TerzaghiConsolidation,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# Nothing in this cell names a numerical method.

thickness, load, void_ratio = 1.0, 1.0e5, 1.0
permeability, viscosity, storage, compressibility = 1.0e-16, 1.0e-3, 1.0e-7, 1.0e-7
consolidation_coefficient = permeability / (viscosity * storage)
time_factor = 0.2
end_time = time_factor * thickness**2 / consolidation_coefficient
cells, coarse_cells = 200, 100
layer = Line(start=(0.0, 0.0, 0.0), end=(thickness, 0.0, 0.0))
clay = TerzaghiConsolidation(
    domain=layer.line,
    permeability=permeability,
    fluid_viscosity=viscosity,
    constrained_storage=storage,
    constrained_compressibility=compressibility,
    total_stress=load,
)
terzaghi = Problem(
    name="Terzaghi consolidation",
    geometry=layer,
    physics=(clay,),
    boundary_conditions=(
        DrainedBoundary(physics=clay, boundary=layer.boundaries.start),
        ImpermeableBoundary(physics=clay, boundary=layer.boundaries.end),
    ),
    initial_conditions=(
        InitialValue(clay.pore_pressure, load),
        InitialValue(clay.vertical_strain, 0.0),
        InitialValue(clay.void_ratio, void_ratio),
    ),
)
meshes = {n: StructuredLineMesher(cells=n).generate(layer) for n in (cells, coarse_cells)}
centres = {n: mesh.cell_centers[:, 0] for n, mesh in meshes.items()}
print(
    f"{meshes[cells].cell_count} cells of {thickness / cells * 1e3:.1f} mm "
    f"(coarse check: {meshes[coarse_cells].cell_count} cells); c_v = "
    f"{consolidation_coefficient:.1e} m²/s, T_v = {time_factor} reached at t = {end_time:.3e} s "
    f"({end_time / 86400:.2f} days)"
)

# ## The exact solution
#
# Terzaghi's series with 2000 terms at the cell centres and on a fine grid, at the final time factor
# and at three earlier ones for the isochrone plot.

terms = np.pi * (2 * np.arange(2000) + 1) / 2
isochrones = (0.02, 0.05, 0.1, time_factor)
exact = {
    key: np.sum(
        2 * load / terms[:, None]
        * np.sin(terms[:, None] * points[None, :] / thickness)
        * np.exp(-terms[:, None] ** 2 * time_factor),
        axis=0,
    )
    for key, points in {**centres, "fine": np.linspace(0.0, thickness, 1001)}.items()
}
fine = np.linspace(0.0, thickness, 1001)
exact_isochrones = {
    factor: np.sum(
        2 * load / terms[:, None]
        * np.sin(terms[:, None] * fine[None, :] / thickness)
        * np.exp(-terms[:, None] ** 2 * factor),
        axis=0,
    )
    for factor in isochrones
}
exact_degree = 1 - np.sum(2 / terms**2 * np.exp(-(terms**2) * time_factor))
print(
    f"degree of consolidation U(T_v = {time_factor}) = {exact_degree:.6f}, final settlement "
    f"{compressibility * load * thickness * 1e3:.1f} mm, settlement now "
    f"{exact_degree * compressibility * load * thickness * 1e3:.3f} mm"
)

# ## The method catalogue
#
# Finite differences with the central and the 4th- and 6th-order polynomial stencils (a gradient
# stencil for the Darcy flux, a Laplacian for the balance); finite volumes with the two-point flux;
# continuous finite elements of degree 1 to 3 with the drained boundary imposed weakly by Nitsche's
# method (symmetric, incomplete and nonsymmetric variants); discontinuous finite elements of degree 1
# to 3 with the symmetric interior penalty.
#
# **Explicit step.** The pressure balance is parabolic; its spectrum lies on the negative real axis.
# Every spatial scheme is given the same explicit-step rule,
# `SpectralEnclosureStability(enclosure="negative_real_axis")`: it estimates the spectral radius of
# the executed coupled operator (Arnoldi iteration on its residual) and returns the largest step for
# which the selected method's amplification stays below one on that axis. Implicit methods are not
# limited by it; their step is $\Delta t=t_{end}/200$.

rule = SpectralEnclosureStability(enclosure="negative_real_axis", spectral_radius="arnoldi")
spatial = {
    "FDM central, 2nd order": FiniteDifference(
        gradient=CentralDifference(), diffusion=CentralLaplacian(), stability=rule
    ),
    **{
        f"FDM polynomial, {order}th order": FiniteDifference(
            gradient=PolynomialDifference(accuracy_order=order),
            diffusion=PolynomialLaplacian(accuracy_order=order),
            stability=rule,
        )
        for order in (4, 6)
    },
    "FVM two-point flux": FiniteVolume(diffusion=TwoPointDiffusion(), stability=rule),
    **{
        f"FEM CG P{degree} + Nitsche": FiniteElement(
            element=Lagrange(degree), boundary_enforcement=Nitsche(), stability=rule
        )
        for degree in (1, 2, 3)
    },
    **{
        f"FEM CG P2 + {name} Nitsche": FiniteElement(
            element=Lagrange(2),
            boundary_enforcement=Nitsche(symmetry=symmetry),
            stability=rule,
        )
        for name, symmetry in (("incomplete", 0.0), ("nonsymmetric", -1.0))
    },
    **{
        f"FEM DG P{degree} + SIPG": FiniteElement(
            element=DiscontinuousLagrange(degree),
            diffusion=SymmetricInteriorPenalty(),
            stability=rule,
        )
        for degree in (1, 2, 3)
    },
}

# Time integrators: explicit one-step and multistep methods, implicit Runge-Kutta, Adams-Moulton and
# BDF methods, Rosenbrock methods and the adaptive integrators. Implicit stages take a Newton solver
# with the analytic Jacobian of the coupled three-field system and a sparse direct solve; the
# linearly implicit Rosenbrock stages take the sparse direct solve alone.

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
    "leapfrog + RAW filter": TimeMethod("leapfrog_raw"),
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
    "BDF2 (Gear)": TimeMethod("bdf2_gear"),
    **{f"BDF{k} (Gear)": TimeMethod(f"bdf{k}_gear", startup_policy="implicit_extrapolation") for k in (3, 4, 5, 6)},
    "Rosenbrock-Euler": TimeMethod("rosenbrock_euler"),
    "Rosenbrock-Wanner Euler": TimeMethod("rosenbrock_wanner_euler"),
    "adaptive BDF": AdaptiveBDF(relative_tolerance=1e-4, absolute_tolerance=1.0),
}
implicit_step = end_time / 200
newton = Newton(linear_solver=SparseDirect(), jacobian="analytic")
# Crank-Nicolson, the implicit midpoint rule and Adams-Moulton 2 are A-stable but not L-stable:
# the stiffest modes, excited by the jump between the loaded interior and the drained boundary, are
# reflected with |R| -> 1 instead of damped. Where they make the coupled void-ratio balance leave its
# admissible range, the run must fail; the stabilized counterpart is the theta method with
# theta = 0.75 (one ingredient: more implicit weight), shown behind the hatching.
# The fixed-step Gear BDF methods of order 3 to 6 start their history with extrapolated
# backward-Euler steps (`TimeMethod("bdf<k>_gear", startup_policy="implicit_extrapolation")`) and
# run at the implicit step. Adams-Moulton 3 and 4 have bounded stability intervals on the negative
# real axis; they are repeated at Δt = 1/ρ, with ρ = 4c_v/Δz² the spectral radius of the
# central-difference operator, inside their stability interval. The fixed-point (Picard)
# iteration contracts for Δt < 1/ρ and is repeated at Δt = 0.5/ρ. On the finite elements the
# spectral radius is far larger (boundary penalty modes), so Adams-Moulton 3 and 4 stay failed there.
oscillating = ("Crank-Nicolson", "implicit midpoint", "Adams-Moulton 2")
radius = 4 * consolidation_coefficient / (thickness / cells) ** 2
stable_step = {
    "Adams-Moulton 3": 1.0 / radius,
    "Adams-Moulton 4": 1.0 / radius,
    "fixed point": 0.5 / radius,
    "fixed point + Aitken": 0.5 / radius,
}
linear_solvers = {
    "sparse LU": SparseDirect(),
    "GMRES": GMRES(),
    "GMRES + ILU": GMRES(preconditioner=ILU()),
    "right GMRES": RightGMRES(),
    "flexible GMRES": FlexibleGMRES(),
    "automatic": AutomaticLinearSolver(),
}
nonlinear_solvers = {
    **{
        (name, linear_name): kind(linear_solver=linear, jacobian="analytic")
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
    },
    ("fixed point", "none"): FixedPoint(),
    ("fixed point + Aitken", "none"): FixedPoint(acceleration="aitken"),
}
# Krylov solvers without a preconditioner stall on the coupled three-field stage system; their
# counterpart adds an ILU preconditioner (one ingredient).
preconditioned = {
    (name, linear_name): (
        JFNK(linear_solver=kind(preconditioner=ILU()))
        if name == "JFNK"
        else solver_kind(linear_solver=kind(preconditioner=ILU()), jacobian="analytic")
    )
    for name, solver_kind in (
        ("Newton", Newton),
        ("damped Newton", DampedNewton),
        ("trust-region Newton", TrustRegionNewton),
        ("quasi-Newton", QuasiNewton),
        ("JFNK", JFNK),
    )
    for linear_name, kind in (
        ("GMRES", GMRES),
        ("right GMRES", RightGMRES),
        ("flexible GMRES", FlexibleGMRES),
    )
}

# ## The test matrix
#
# Three cuts through the space of combinations:
#
# 1. **Space × time integrator** — every spatial scheme with backward Euler, BDF2, Crank-Nicolson
#    and SDIRK2 at $\Delta t=t_{end}/200$, and the finite differences and volumes also with explicit
#    classical RK4 at its stable step.
# 2. **Time integrator × space** — every integrator on the central differences and the finite
#    volumes; the implicit integrators also on continuous and discontinuous P1 elements.
# 3. **Nonlinear × linear solver** — backward Euler on the central differences (a coupled system of
#    three fields with a pressure-dependent void-ratio mass).
#
# Explicit finite elements are outside these cuts (a dash in the matrices): the weakly imposed
# drained boundary (Nitsche) and the interior penalty add boundary modes about 25 times stiffer than
# the interior diffusion, so an explicit step at 200 cells takes tens of thousands of stages of the
# coupled three-field element operator. The library runs them with the same stability rule; they
# are simply not a sensible way to compute this case.
#
# A few representative combinations are repeated on 100 cells for the mesh check.

columns = {
    "classical RK4": ClassicalRungeKutta4(),
    "backward Euler": BackwardEuler(),
    "BDF2 (Gear)": TimeMethod("bdf2_gear"),
    "Crank-Nicolson": TimeMethod("crank_nicolson"),
    "SDIRK2": TimeMethod("dirk_sdirk2"),
}
time_spatial = {
    name: spatial[name]
    for name in (
        "FDM central, 2nd order",
        "FVM two-point flux",
        "FEM CG P1 + Nitsche",
        "FEM DG P1 + SIPG",
    )
}
explicit_rows = {name for name in spatial if name.startswith(("FDM", "FVM"))}
runs = [
    dict(matrix="space x time", row=row, column=column, cells=cells, spatial=scheme, time=method)
    for row, scheme in spatial.items()
    for column, method in columns.items()
    if getattr(method, "implicit", False) or row in explicit_rows
]
runs += [
    dict(
        matrix="time integrator x space",
        row=name,
        column=space,
        cells=cells,
        spatial=scheme,
        time=integrator,
    )
    for name, integrator in integrators.items()
    for space, scheme in time_spatial.items()
    if getattr(integrator, "implicit", False) or space in explicit_rows
]
runs += [
    dict(
        matrix="nonlinear x linear solver",
        row=row,
        column=column,
        cells=cells,
        spatial=spatial["FDM central, 2nd order"],
        time=BackwardEuler(),
        nonlinear=solver,
    )
    for (row, column), solver in nonlinear_solvers.items()
]
runs += [
    {
        **run,
        "variant": "stabilized",
        "time": TimeMethod("theta_075"),
        "note_stabilized": "theta = 0.75 instead of the undamped method",
    }
    for run in list(runs)
    if (run["column"] if run["matrix"] == "space x time" else run["row"]) in oscillating
]
runs += [
    {
        **run,
        "variant": "stabilized",
        "step": stable_step[run["row"]],
        "note_stabilized": f"{run['row']} at Δt = {stable_step[run['row']]:.3g} s",
    }
    for run in list(runs)
    if run["matrix"] in ("time integrator x space", "nonlinear x linear solver")
    and run["row"] in stable_step
    and run.get("variant") is None
]
runs += [
    {**run, "variant": "stabilized", "nonlinear": preconditioned[(run["row"], run["column"])]}
    for run in list(runs)
    if run["matrix"] == "nonlinear x linear solver"
    and (run["row"], run["column"]) in preconditioned
    and run.get("variant") is None
]
representative = [
    ("FDM central, 2nd order", "backward Euler"),
    ("FDM polynomial, 4th order", "SDIRK2"),
    ("FVM two-point flux", "BDF2 (Gear)"),
    ("FEM CG P1 + Nitsche", "BDF2 (Gear)"),
    ("FEM CG P2 + Nitsche", "SDIRK2"),
    ("FEM DG P2 + SIPG", "SDIRK2"),
]
runs += [
    {**run, "matrix": "mesh check", "cells": coarse_cells}
    for run in runs
    if run["matrix"] == "space x time"
    and run.get("variant") is None
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
# A stabilized counterpart is run only where the declared combination did not run.
# One loop, the same problem object every time. A combination the library refuses is recorded with
# its validation message; a run that stops because the void ratio left its admissible range or the
# state became non-finite is recorded as failed.

for run in runs:
    if run.get("variant") == "stabilized" and any(
        other.get("variant") is None
        and other.get("status") == "ran"
        and (other["matrix"], other["row"], other["column"])
        == (run["matrix"], run["row"], run["column"])
        for other in runs
    ):
        run.update(status="not needed", note="", error=np.nan, degree=np.nan, void=np.nan, steps=0)
        run["seconds"] = 0.0
        continue
    implicit = getattr(run["time"], "implicit", False)
    rosenbrock = "rosenbrock" in getattr(run["time"], "method_id", "")
    simulation = Simulation(
        problem=terzaghi,
        mesh=meshes[run["cells"]],
        numerics=Numerics(
            spatial=run["spatial"],
            linear_solver=(
                SparseDirect()
                if isinstance(run["spatial"], FiniteElement) or rosenbrock
                else None
            ),
            time_integrator=run["time"],
            nonlinear_solver=(
                run.get("nonlinear", newton) if implicit and not rosenbrock else None
            ),
        ),
    )
    started = time.perf_counter()
    check = simulation.validate()
    run.update(status="rejected", note="", error=np.nan, degree=np.nan, void=np.nan, steps=0)
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = simulation.run(
                start_time=0.0,
                end_time=end_time,
                time_step=run.get("step", implicit_step if implicit else end_time),
            )
            points = np.column_stack((centres[run["cells"]], np.zeros((run["cells"], 2))))
            run["profile"], strain, voids = (
                result.field(field).evaluate(points).reshape(-1)
                for field in (clay.pore_pressure, clay.vertical_strain, clay.void_ratio)
            )
            run["error"] = np.sqrt(np.mean((run["profile"] - exact[run["cells"]]) ** 2)) / load
            run["degree"] = np.mean(strain) / (compressibility * load) - exact_degree
            run["void"] = np.max(abs(voids - ((1 + void_ratio) * np.exp(-strain) - 1)))
            run["steps"] = len(result.times) - 1
            run["status"] = "ran" if np.isfinite(run["error"]) and run["error"] < 1 else "failed"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "failed"
                if "finite" in message
                or "void ratio" in message
                or "did not converge" in message
                or "maximum iteration" in message
                else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    print(
        f"{run['matrix'][:12]:12s} {run['row'][:30]:30s} {run['column'][:24]:24s} "
        f"{run.get('variant') or '':10s} {run['status']:8s} {run['seconds']:7.1f} s  "
        + (
            f"L2 {run['error']:.2e}  dU {run['degree']:+.1e}"
            if run["status"] == "ran"
            else run["note"][:90]
        )
    )

# ## Space × time integrator
#
# The pore-pressure error of every spatial scheme with an explicit and four implicit integrators.

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
legend_head = (
    "Hatched: the declared integrator reflects the stiffest modes undamped (not L-stable) and the "
    "void-ratio balance leaves its admissible range; colour and value show theta = 0.75."
)

space_key = "space x time"
rows = list(spatial)
fig = plots.figure(1, 1, size=(14, 9))
plots.matrix(
    [
        [cell[(space_key, row, c)]["error"] if (space_key, row, c) in cell else np.nan for c in columns]
        for row in rows
    ],
    rows,
    list(columns),
    scale="log",
    label="relative L2 pore-pressure error at T_v = 0.2",
    text=[
        [
            (
                f"{cell[(space_key, row, c)]['error']:.2e}\n{cell[(space_key, row, c)]['steps']} steps"
                if cell[(space_key, row, c)]["status"] == "ran"
                else cell[(space_key, row, c)]["status"]
            )
            if (space_key, row, c) in cell
            else "–"
            for c in columns
        ]
        for row in rows
    ],
    hatched=[[(space_key, row, c) in hatch for c in columns] for row in rows],
    notes=[legend_head] if any(key[0] == space_key for key in hatch) else None,
    title=f"Space x time integrator, {cells} cells",
    ax=fig[0],
)
plots.show()

# All measures for the combinations that ran, sorted by the pressure error.

matrix = [r for r in runs if r["matrix"] == space_key and r.get("variant") is None]
print(
    f"{'scheme':34s} {'time':16s} {'L2 p/σ':>9s} {'U error':>9s} {'void dev':>9s} "
    f"{'steps':>6s} {'time':>8s}"
)
for r in sorted((r for r in matrix if r["status"] == "ran"), key=lambda r: r["error"]):
    print(
        f"{r['row']:34s} {r['column']:16s} {r['error']:9.2e} {r['degree']:+9.1e} "
        f"{r['void']:9.1e} {r['steps']:6d} {r['seconds']:7.1f}s"
    )

# ## Isochrones and representative profiles
#
# Exact isochrones at $T_v=0.02,\,0.05,\,0.1,\,0.2$ and the computed pore pressure of the
# representative combinations at $T_v=0.2$.

shown = [
    r for r in matrix if (r["row"], r["column"]) in representative and r["status"] == "ran"
]
fig = plots.figure(1, 1, size=(10, 6))
plots.line(
    y={
        f"{r['row']}, {r['column']}": (centres[cells], r["profile"] / load) for r in shown
    },
    reference={
        f"exact, T_v = {factor}": (fine, exact_isochrones[factor] / load) for factor in isochrones
    },
    xlabel="depth z [m]",
    ylabel="excess pore pressure p / σ",
    title="Pore-pressure isochrones",
    legend="upper right",
    ax=fig[0],
)
plots.show()

# ## Time integrator × space
#
# Explicit integrators at their stable step, implicit integrators at $\Delta t=t_{end}/200$.

time_key = "time integrator x space"
time_rows, time_columns = list(integrators), list(time_spatial)
fig = plots.figure(1, 2, size=(18, 12))
time_hatched = [[(time_key, row, c) in hatch for c in time_columns] for row in time_rows]
time_notes = (
    [
        "Hatched: the declared integrator fails at Δt = t_end/200; colour and value show its "
        "stabilized counterpart:"
    ]
    + [
        f"  {row}: theta = 0.75 (damps the stiff modes it reflects undamped)."
        for row in oscillating
        if any((time_key, row, c) in hatch for c in time_columns)
    ]
    + [
        f"  {row}: same method at Δt = {stable_step[row]:.3g} s (1/ρ)."
        for row in stable_step
        if any((time_key, row, c) in hatch for c in time_columns)
    ]
)
for panel, measure, label, title in (
    (fig[0], "error", "relative L2 pore-pressure error", "Pore-pressure error"),
    (fig[1], "seconds", "wall time [s]", "Wall time and accepted steps"),
):
    plots.matrix(
        [
            [
                cell[(time_key, row, c)][measure]
                if (time_key, row, c) in cell and cell[(time_key, row, c)]["status"] == "ran"
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
                if (time_key, row, c) in cell
                else "–"
                for c in time_columns
            ]
            for row in time_rows
        ],
        hatched=time_hatched,
        notes=time_notes if measure == "error" and len(time_notes) > 1 else None,
        title=title,
        ax=panel,
    )
plots.show()

# ## Nonlinear × linear solver
#
# Backward Euler on the central differences. The void-ratio mass makes the stage system nonlinear;
# every converging solver reaches the same discrete solution, the wall time differs.

solver_key = "nonlinear x linear solver"
solver_rows = list(dict.fromkeys(row for row, _ in nonlinear_solvers))
solver_columns = [*linear_solvers, "none"]
fig = plots.figure(1, 1, size=(13, 6))
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
                    f"{cell[(solver_key, row, c)]['error']:.3e}"
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
        "Hatched: the declared solver fails at Δt = t_end/200; colour and value show the "
        f"fixed-point iteration at Δt = {stable_step['fixed point']:.3g} s (0.5/ρ), or the Krylov "
        "solver with an ILU preconditioner."
    ],
    title="Backward Euler: wall time and relative L2 pore-pressure error per solver pair",
    ax=fig[0],
)
plots.show()

# ## Mesh check: 200 against 100 cells
#
# The representative combinations on half as many cells. Early in the consolidation the pressure
# has a steep boundary layer at the drained top; at $T_v=0.2$ it is smooth, and the error of a
# second-order scheme falls by about four as long as the time error stays below the spatial one.

coarse = {(r["row"], r["column"]): r for r in runs if r["matrix"] == "mesh check"}
grid = {(r["row"], r["column"]): r for r in matrix}
print(f"{'scheme':34s} {'time':16s} {'L2 100':>9s} {'L2 200':>9s} {'ratio':>6s}")
for key, r in coarse.items():
    fine_run = grid[key]
    if r["status"] == "ran" and fine_run["status"] == "ran":
        print(
            f"{key[0]:34s} {key[1]:16s} {r['error']:9.2e} {fine_run['error']:9.2e} "
            f"{r['error'] / fine_run['error']:6.2f}"
        )

# ## What the library refused, and what failed
#
# A refused combination is caught by validation or binding before any time step; a failed one
# started and left the admissible state.

for status in ("rejected", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(
            f"  {r['matrix'][:12]:12s} {r['row'][:30]:30s} {r['column'][:24]:24s} "
            f"{r.get('variant') or '':10s} {r['note'][:90]}"
        )
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.0f} min in total"
)

# ## Methods that do not apply to consolidation
#
# Riemann solvers, reconstructions and limiters are for convective fluxes; the consolidation balance
# has none. The leapfrog scheme without filter is unstable for every dissipative mode (its
# stability region is a segment of the imaginary axis), so only its Robert-Asselin-Williams-filtered
# form is in the matrix. Runge-Kutta-Chebyshev and -Legendre super-steps would suit this parabolic
# operator but are not yet selectable as a time integrator of `Numerics`.
#
# ## Going further
#
# Make both ends drained (`DrainedBoundary` on `layer.boundaries.all`): Terzaghi's series then holds
# with the drainage length $H/2$. Or prescribe a water supply (`fluid_supply=`) for a pumping test;
# the matrix code stays as it is.
