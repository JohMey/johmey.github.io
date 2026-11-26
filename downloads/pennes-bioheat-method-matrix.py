"""OneSimulation example: the Pennes bioheat equation through every method combination OneSimulation offers.
Requires the installed, activated OneSimulation runtime.

One physics declaration, a perfused tissue layer whose skin surface is brought to 45 °C, is solved
with every applicable combination of discretization family, diffusion operator, boundary
enforcement, time integrator, nonlinear and linear solver, as a stationary problem and as a
heating transient. Each run is graded against the closed-form steady state and the
eigenfunction series of the transient.
"""

# # Heating of perfused tissue: one model, every method
#
# Blood flowing through the capillaries removes heat from warm tissue. Pennes (1948) modelled
# this as a sink proportional to the difference between the tissue temperature and the arterial
# blood temperature $T_b$:
#
# $$\rho c\,\partial_tT=k\,\partial_x^2T+\omega_b\rho_bc_b\,(T_b-T).$$
#
# A tissue layer of depth $L$ starts at $T_b$; at $t=0$ its skin surface $x=0$ is brought to
# $T_s=45$ °C, the deep side $x=L$ stays at $T_b$. The model is declared once; everything numerical
# is an argument of `Numerics`.
#
# **Reference.** H. H. Pennes, J. Appl. Physiol. 1 (1948) 93; H. S. Carslaw and J. C. Jaeger,
# *Conduction of Heat in Solids*, 2nd ed. (1959), §3.4. With $\theta=T-T_b$,
# $m^2=\omega_b\rho_bc_b/k$, $\alpha=k/\rho c$, $w=\omega_b\rho_bc_b/\rho c$ and
# $\Delta T=T_s-T_b$:
#
# $$\theta(x,t)=\Delta T\,\frac{\sinh m(L-x)}{\sinh mL}
#   -\Delta T\sum_{n\ge1}\frac{2n\pi}{(n\pi)^2+(mL)^2}\,\sin\frac{n\pi x}{L}\,
#   e^{-(\alpha n^2\pi^2/L^2+w)\,t}.$$
#
# The first term is the steady state.
#
# **Grading.** Every run is evaluated at the cell centres $x_i$: the largest temperature
# deviation $\max_i|T_h(x_i)-T(x_i)|$ from the steady state (stationary solves) or from the
# series at $t=5$ min (transients). The wall time covers validation, binding and the solve.

import time
from dataclasses import replace

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import CentralLaplacian, FiniteDifference, PolynomialLaplacian
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
from onesim.physics.biophysics import PennesBioheat
from onesim.simulation import PrescribedValueBoundary, Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# Soft tissue 30 mm deep (density 1050 kg/m³, heat capacity 3600 J/(kg K), conductivity
# 0.5 W/(m K)), perfused at $\omega_b=5\times10^{-4}$ 1/s by blood with
# $\rho_bc_b=1060\cdot3770$ J/(m³ K).

depth, density, capacity, conductivity = 0.03, 1050.0, 3600.0, 0.5
blood, surface, perfusion = 310.15, 318.15, 5e-4 * 1060.0 * 3770.0
cell_counts, time_cells, moment = (50, 100, 200, 400), 100, 300.0
layer = Line(start=(0.0, 0.0, 0.0), end=(depth, 0.0, 0.0))
tissue = PennesBioheat(
    domain=layer.line,
    density=density,
    heat_capacity=capacity,
    conductivity=conductivity,
    perfusion_capacity_rate=perfusion,
    blood_temperature=blood,
)
heating = Problem(
    name="Pennes bioheat, skin heating",
    geometry=layer,
    physics=(tissue,),
    boundary_conditions=(
        PrescribedValueBoundary(physics=tissue, boundary=layer.boundaries.start, value=lambda time, **_: {"temperature": surface}),
        PrescribedValueBoundary(physics=tissue, boundary=layer.boundaries.end, value=lambda time, **_: {"temperature": blood}),
    ),
    initial_conditions=(InitialValue(tissue.temperature, blood),),
)
meshes = {n: StructuredLineMesher(cells=n).generate(layer) for n in cell_counts}
centres = {n: mesh.cell_centers[:, 0] for n, mesh in meshes.items()}
diffusivity, rate = conductivity / (density * capacity), perfusion / (density * capacity)
m = np.sqrt(perfusion / conductivity)
print(
    f"meshes of {', '.join(str(meshes[n].cell_count) for n in cell_counts)} cells (finest width {1e6 * depth / cell_counts[-1]:.0f} µm); "
    f"penetration depth 1/m = {1e3 / m:.1f} mm, diffusion time L²/α = {depth**2 / diffusivity / 60:.0f} min"
)

# ## The exact solution
#
# The steady profile on every grid, and 4000 terms of the series at $t=5$ min.

n = np.arange(1, 4001)[:, None]
steady = {key: blood + (surface - blood) * np.sinh(m * (depth - x)) / np.sinh(m * depth) for key, x in centres.items()}
transient = {
    key: steady[key]
    - (surface - blood)
    * (
        2 * n * np.pi / ((n * np.pi) ** 2 + (m * depth) ** 2)
        * np.sin(n * np.pi * x / depth)
        * np.exp(-(diffusivity * (n * np.pi / depth) ** 2 + rate) * moment)
    ).sum(axis=0)
    for key, x in centres.items()
}
fine = np.linspace(0.0, depth, 1201)
fine_steady = blood + (surface - blood) * np.sinh(m * (depth - fine)) / np.sinh(m * depth)
fine_transient = fine_steady - (surface - blood) * (
    2 * n * np.pi / ((n * np.pi) ** 2 + (m * depth) ** 2)
    * np.sin(n * np.pi * fine / depth)
    * np.exp(-(diffusivity * (n * np.pi / depth) ** 2 + rate) * moment)
).sum(axis=0)
print(f"T(5 mm): steady {np.interp(0.005, fine, fine_steady) - 273.15:.4f} °C, after 5 min {np.interp(0.005, fine, fine_transient) - 273.15:.4f} °C")

# ## The method catalogue
#
# Finite differences with the central and polynomial Laplacians of order 2, 4 and 6; finite
# volumes with two-point fluxes; continuous Galerkin of degree 1 to 4 with symmetric and
# non-symmetric Nitsche enforcement of the surface temperature; discontinuous Galerkin of degree
# 1 to 4 with symmetric interior penalty at two penalty levels.

spatial = {
    "FDM central Laplacian": FiniteDifference(diffusion=CentralLaplacian()),
    **{f"FDM polynomial Laplacian, order {order}": FiniteDifference(diffusion=PolynomialLaplacian(accuracy_order=order)) for order in (2, 4, 6)},
    "FVM two-point diffusion": FiniteVolume(diffusion=TwoPointDiffusion(), cell_quadrature_order=4),
    **{f"FEM CG P{degree} + symmetric Nitsche": FiniteElement(element=Lagrange(degree), boundary_enforcement=Nitsche()) for degree in (1, 2, 3, 4)},
    **{
        f"FEM CG P{degree} + non-symmetric Nitsche": FiniteElement(element=Lagrange(degree), boundary_enforcement=Nitsche(symmetry=-1.0))
        for degree in (1, 2)
    },
    **{f"FEM DG P{degree} + SIPG (σ = 10)": FiniteElement(element=DiscontinuousLagrange(degree), diffusion=SymmetricInteriorPenalty()) for degree in (1, 2, 3, 4)},
    "FEM DG P2 + SIPG (σ = 50)": FiniteElement(element=DiscontinuousLagrange(2), diffusion=SymmetricInteriorPenalty(penalty=50.0)),
}
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
    "adaptive BDF": AdaptiveBDF(relative_tolerance=1e-6, absolute_tolerance=1e-6),
}
newton = Newton(linear_solver=SparseDirect(), jacobian="analytic")
linear_solvers = {
    "sparse LU": SparseDirect(),
    "conjugate gradients": ConjugateGradient(),
    "CG + algebraic multigrid": ConjugateGradient(preconditioner=AlgebraicMultigrid()),
    "GMRES": GMRES(),
    "GMRES + ILU": GMRES(preconditioner=ILU()),
    "GMRES + algebraic multigrid": GMRES(preconditioner=AlgebraicMultigrid()),
    "right GMRES": RightGMRES(),
    "flexible GMRES": FlexibleGMRES(),
    "automatic": AutomaticLinearSolver(),
}
nonlinear_solvers = {
    **{
        (name, linear_name): kind(linear_solver=linear, jacobian="analytic")
        for name, kind in (("Newton", Newton), ("damped Newton", DampedNewton), ("trust-region Newton", TrustRegionNewton), ("quasi-Newton", QuasiNewton))
        for linear_name, linear in linear_solvers.items()
    },
    **{("JFNK", linear_name): JFNK(linear_solver=linear) for linear_name, linear in linear_solvers.items()},
    ("fixed point", "none"): FixedPoint(),
    ("fixed point + Aitken", "none"): FixedPoint(acceleration="aitken"),
}

# ## Schemes that must fail, and their stabilized counterparts
#
# The heating starts with a jump at the surface and the diffusion operator is stiff. Leapfrog
# has no damping and is unstable on it at every step; its counterpart adds the
# Robert-Asselin-Williams filter. Adams-Moulton 3 and 4 have bounded stability intervals on the
# negative real axis ($[-6,0]$ and $[-3,0]$). A fixed-point (Picard) iteration contracts only for
# steps below the inverse spectral radius. The counterpart of each is the same method at a step it is
# stable for. The finite-element spectra are larger: quadratic continuous elements would need steps
# of a few $10^{-2}$ s and discontinuous elements a few $10^{-3}$ s for Adams-Moulton 3 and 4 (tens of
# thousands of steps); those cells stay marked as failed. The fixed-step Gear BDF methods of order 3
# to 6 start their history with extrapolated backward-Euler steps
# (`TimeMethod("bdf<k>_gear", startup_policy="implicit_extrapolation")`) and run at
# $\Delta t=10$ s on every spatial family.

implicit_step = 10.0
stabilized_time = {
    "leapfrog": (
        "leapfrog + Robert-Asselin-Williams filter (ν = 0.2, α = 0.53; first order), safety factor 0.15",
        TimeMethod("leapfrog_raw"),
        0.15,
    ),
}
stable_step = {
    "Adams-Moulton 3": 0.25,
    "Adams-Moulton 4": 0.25,
    "fixed point": 0.1,
    "fixed point + Aitken": 0.1,
}

# Unpreconditioned GMRES variants stall on the stiff stationary diffusion system within their
# default 1000 iterations; their counterpart adds an incomplete-LU preconditioner.

preconditioned = {
    "GMRES": GMRES(preconditioner=ILU()),
    "right GMRES": RightGMRES(preconditioner=ILU()),
    "flexible GMRES": FlexibleGMRES(preconditioner=ILU()),
}

# ## The test matrix
#
# 1. **Space × mesh** — every spatial scheme, stationary solve with Newton on 50 to 400 cells.
# 2. **Time integrator × space** — every integrator from body temperature to $t=5$ min on 100
#    cells. Explicit integrators run at their stable step (safety factor 0.5) on the finite
#    differences and finite volumes; implicit integrators with $\Delta t=10$ s on all four
#    families.
# 3. **Nonlinear × linear solver** — the stationary problem on 400 cells with quadratic Nitsche
#    elements, solved by Newton, damped, trust-region and quasi-Newton and by Jacobian-free
#    Newton-Krylov with every linear solver and by each linear solver alone, graded against the
#    steady state. The Pennes operator with symmetric Nitsche enforcement is symmetric positive
#    definite, so conjugate gradients and algebraic multigrid apply. The fixed-point (Picard)
#    iteration is a stage solver of implicit steps: it runs backward Euler on the central finite
#    differences, 100 cells, up to $t=5$ min, graded against the series.

time_spatial = {
    "FDM central Laplacian": spatial["FDM central Laplacian"],
    "FVM two-point diffusion": spatial["FVM two-point diffusion"],
    "FEM CG P2": spatial["FEM CG P2 + symmetric Nitsche"],
    "FEM DG P2": spatial["FEM DG P2 + SIPG (σ = 10)"],
}
explicit_columns = ("FDM central Laplacian", "FVM two-point diffusion")
stationary_solvers = ("Newton", "damped Newton", "trust-region Newton", "quasi-Newton", "JFNK")
runs = [
    dict(matrix="space x mesh", row=row, column=f"{cells} cells", cells=cells, spatial=scheme, study="stationary", nonlinear=newton)
    for row, scheme in spatial.items()
    for cells in cell_counts
]
runs += [
    dict(
        matrix="time integrator x space",
        row=name,
        column=space,
        cells=time_cells,
        spatial=time_spatial[space],
        study="transient",
        time=integrator,
        nonlinear=newton if getattr(integrator, "implicit", False) and "rosenbrock" not in getattr(integrator, "method_id", "") else None,
        linear=SparseDirect() if "rosenbrock" in getattr(integrator, "method_id", "") or isinstance(time_spatial[space], FiniteElement) else None,
        step=implicit_step if getattr(integrator, "implicit", False) else moment,
    )
    for name, integrator in integrators.items()
    for space in (time_spatial if getattr(integrator, "implicit", False) else explicit_columns)
]
runs += [
    dict(
        matrix="nonlinear x linear solver",
        row=row,
        column=column,
        cells=cell_counts[-1] if row in stationary_solvers else time_cells,
        spatial=time_spatial["FEM CG P2"] if row in stationary_solvers else time_spatial["FDM central Laplacian"],
        study="stationary" if row in stationary_solvers else "transient",
        time=None if row in stationary_solvers else BackwardEuler(),
        nonlinear=solver,
        step=implicit_step,
    )
    for (row, column), solver in nonlinear_solvers.items()
]
runs += [
    dict(
        matrix="nonlinear x linear solver",
        row="stationary, linear solve only",
        column=name,
        cells=cell_counts[-1],
        spatial=time_spatial["FEM CG P2"],
        study="stationary",
        nonlinear=None,
        linear=solver,
    )
    for name, solver in linear_solvers.items()
]
runs += [
    {**run, "variant": "stabilized", "time": stabilized_time[run["row"]][1], "safety": stabilized_time[run["row"]][2], "note_stabilized": stabilized_time[run["row"]][0]}
    for run in list(runs)
    if run["matrix"] == "time integrator x space" and run["row"] in stabilized_time
]
runs += [
    {**run, "variant": "stabilized", "step": stable_step[run["row"]], "note_stabilized": f"{run['row']} at Δt = {stable_step[run['row']]:g} s"}
    for run in list(runs)
    if run["matrix"] in ("time integrator x space", "nonlinear x linear solver") and run["row"] in stable_step and run.get("variant") is None
]
runs += [
    {
        **run,
        "variant": "stabilized",
        "linear": preconditioned[run["column"]] if run["nonlinear"] is None else None,
        "nonlinear": None if run["nonlinear"] is None else replace(run["nonlinear"], linear_solver=preconditioned[run["column"]]),
        "note_stabilized": f"{run['column']} + ILU preconditioner",
    }
    for run in list(runs)
    if run["matrix"] == "nonlinear x linear solver" and run["column"] in preconditioned and run.get("variant") is None
]
print(f"{len(runs)} runs: " + ", ".join(f"{sum(r['matrix'] == k for r in runs)} {k}" for k in dict.fromkeys(r["matrix"] for r in runs)))

# ## Run everything
#
# One loop over the same problem object. A refused combination is recorded with its validation
# message, a diverging one as failed.

for run in runs:
    simulation = Simulation(
        problem=heating,
        mesh=meshes[run["cells"]],
        numerics=Numerics(
            spatial=run["spatial"],
            linear_solver=run.get("linear"),
            time_integrator=run.get("time"),
            nonlinear_solver=run["nonlinear"],
            step_limits=TimeStepLimits(stability_safety_factor=run.get("safety", 0.5)),
        ),
    )
    started = time.perf_counter()
    check = simulation.validate()
    run.update(status="rejected", note="", error=np.nan, steps=0)
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = (
                simulation.solve_stationary()
                if run["study"] == "stationary"
                else simulation.run(start_time=0.0, end_time=moment, time_step=run["step"])
            )
            points = np.column_stack((centres[run["cells"]], np.zeros((run["cells"], 2))))
            run["profile"] = result.field(tissue.temperature).evaluate(points).reshape(-1)
            reference = steady if run["study"] == "stationary" else transient
            run["error"] = float(np.abs(run["profile"] - reference[run["cells"]]).max())
            run["steps"] = 0 if run["study"] == "stationary" else len(result.times) - 1
            run["status"] = "ran" if np.isfinite(run["profile"]).all() and run["error"] < surface - blood else "failed"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = "failed" if any(word in message for word in ("finite", "nonnegative", "did not converge", "maximum iteration", "unstable", "no stable")) else "rejected"
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    print(
        f"{run['matrix'][:12]:12s} {run['row'][:38]:38s} {run['column'][:24]:24s} {run.get('variant') or '':10s} "
        f"{run['status']:8s} {run['seconds']:7.2f} s  " + (f"max |T - T_exact| {run['error']:.2e} K" if run["status"] == "ran" else run["note"][:90])
    )

main = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") is None}
substitute = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") == "stabilized"}
hatch = {key for key, r in main.items() if r["status"] != "ran" and substitute.get(key, {}).get("status") == "ran"}
cell = {key: substitute[key] if key in hatch else r for key, r in main.items()}

# ## Space × mesh: the steady state
#
# The largest deviation from the $\sinh$ profile for every scheme on four meshes; reading along a
# row gives the observed order of accuracy.

space_key = "space x mesh"
rows, columns = list(spatial), [f"{n} cells" for n in cell_counts]
fig = plots.figure(1, 1, size=(13, 9))
plots.matrix(
    [[cell[(space_key, row, c)]["error"] for c in columns] for row in rows],
    rows,
    columns,
    scale="log",
    label="max |T - T_exact| [K], stationary",
    text=[[f"{cell[(space_key, row, c)]['error']:.1e}" if cell[(space_key, row, c)]["status"] == "ran" else cell[(space_key, row, c)]["status"] for c in columns] for row in rows],
    title="Space x mesh, stationary Newton solve",
    ax=fig[0],
)
plots.show()

print(f"{'scheme':40s} " + " ".join(f"{'order ' + str(a) + '-' + str(b):>13s}" for a, b in zip(cell_counts[:-1], cell_counts[1:])))
for row in rows:
    errors = [cell[(space_key, row, c)]["error"] for c in columns]
    print(f"{row:40s} " + " ".join(f"{np.log2(e0 / e1):13.2f}" for e0, e1 in zip(errors[:-1], errors[1:])))

# ## Profiles
#
# The steady state and the temperature after five minutes of heating for the four families on
# 100 cells, against the closed form.

fig = plots.figure(1, 2, size=(16, 6))
plots.line(
    y={
        row: (1e3 * centres[100], cell[(space_key, row, "100 cells")]["profile"] - 273.15)
        for row in ("FDM central Laplacian", "FVM two-point diffusion", "FEM CG P2 + symmetric Nitsche", "FEM DG P2 + SIPG (σ = 10)")
    },
    reference={"steady state": (1e3 * fine, fine_steady - 273.15)},
    xlabel="depth below the skin x [mm]",
    ylabel="tissue temperature [°C]",
    title="Steady state (dashed: closed form)",
    legend="upper right",
    ax=fig[0],
)
plots.line(
    y={space: (1e3 * centres[time_cells], cell[("time integrator x space", "BDF2 (variable step)", space)]["profile"] - 273.15) for space in time_spatial},
    reference={"series, t = 5 min": (1e3 * fine, fine_transient - 273.15)},
    xlabel="depth below the skin x [mm]",
    ylabel="tissue temperature [°C]",
    title="After 5 min of heating, BDF2 (dashed: series)",
    legend="upper right",
    ax=fig[1],
)
plots.show()

# ## Time integrator × space
#
# Explicit integrators at their stable step on finite differences and finite volumes, implicit
# integrators at $\Delta t=10$ s on all four families. A dash marks a combination outside this cut
# (explicit steps on finite elements, whose spectrum has no cell-width bound).

time_key = "time integrator x space"
timing = [r for r in runs if r["matrix"] == time_key and r.get("variant") is None]
time_rows, time_columns = list(dict.fromkeys(r["row"] for r in timing)), list(time_spatial)
time_hatched = [[(time_key, row, c) in hatch for c in time_columns] for row in time_rows]
time_notes = (
    ["Hatched: the declared integrator is unstable at this step (not A-stable, or undamped); colour and value show its stabilized counterpart:"]
    + [f"  {row}: same method at Δt = {stable_step[row]:g} s instead of {implicit_step:g} s." for row in stable_step if any((time_key, row, c) in hatch for c in time_columns)]
    + [f"  {row}: {description}." for row, (description, _, _) in stabilized_time.items() if any((time_key, row, c) in hatch for c in time_columns)]
)
fig = plots.figure(1, 2, size=(18, 11))
plots.matrix(
    [[cell[(time_key, row, c)]["error"] if (time_key, row, c) in cell else np.nan for c in time_columns] for row in time_rows],
    time_rows,
    time_columns,
    scale="log",
    label="max |T - T_series| at t = 5 min [K]",
    text=[
        [
            (f"{cell[(time_key, row, c)]['error']:.1e}" if cell[(time_key, row, c)]["status"] == "ran" else cell[(time_key, row, c)]["status"])
            if (time_key, row, c) in cell
            else "–"
            for c in time_columns
        ]
        for row in time_rows
    ],
    hatched=time_hatched,
    notes=time_notes,
    title=f"Error at t = 5 min, {time_cells} cells",
    ax=fig[0],
)
plots.matrix(
    [[cell[(time_key, row, c)]["seconds"] if (time_key, row, c) in cell and cell[(time_key, row, c)]["status"] == "ran" else np.nan for c in time_columns] for row in time_rows],
    time_rows,
    time_columns,
    scale="log",
    label="wall time [s]",
    text=[
        [
            (f"{cell[(time_key, row, c)]['seconds']:.1f} s, {cell[(time_key, row, c)]['steps']} steps" if cell[(time_key, row, c)]["status"] == "ran" else cell[(time_key, row, c)]["status"])
            if (time_key, row, c) in cell
            else "–"
            for c in time_columns
        ]
        for row in time_rows
    ],
    hatched=time_hatched,
    title="Wall time and accepted steps",
    ax=fig[1],
)
plots.show()

# ## Nonlinear × linear solver
#
# The stationary problem on quadratic elements: the error against the steady state must not
# depend on the solver pair beyond the iterative tolerances; the wall time does. Jacobian-free
# Newton-Krylov with an unpreconditioned Krylov solver stops at its nonlinear tolerance on the
# ill-conditioned element system, so its error is that tolerance, not the discretization error.
# The fixed-point rows run backward Euler on finite differences and are graded against the
# series; the last row solves the stationary problem with each linear solver alone.

solver_key = "nonlinear x linear solver"
solving = [r for r in runs if r["matrix"] == solver_key and r.get("variant") is None]
solver_rows, solver_columns = list(dict.fromkeys(r["row"] for r in solving)), [*linear_solvers, "none"]
fig = plots.figure(1, 1, size=(15, 7))
plots.matrix(
    [[cell[(solver_key, row, c)]["seconds"] if (solver_key, row, c) in cell and cell[(solver_key, row, c)]["status"] == "ran" else np.nan for c in solver_columns] for row in solver_rows],
    solver_rows,
    solver_columns,
    scale="log",
    label="wall time [s]",
    text=[
        [
            (f"{cell[(solver_key, row, c)]['seconds']:.1f} s\n{cell[(solver_key, row, c)]['error']:.1e} K" if cell[(solver_key, row, c)]["status"] == "ran" else cell[(solver_key, row, c)]["status"])
            if (solver_key, row, c) in cell
            else ""
            for c in solver_columns
        ]
        for row in solver_rows
    ],
    hatched=[[(solver_key, row, c) in hatch for c in solver_columns] for row in solver_rows],
    notes=["Hatched: the declared pair does not converge; colour and value show its stabilized counterpart:"]
    + [f"  {row}: Δt = {stable_step[row]:g} s instead of {implicit_step:g} s." for row in ("fixed point", "fixed point + Aitken") if any((solver_key, row, c) in hatch for c in solver_columns)]
    + [f"  {column}: + incomplete-LU preconditioner." for column in preconditioned if any((solver_key, row, column) in hatch for row in solver_rows)],
    title="Wall time and error per solver pair",
    ax=fig[0],
)
plots.show()

# ## What the library refused, and what failed

for status in ("rejected", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(f"  {r['matrix'][:12]:12s} {r['row'][:38]:38s} {r['column'][:24]:24s} {r.get('variant') or '':10s} {r['note'][:100]}")
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.0f} min in total"
)

# ## Methods that do not apply to the Pennes equation
#
# The equation has no convection: interface fluxes, reconstructions, limiters and SUPG-type
# stabilizations have nothing to act on. Discontinuous elements of degree 0 carry no gradient for
# interior-penalty diffusion. Staggered Verlet needs a second-order system. Continuous interior
# penalties (CIP, volume interior penalty) stabilize convection and are left out for the same
# reason as SUPG.
#
# ## Going further
#
# Add the metabolic heat as a source and the equilibrium shifts to $T_b+Q_m/(\omega_b\rho_bc_b)$;
# a time-dependent surface temperature gives the periodic skin-heating problem. The matrix code
# stays as it is.
