"""OneSimulation example: a stiff RC ladder network through every time integrator and solver OneSimulation offers.
Requires the installed, activated OneSimulation runtime.

A resistor-capacitor ladder charged through a voltage step is declared once as a circuit and advanced
with every time integrator, every implicit integrator with every nonlinear solver, and every pair of
nonlinear and linear solvers. The number of stages sets the stiffness; explicit methods that cannot
be stable at the common step are shown with their stable-step counterpart. Every run is graded
against the exact modal solution of the ladder.
"""

# # A stiff RC ladder: one circuit, every time integrator and solver
#
# A voltage source of $U=1\,\mathrm{V}$ is switched onto a ladder of $N$ equal stages: a series
# resistor $R=R_\mathrm{tot}/N$ followed by a capacitor $C=C_\mathrm{tot}/N$ to ground. The far end is
# open. With the capacitor voltages $v_1,\dots,v_N$ and $v_0=U$, Kirchhoff's current law gives
#
# $$RC\,\dot v_i=v_{i-1}-2v_i+v_{i+1}\quad(i<N),\qquad RC\,\dot v_N=v_{N-1}-v_N .$$
#
# For growing $N$ the ladder approaches a diffusion line (an RC transmission line): the slowest time
# constant stays near $4R_\mathrm{tot}C_\mathrm{tot}/\pi^2$ while the fastest shrinks like
# $R_\mathrm{tot}C_\mathrm{tot}/(4N^2)$. The ratio of the two is the stiffness that explicit methods
# must resolve and implicit methods may step over.
#
# **Reference.** The matrix is symmetric and tridiagonal; its eigenpairs are known in closed form.
# With $\vartheta_k=(2k-1)\pi/(2N+1)$ the decay rates are
# $\lambda_k=(2-2\cos\vartheta_k)/(RC)$ and the modes $\phi_k(i)=\sin(i\vartheta_k)$, so
#
# $$v_N(t)=U\Big(1-\sum_{k=1}^{N}c_k\sin(N\vartheta_k)\,e^{-\lambda_k t}\Big),\qquad
# c_k=\frac{\sum_i\sin(i\vartheta_k)}{\sum_i\sin^2(i\vartheta_k)}$$
#
# (e.g. G. Strang, *Introduction to Applied Mathematics*, §1.5 on second-difference matrices).
#
# **Grading.** The far-end voltage $v_N$ is compared with the exact value at every accepted step; the
# error is the largest deviation over the run in volts. By the maximum principle every exact voltage
# stays between 0 and 1 V; a run that leaves $[-0.5, 1.5]\,\mathrm{V}$ has become unstable and is
# recorded as failed.
#
# **Resolution.** A circuit has no spatial mesh: `LumpedMesh` carries all unknowns on one global
# node; the ladders below have 4, 16 and 32 capacitor states. The resolution is the time step; a
# step check at half the step closes the notebook.

import time

import numpy as np
from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
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
    TimeMethod,
)
from onesim.physics.systems import (
    Circuit,
    CircuitCapacitor,
    CircuitResistor,
    CircuitVoltageSource,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# $R_\mathrm{tot}=1\,\mathrm{k\Omega}$ and $C_\mathrm{tot}=1\,\mathrm{mF}$, so
# $R_\mathrm{tot}C_\mathrm{tot}=1\,\mathrm{s}$. The same circuit declaration with 4, 16 and 32 stages
# forms the columns of the integrator matrix. `Circuit.as_model` turns each network into a lumped
# model whose consistent initial state (all capacitors empty, source on) comes from
# `initial_values`.

supply, total_resistance, total_capacitance, end_time, step = 1.0, 1e3, 1e-3, 1.0, 1e-2
sizes = (4, 16, 32)
domain = LumpedDomain()
ladders = {
    n: Circuit(
        components=(CircuitVoltageSource(name="U", positive="n0", negative="0", voltage=supply),)
        + tuple(
            element
            for i in range(n)
            for element in (
                CircuitResistor(
                    name=f"R{i}", positive=f"n{i}", negative=f"n{i + 1}", resistance=total_resistance / n
                ),
                CircuitCapacitor(
                    name=f"C{i}",
                    positive=f"n{i + 1}",
                    negative="0",
                    capacitance=total_capacitance / n,
                    initial_voltage=0.0,
                ),
            )
        )
    ).as_model(domain=domain.system)
    for n in sizes
}
problems = {
    n: Problem(
        name=f"RC ladder with {n} stages",
        geometry=domain,
        physics=(model,),
        initial_conditions=model.initial_values(),
    )
    for n, model in ladders.items()
}
mesh = LumpedMesh(domain)

# ## The exact modal solution
#
# Decay rates, mode shapes and modal coefficients of each ladder, straight from the formulas above.

angles = {n: (2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n + 1) for n in sizes}
rates = {
    n: (2 - 2 * np.cos(angles[n])) * n**2 / (total_resistance * total_capacitance)
    for n in sizes
}
shapes = {n: np.sin(np.outer(np.arange(1, n + 1), angles[n])) for n in sizes}
weights = {
    n: supply * shapes[n].sum(axis=0) / (shapes[n] ** 2).sum(axis=0) * shapes[n][-1]
    for n in sizes
}
fine = np.linspace(0.0, end_time, 2001)
exact_fine = {
    n: supply - (weights[n] * np.exp(-np.outer(fine, rates[n]))).sum(axis=1) for n in sizes
}
stable_step = {n: 0.5 / rates[n].max() for n in sizes}
for n in sizes:
    print(
        f"{n:3d} stages: slowest rate {rates[n].min():7.3f} 1/s, fastest {rates[n].max():9.1f} 1/s, "
        f"stiffness ratio {rates[n].max() / rates[n].min():7.0f}, λ_max·Δt = {rates[n].max() * step:6.1f}, "
        f"stable explicit step {stable_step[n]:.2e} s"
    )

# ## The method catalogue
#
# Every time integrator of the library that applies to a first-order network. Implicit stages take
# Newton's method with a sparse direct solve; Rosenbrock stages take the sparse direct solve alone.
# The nonlinear and linear solvers are varied on their own further below.

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
    "Dormand-Prince 5(4)": TimeMethod("dormand_prince54"),
    "leapfrog": TimeMethod("leapfrog"),
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
    **{f"BDF{k} (Gear)": TimeMethod(f"bdf{k}_gear") for k in (2, 3, 4, 5, 6)},
    "Rosenbrock-Euler": TimeMethod("rosenbrock_euler"),
    "Rosenbrock-Wanner Euler": TimeMethod("rosenbrock_wanner_euler"),
    "adaptive BDF": AdaptiveBDF(relative_tolerance=1e-6, absolute_tolerance=1e-9),
}
newton = Newton(linear_solver=SparseDirect())
linear_solvers = {
    "sparse LU": SparseDirect(),
    "GMRES": GMRES(),
    "GMRES + ILU": GMRES(preconditioner=ILU()),
    "right GMRES": RightGMRES(),
    "flexible GMRES": FlexibleGMRES(),
    "automatic": AutomaticLinearSolver(),
}
nonlinear_kinds = {
    "Newton": Newton,
    "damped Newton": DampedNewton,
    "trust-region Newton": TrustRegionNewton,
    "quasi-Newton": QuasiNewton,
}

# ## Schemes that must fail, and their stabilized counterparts
#
# An explicit method is stable only while $\lambda_\max\Delta t$ stays inside its stability interval
# on the negative real axis (2 for forward Euler and the second-order Runge-Kutta methods, about 2.8
# for RK4, 1 for Adams-Bashforth 2). At the common step $\Delta t=10^{-2}\,\mathrm{s}$ that holds for
# the 4-stage ladder but not for 16 or 32 stages. Adams-Moulton 3 and 4 have bounded stability
# intervals too, and a fixed-point (Picard) iteration contracts only while $\lambda_\max\Delta t<1$.
# Leapfrog is unstable on any decaying mode. A failed cell is hatched; its colour and value are those
# of the counterpart: the same method at the stable step $\Delta t_s=0.5/\lambda_\max$; for leapfrog
# the Robert-Asselin-Williams filtered leapfrog, which is stable only for
# $\lambda\Delta t<0.395$, at $0.3/\lambda_\max$; for the fixed-point iteration a step at which it
# contracts by a factor of ten per iteration, $0.1/\lambda_\max$.

stabilized_time = {
    "leapfrog": ("leapfrog + Robert-Asselin-Williams filter", TimeMethod("leapfrog_raw")),
}
step_factor = {"leapfrog": 0.6, "leapfrog + RAW filter": 0.6, "fixed point": 0.1}

# ## The test matrix
#
# 1. **Time integrator × ladder size** — every integrator on 4, 16 and 32 stages at
#    $\Delta t=10^{-2}\,\mathrm{s}$.
# 2. **Implicit integrator × nonlinear solver** — 16 stages, sparse direct linear solves.
# 3. **Nonlinear × linear solver** — backward Euler on 32 stages, plus the Rosenbrock-Euler step, which
#    needs only a linear solver.
# 4. **Step check** — representative implicit integrators on 16 stages at half the step.

implicit_names = [
    name for name, method in integrators.items()
    if method.implicit and "rosenbrock" not in method.method_id
]
runs = [
    dict(
        matrix="time integrator x ladder",
        row=name,
        column=f"{n} stages",
        stages=n,
        time=method,
        nonlinear=newton if name in implicit_names else None,
        linear=SparseDirect() if "rosenbrock" in method.method_id else None,
        step=step,
    )
    for name, method in integrators.items()
    for n in sizes
]
runs += [
    dict(
        matrix="integrator x nonlinear solver",
        row=name,
        column=solver_name,
        stages=16,
        time=integrators[name],
        nonlinear=solver,
        linear=None,
        step=step,
    )
    for name in implicit_names
    for solver_name, solver in {
        **{key: kind(linear_solver=SparseDirect()) for key, kind in nonlinear_kinds.items()},
        "JFNK": JFNK(linear_solver=GMRES()),
        "fixed point": FixedPoint(),
    }.items()
]
runs += [
    dict(
        matrix="nonlinear x linear solver",
        row=name,
        column=linear_name,
        stages=32,
        time=BackwardEuler(),
        nonlinear=(JFNK if name == "JFNK" else nonlinear_kinds[name])(linear_solver=linear),
        linear=None,
        step=step,
    )
    for name in (*nonlinear_kinds, "JFNK")
    for linear_name, linear in linear_solvers.items()
    if name != "JFNK" or linear_name not in ("sparse LU", "automatic")
]
runs += [
    dict(
        matrix="nonlinear x linear solver",
        row="Rosenbrock-Euler (no nonlinear solver)",
        column=linear_name,
        stages=32,
        time=TimeMethod("rosenbrock_euler"),
        nonlinear=None,
        linear=linear,
        step=step,
    )
    for linear_name, linear in linear_solvers.items()
]
runs += [
    {**run, "matrix": "step check", "step": step / 2}
    for run in runs
    if run["matrix"] == "time integrator x ladder"
    and run["stages"] == 16
    and run["row"] in ("backward Euler", "Crank-Nicolson", "implicit midpoint", "SDIRK2", "BDF2 (Gear)", "BDF3 (Gear)")
]
print(
    f"{len(runs)} runs before counterparts: "
    + ", ".join(f"{sum(r['matrix'] == m for r in runs)} {m}" for m in dict.fromkeys(r["matrix"] for r in runs))
)

# ## Run everything
#
# One loop, the same problem objects every time. A refused combination is recorded with the library's
# message; a run that becomes unstable is recorded as failed, and the loop appends its stabilized
# counterpart, which runs in the same loop.

for run in runs:
    model = ladders[run["stages"]]
    simulation = Simulation(
        problem=problems[run["stages"]],
        mesh=mesh,
        numerics=Numerics(
            spatial=LumpedParameters(),
            time_integrator=run["time"],
            nonlinear_solver=run["nonlinear"],
            linear_solver=run["linear"],
        ),
    )
    started = time.perf_counter()
    check = simulation.validate()
    run.update(status="rejected", note="", error=np.nan, steps=0)
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = simulation.run(start_time=0.0, end_time=end_time, time_step=run["step"])
            times, far = result.field(model.terminal_voltage(f"C{run['stages'] - 1}")).time_series()
            exact = supply - (weights[run["stages"]] * np.exp(-np.outer(times, rates[run["stages"]]))).sum(axis=1)
            run["error"] = np.abs(far - exact).max()
            run["series"] = (times, far)
            run["steps"] = len(times) - 1
            run["status"] = (
                "ran" if np.isfinite(far).all() and np.abs(far - 0.5).max() <= 1.0 else "failed"
            )
            run["note"] = "" if run["status"] == "ran" else "voltage leaves [-0.5, 1.5] V: unstable"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "failed"
                if "unstable" in message
                or "finite" in message
                or "did not converge" in message
                or "converge" in message
                else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    if run["status"] == "failed" and run.get("variant") is None and run["matrix"] != "step check":
        replacement = stabilized_time.get(run["row"], (None, run["time"]))
        safe = stable_step[run["stages"]] * step_factor.get(run["row"], step_factor.get(run["column"], 1.0))
        runs.append(
            {
                **run,
                "variant": "stabilized",
                "time": replacement[1],
                "step": safe,
                "note_stabilized": (replacement[0] or run["row"]) + f" at Δt = {safe:.1e} s",
            }
        )
    print(
        f"{run['matrix'][:12]:12s} {run['row'][:28]:28s} {run['column'][:20]:20s} "
        f"{run.get('variant') or '':10s} {run['status']:8s} {run['seconds']:6.1f} s  "
        + (f"error {run['error']:.2e} V  steps {run['steps']}" if run["status"] == "ran" else run["note"][:80])
    )

# Every matrix below shows the declared combination. Where it failed and the counterpart ran, the
# cell is hatched and shows the counterpart; the legend names it.

main = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") is None}
substitute = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") == "stabilized"}
hatch = {
    key
    for key, r in main.items()
    if r["status"] != "ran" and substitute.get(key, {}).get("status") == "ran"
}
cell = {key: substitute[key] if key in hatch else r for key, r in main.items()}

# ## Time integrator × ladder size
#
# The largest far-end voltage error over one second, and the wall time with the number of steps.

key = "time integrator x ladder"
rows, columns = list(integrators), [f"{n} stages" for n in sizes]
fig = plots.figure(1, 2, size=(17, 12))
for panel, measure, label, title in (
    (fig[0], "error", "max |v_N − v_N,exact| [V]", "Far-end voltage error, Δt = 10⁻² s"),
    (fig[1], "seconds", "wall time [s]", "Wall time and accepted steps"),
):
    plots.matrix(
        [
            [
                cell[(key, row, c)][measure] if cell[(key, row, c)]["status"] == "ran" else np.nan
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
                        f"{cell[(key, row, c)]['error']:.2e}"
                        if measure == "error"
                        else f"{cell[(key, row, c)]['seconds']:.1f} s, {cell[(key, row, c)]['steps']}"
                    )
                    if cell[(key, row, c)]["status"] == "ran"
                    else cell[(key, row, c)]["status"]
                )
                for c in columns
            ]
            for row in rows
        ],
        hatched=[[(key, row, c) in hatch for c in columns] for row in rows],
        notes=(
            ["Hatched: the declared integrator is unstable at Δt = 10⁻² s; colour and value show its counterpart:"]
            + sorted({f"  {k[1]}, {k[2]}: {substitute[k]['note_stabilized']}." for k in hatch if k[0] == key})
            if measure == "error"
            else None
        ),
        title=title,
        ax=panel,
    )
plots.show()

# ## Far-end voltage
#
# The 32-stage ladder with implicit integrators at $\Delta t=10^{-2}\,\mathrm{s}$ against the modal
# solution. Crank-Nicolson and implicit midpoint are A-stable but not L-stable: their fastest modes
# are not damped, which shows as a small ripple at the start.

fig = plots.figure(1, 1, size=(10, 5))
plots.line(
    y={
        name: main[(key, name, "32 stages")]["series"]
        for name in ("backward Euler", "Crank-Nicolson", "implicit midpoint", "SDIRK2", "BDF2 (Gear)", "Rosenbrock-Euler")
        if main[(key, name, "32 stages")]["status"] == "ran"
    },
    reference={"exact modal solution": (fine, exact_fine[32])},
    xlabel="time [s]",
    ylabel="far-end voltage v_N [V]",
    title="32-stage ladder, Δt = 10⁻² s",
    legend="lower right",
    ax=fig[0],
)
plots.show()

# ## Implicit integrator × nonlinear solver
#
# The network is linear, so every converging nonlinear solver gives the same discrete solution; the
# cost differs. A fixed-point iteration cannot contract at $\lambda_\max\Delta t\approx 10$.

for key, title, label in (
    ("integrator x nonlinear solver", "16 stages: far-end error per integrator and nonlinear solver", "max error [V]"),
    ("nonlinear x linear solver", "Backward Euler, 32 stages: wall time and error per solver pair", "wall time [s]"),
):
    selected = [r for r in runs if r["matrix"] == key and r.get("variant") is None]
    matrix_rows = list(dict.fromkeys(r["row"] for r in selected))
    matrix_columns = list(dict.fromkeys(r["column"] for r in selected))
    measure = "error" if key == "integrator x nonlinear solver" else "seconds"
    fig = plots.figure(1, 1, size=(13, 9 if measure == "error" else 5))
    plots.matrix(
        [
            [
                cell[(key, row, c)][measure]
                if (key, row, c) in cell and cell[(key, row, c)]["status"] == "ran"
                else np.nan
                for c in matrix_columns
            ]
            for row in matrix_rows
        ],
        matrix_rows,
        matrix_columns,
        scale="log",
        label=label,
        text=[
            [
                (
                    (
                        f"{cell[(key, row, c)]['error']:.2e}"
                        if measure == "error"
                        else f"{cell[(key, row, c)]['seconds']:.1f} s\n{cell[(key, row, c)]['error']:.1e}"
                    )
                    if cell[(key, row, c)]["status"] == "ran"
                    else cell[(key, row, c)]["status"]
                )
                if (key, row, c) in cell
                else ""
                for c in matrix_columns
            ]
            for row in matrix_rows
        ],
        hatched=[[(key, row, c) in hatch for c in matrix_columns] for row in matrix_rows],
        notes=(
            ["Hatched: the iteration cannot converge at Δt = 10⁻² s; colour and value show:"]
            + sorted({f"  {k[1]}, {k[2]}: {substitute[k]['note_stabilized']}." for k in hatch if k[0] == key})
            if any(k[0] == key for k in hatch)
            else None
        ),
        title=title,
        ax=fig[0],
    )
    plots.show()

# ## Step check: Δt against Δt/2
#
# Implicit integrators on 16 stages at half the step. The error ratio shows the observed order: about
# 4 for a second-order method, 2 for a first-order one. Crank-Nicolson and implicit midpoint can show
# less while the undamped fast modes dominate the error.

print(f"{'integrator':26s} {'error Δt':>11s} {'error Δt/2':>11s} {'ratio':>7s} {'order':>6s}")
for r in (r for r in runs if r["matrix"] == "step check"):
    full = main[("time integrator x ladder", r["row"], "16 stages")]
    if r["status"] == "ran" and full["status"] == "ran":
        print(
            f"{r['row']:26s} {full['error']:11.3e} {r['error']:11.3e} {full['error'] / r['error']:7.2f} "
            f"{np.log2(full['error'] / r['error']):6.2f}"
        )

# ## What the library refused, and what failed

for status in ("rejected", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(f"  {r['matrix'][:12]:12s} {r['row'][:28]:28s} {r['column'][:20]:20s} {r['note'][:90]}")
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(runs)} runs ({sum(r.get('variant') == 'stabilized' for r in runs)} counterparts), "
    + f"{sum(r['seconds'] for r in runs) / 60:.1f} min in total"
)

# ## Methods that do not apply to an RC network
#
# The staggered (velocity) Verlet and symplectic Euler steps need a second-order system, positions
# whose rate is a velocity; capacitor charges are first-order storage states, and the library refuses
# both with that reason. Conjugate gradients and algebraic multigrid need a declared symmetric positive
# definite stage matrix; the reduced nodal storage operator $-C^{-1}G$ of a general network is only
# similar to a symmetric one, so the circuit model declares no such property and the library refuses CG
# with that reason. JFNK never forms the Jacobian, so it has no matrix for a sparse direct (or
# automatic, which selects the direct) inner solve; it is paired with the Krylov solvers only. A
# circuit has no spatial discretization, so there is no space axis and no mesh
# refinement; the step check takes its place. Runge-Kutta-Chebyshev and -Legendre super-steps and IMEX
# splits are offered for spatial operators that publish a parabolic spectrum or a stiff/non-stiff
# split, which the circuit model does not.
#
# ## Going further
#
# Add an inductor to each stage for an LC transmission line, or a diode at the far end for a
# nonlinear load; the matrix code stays as it is.
