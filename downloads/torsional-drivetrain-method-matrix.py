"""OneSimulation example: a torsional drivetrain and a pendulum through every time integrator OneSimulation offers.
Requires the installed, activated OneSimulation runtime.

A chain of eight flywheels coupled by elastic shafts is declared once and advanced with every
applicable time integrator, including the staggered (velocity) Verlet and symplectic Euler steps for
second-order mechanics, for an undamped, a damped and a stiff variant. Each run is graded against the
exact modal solution. A large-amplitude pendulum is graded against its elliptic-integral period, and
its implicit steps are solved with every pair of nonlinear and linear solvers.
"""

# # A torsional drivetrain: one model, every time integrator
#
# Eight flywheels with moments of inertia $J_i$ sit on a shaft line; neighbouring wheels are coupled
# by shafts of torsional stiffness $k_i$ and damping $c_i$. With the twist $e_i=\theta_i-\theta_{i+1}$
# the shaft torque is $\tau_i=k_i e_i+c_i\dot e_i$ and every wheel obeys
#
# $$\dot\theta_i=\omega_i,\qquad J_i\dot\omega_i=\tau_{i-1}-\tau_i .$$
#
# In matrix form $M\ddot\theta+C\dot\theta+K\theta=0$ with $K=D^\top\operatorname{diag}(k)D$,
# $C=D^\top\operatorname{diag}(c)D$ and the difference matrix $D$. The chain is free at both ends, so
# it has one rigid-body mode and seven elastic modes.
#
# **Reference.** The system is linear, so the exact solution is its modal expansion: with the
# first-order matrix $A=\begin{pmatrix}0&I\\-M^{-1}K&-M^{-1}C\end{pmatrix}$ and its eigen
# decomposition $A=V\Lambda V^{-1}$, the state is $x(t)=V e^{\Lambda t}V^{-1}x_0$ (e.g. Meirovitch,
# *Fundamentals of Vibrations*, ch. 7). The eigenvalues are computed with NumPy below.
#
# **Grading.** The angle of the last wheel is compared with the modal solution at every accepted step;
# the error is the largest deviation over the run divided by the largest exact angle. The wall time
# covers validation, binding and time stepping.
#
# **Resolution.** A lumped model has no spatial mesh: `LumpedMesh` carries all unknowns on one global
# node, here 16 state values (8 angles, 8 angular velocities). The resolution is the time step; a step
# check at half the step closes the notebook.

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
    StaggeredVerlet,
    TimeMethod,
)
from onesim.physics.multibody import SimplePendulum, TorsionalDrivetrain
from onesim.results.sampling import crossing_times
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.special import ellipk

theme.set()

# ## The physics, declared once
#
# Eight equal wheels of $J=0.02\,\mathrm{kg\,m^2}$ on shafts of $k=4000\,\mathrm{N\,m/rad}$. Three
# variants of the same declaration form the columns of the matrix:
#
# - **undamped** — no shaft damping;
# - **damped** — stiffness-proportional shaft damping $c_i=\beta k_i$ with 2 % damping in the lowest
#   elastic mode;
# - **stiff coupling** — undamped, but the middle shaft is a thousand times stiffer (a rigid
#   coupling modelled elastically); its mode is about 25 times faster than the others.
#
# The line starts at rest with a linear twist of $\pm 5\,\mathrm{mrad}$, which excites every
# antisymmetric elastic mode and no rigid rotation.

wheels, inertia, stiffness = 8, 0.02, 4000.0
lowest = 2 * np.sqrt(stiffness / inertia) * np.sin(np.pi / (2 * wheels))
period, beta = 2 * np.pi / lowest, 2 * 0.02 / lowest
shafts = {
    "undamped": ((stiffness,) * (wheels - 1), (0.0,) * (wheels - 1)),
    "damped": ((stiffness,) * (wheels - 1), (beta * stiffness,) * (wheels - 1)),
    "stiff coupling": (
        tuple(stiffness * (1000.0 if i == wheels // 2 - 1 else 1.0) for i in range(wheels - 1)),
        (0.0,) * (wheels - 1),
    ),
}
twist = 0.01 * (np.arange(wheels) / (wheels - 1) - 0.5)
domain = LumpedDomain()
lines = {
    case: TorsionalDrivetrain(
        domain=domain.system,
        inertias=(inertia,) * wheels,
        shaft_stiffness=k,
        shaft_damping=c,
    )
    for case, (k, c) in shafts.items()
}
problems = {
    case: Problem(
        name=f"Torsional drivetrain, {case}",
        geometry=domain,
        physics=(line,),
        initial_conditions=tuple(
            InitialValue(line.angle(i), float(twist[i])) for i in range(wheels)
        )
        + tuple(InitialValue(line.angular_velocity(i), 0.0) for i in range(wheels)),
    )
    for case, line in lines.items()
}
mesh = LumpedMesh(domain)
end_time, step = 3 * period, period / 100
print(
    f"{wheels} wheels, {2 * wheels} state values on one global node; lowest elastic period "
    f"{1e3 * period:.2f} ms, end time {1e3 * end_time:.1f} ms, step {1e6 * step:.1f} µs"
)

# ## The exact modal solution
#
# The first-order system matrix of each variant and its eigen decomposition. The modal coefficients
# $V^{-1}x_0$ are fixed by the initial twist; the solution at any time is a sum of damped
# exponentials.

difference = np.eye(wheels - 1, wheels) - np.eye(wheels - 1, wheels, k=1)
system = {
    case: np.block(
        [
            [np.zeros((wheels, wheels)), np.eye(wheels)],
            [
                -difference.T @ np.diag(k) @ difference / inertia,
                -difference.T @ np.diag(c) @ difference / inertia,
            ],
        ]
    )
    for case, (k, c) in shafts.items()
}
eigen = {case: np.linalg.eig(matrix) for case, matrix in system.items()}
start = np.concatenate((twist, np.zeros(wheels)))
coefficients = {case: np.linalg.solve(vectors, start) for case, (_, vectors) in eigen.items()}
fastest = {case: np.abs(values).max() for case, (values, _) in eigen.items()}
fine = np.linspace(0.0, end_time, 3001)
exact_fine = {
    case: (
        eigen[case][1][wheels - 1]
        @ (coefficients[case][:, None] * np.exp(np.outer(eigen[case][0], fine)))
    ).real
    for case in shafts
}
amplitude = {case: np.abs(values).max() for case, values in exact_fine.items()}
for case in shafts:
    print(
        f"{case:15s} highest frequency {fastest[case]:9.1f} rad/s, ω_max·Δt = "
        f"{fastest[case] * step:6.2f}, lowest damping ratio "
        f"{-np.sort_complex(eigen[case][0])[-3].real / lowest:.3f}"
    )

# ## The method catalogue
#
# Every time integrator of the library: explicit one-step and multistep methods, the kick-drift
# methods for second-order systems (staggered velocity Verlet and symplectic Euler, which use the
# position/velocity structure $\dot\theta=\omega$ that `TorsionalDrivetrain` declares), implicit
# Runge-Kutta, Adams-Moulton and BDF methods, Rosenbrock methods and the adaptive integrators.
# Implicit stages take Newton's method with a sparse direct solve; Rosenbrock stages take the sparse
# direct solve alone.

integrators = {
    "forward Euler": ForwardEuler(),
    "symplectic Euler": TimeMethod("symplectic_euler"),
    "staggered Verlet": StaggeredVerlet(),
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

# ## Schemes that must fail, and their stabilized counterparts
#
# Forward Euler amplifies every undamped oscillation at any step ($|1+i\omega\Delta t|>1$); the library
# refuses it for the undamped variants. Its counterpart adds exactly one ingredient: the position
# update uses the new velocity (symplectic or Euler-Cromer step), at the stable step where needed.
# Leapfrog is unstable on the damping of the damped variant. On the stiff coupling no explicit method
# is stable at $\Delta t=T_1/100$ ($\omega_\max\Delta t\approx 7$), and Adams-Moulton 3 and 4 are not
# A-stable. Each such cell is hatched; its colour and value are those of the counterpart, which is
# the same method at the stable step $\Delta t_s=0.5/\omega_\max$, but at most a quarter of the
# declared step (methods such as Adams-Bashforth 2 amplify undamped oscillations at every step and
# only become accurate over the run at a smaller one; for Adams-Bashforth 2, whose growth per step is
# about $(\omega\Delta t)^4/4$, the step is $0.1/\omega_\max$), unless noted otherwise.

stabilized_time = {
    "forward Euler": (
        "symplectic Euler (position update with the new velocity)",
        TimeMethod("symplectic_euler"),
    ),
    "leapfrog": ("leapfrog + Robert-Asselin-Williams filter", TimeMethod("leapfrog_raw")),
}
stable_step = {case: 0.5 / rate for case, rate in fastest.items()}
step_factor = {"Adams-Bashforth 2": 0.2}

# ## The test matrix
#
# 1. **Time integrator × variant** — every integrator on the three drivetrains at
#    $\Delta t=T_1/100$.
# 2. **Pendulum period × amplitude** — further below.
# 3. **Nonlinear × linear solver** — implicit midpoint steps of the large-amplitude pendulum.
# 4. **Step check** — representative combinations at half the step.

runs = [
    dict(
        matrix="time integrator x variant",
        row=name,
        column=case,
        time=method,
        step=step,
    )
    for name, method in integrators.items()
    for case in shafts
]
runs += [
    {**run, "matrix": "step check", "step": step / 2}
    for run in runs
    if run["row"] in ("staggered Verlet", "symplectic Euler", "classical RK4", "implicit midpoint", "BDF2 (Gear)", "backward Euler")
    and run["column"] == "damped"
]
print(f"{len(runs)} drivetrain runs before counterparts")

# ## Run everything
#
# One loop, the same problem objects every time. A combination the library refuses is recorded with
# its message. A run whose amplitude grows beyond ten times the exact one, or that the library stops
# as unstable, is recorded as failed; for it the loop appends its stabilized counterpart, which runs
# in the same loop.

for run in runs:
    line = lines[run["column"]]
    simulation = Simulation(
        problem=problems[run["column"]],
        mesh=mesh,
        numerics=Numerics(
            spatial=LumpedParameters(),
            time_integrator=run["time"],
            nonlinear_solver=(
                newton
                if run["time"].implicit and "rosenbrock" not in run["time"].method_id
                else None
            ),
            linear_solver=SparseDirect() if "rosenbrock" in run["time"].method_id else None,
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
            times, angles = result.field(line.angle(wheels - 1)).time_series()
            exact = (
                eigen[run["column"]][1][wheels - 1]
                @ (coefficients[run["column"]][:, None] * np.exp(np.outer(eigen[run["column"]][0], times)))
            ).real
            run["error"] = np.abs(angles - exact).max() / amplitude[run["column"]]
            run["series"] = (times, angles)
            run["steps"] = len(times) - 1
            run["status"] = "ran" if np.isfinite(run["error"]) and run["error"] < 10 else "failed"
            run["note"] = "" if run["status"] == "ran" else "amplitude grows without bound"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "failed"
                if "unstable" in message or "finite" in message or "did not converge" in message
                else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    if run["status"] == "failed" and run.get("variant") is None and run["matrix"] != "step check":
        replacement = stabilized_time.get(run["row"], (None, run["time"]))
        safe = min(
            run["step"] / (1 if replacement[0] else 4),
            stable_step[run["column"]] * step_factor.get(run["row"], 1.0),
        )
        runs.append(
            {
                **run,
                "variant": "stabilized",
                "time": replacement[1],
                "step": safe,
                "note_stabilized": (replacement[0] or run["row"])
                + (f" at Δt = {safe:.1e} s" if safe < run["step"] else ""),
            }
        )
    print(
        f"{run['matrix'][:12]:12s} {run['row'][:26]:26s} {run['column'][:15]:15s} "
        f"{run.get('variant') or '':10s} {run['status']:8s} {run['seconds']:6.1f} s  "
        + (f"error {run['error']:.2e}  steps {run['steps']}" if run["status"] == "ran" else run["note"][:80])
    )

# ## Time integrator × variant
#
# The relative error of the last wheel's angle over three lowest periods, and the wall time.

main = {
    (r["row"], r["column"]): r
    for r in runs
    if r.get("variant") is None and r["matrix"] == "time integrator x variant"
}
substitute = {(r["row"], r["column"]): r for r in runs if r.get("variant") == "stabilized"}
hatch = {
    key
    for key, r in main.items()
    if r["status"] != "ran" and substitute.get(key, {}).get("status") == "ran"
}
cell = {key: substitute[key] if key in hatch else r for key, r in main.items()}
rows, columns = list(integrators), list(shafts)
notes = [
    "Hatched: the declared integrator is unstable for this variant at Δt = T1/100; colour and "
    "value show its counterpart:"
] + sorted(
    {
        f"  {key[0]}: {substitute[key]['note_stabilized']}."
        for key in hatch
    }
)
fig = plots.figure(1, 2, size=(17, 12))
for panel, measure, label, title in (
    (fig[0], "error", "relative angle error", "Relative error of the last wheel's angle"),
    (fig[1], "seconds", "wall time [s]", "Wall time and accepted steps"),
):
    plots.matrix(
        [
            [
                cell[(row, c)][measure] if cell[(row, c)]["status"] == "ran" else np.nan
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
                        f"{cell[(row, c)]['error']:.2e}"
                        if measure == "error"
                        else f"{cell[(row, c)]['seconds']:.1f} s, {cell[(row, c)]['steps']}"
                    )
                    if cell[(row, c)]["status"] == "ran"
                    else cell[(row, c)]["status"]
                )
                for c in columns
            ]
            for row in rows
        ],
        hatched=[[(row, c) in hatch for c in columns] for row in rows],
        notes=notes if measure == "error" else None,
        title=title,
        ax=panel,
    )
plots.show()

# ## Time histories
#
# The last wheel of the damped line with second-order methods against the modal solution, and the
# energy-conserving kick-drift step on the undamped line against forward Euler's counterpart.

fig = plots.figure(1, 2, size=(16, 5))
for panel, case, names in (
    (fig[0], "damped", ("staggered Verlet", "classical RK4", "Crank-Nicolson", "BDF2 (Gear)", "backward Euler")),
    (fig[1], "undamped", ("staggered Verlet", "symplectic Euler", "implicit midpoint", "backward Euler")),
):
    plots.line(
        y={
            name: (1e3 * main[(name, case)]["series"][0], 1e3 * main[(name, case)]["series"][1])
            for name in names
            if main[(name, case)]["status"] == "ran"
        },
        reference={"exact modal solution": (1e3 * fine, 1e3 * exact_fine[case])},
        xlabel="time [ms]",
        ylabel="angle of the last wheel [mrad]",
        title=f"{case.capitalize()} line, Δt = T1/100",
        legend="lower left",
        ax=panel,
    )
plots.show()

# ## The pendulum: period against the elliptic integral
#
# A point mass on a massless rod of length $L$: $\ddot\theta+(g/L)\sin\theta=0$. Released from rest at
# $\theta_0$, its exact period is
#
# $$T=4\sqrt{L/g}\;K\!\left(\sin^2\tfrac{\theta_0}{2}\right),$$
#
# with $K$ the complete elliptic integral of the first kind (Landau and Lifshitz, *Mechanics*, §11).
# Every integrator runs four exact periods at $\Delta t=T_0/200$ ($T_0=2\pi\sqrt{L/g}$) for amplitudes
# from 30° to 175°; the measured period is the mean spacing of the downward zero crossings, and the
# matrix shows its relative deviation. Forward Euler is refused (unstable for an undamped swing); its
# counterpart is again the symplectic Euler step. A method whose numerical damping or growth destroys
# the swing near the separatrix (175°) before a full period is repeated at a quarter of the step.

length, gravity = 1.0, 9.81
pendulum = SimplePendulum(domain=domain.system, length=length, mass=0.5, gravity=gravity)
amplitudes = (30, 90, 150, 175)
small_angle = 2 * np.pi * np.sqrt(length / gravity)
periods = {a: 4 * np.sqrt(length / gravity) * ellipk(np.sin(np.radians(a) / 2) ** 2) for a in amplitudes}
swings = {
    a: Problem(
        name=f"Pendulum released at {a}°",
        geometry=domain,
        physics=(pendulum,),
        initial_conditions=(
            InitialValue(pendulum.angle, np.radians(a)),
            InitialValue(pendulum.angular_velocity, 0.0),
        ),
    )
    for a in amplitudes
}
print("exact periods: " + ", ".join(f"{a}° {periods[a]:.5f} s" for a in amplitudes))
pendulum_runs = [
    dict(row=name, column=f"{a}°", amplitude=a, time=method, step=small_angle / 200)
    for name, method in integrators.items()
    for a in amplitudes
]
for run in pendulum_runs:
    simulation = Simulation(
        problem=swings[run["amplitude"]],
        mesh=mesh,
        numerics=Numerics(
            spatial=LumpedParameters(),
            time_integrator=run["time"],
            nonlinear_solver=(
                newton
                if run["time"].implicit and "rosenbrock" not in run["time"].method_id
                else None
            ),
            linear_solver=SparseDirect() if "rosenbrock" in run["time"].method_id else None,
        ),
    )
    started = time.perf_counter()
    run.update(status="rejected", note="", error=np.nan, steps=0)
    try:
        result = simulation.run(
            start_time=0.0, end_time=4 * periods[run["amplitude"]], time_step=run["step"]
        )
        times, angles = result.field(pendulum.angle).time_series()
        crossings = crossing_times(times, angles, 0.0, direction="falling")
        run["error"] = (
            abs(np.diff(crossings).mean() / periods[run["amplitude"]] - 1)
            if len(crossings) > 1
            else np.nan
        )
        run["steps"] = len(times) - 1
        run["status"] = "ran" if np.isfinite(run["error"]) else "failed"
        run["note"] = "" if run["status"] == "ran" else "the swing dies out before a full period"
    except Exception as error:
        message = str(error).splitlines()[0]
        run["status"] = "failed" if "unstable" in message or "finite" in message else "rejected"
        run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    if run["status"] == "failed" and run.get("variant") is None:
        pendulum_runs.append(
            {
                **run,
                "variant": "stabilized",
                "time": stabilized_time.get(run["row"], (None, run["time"]))[1],
                "step": run["step"] / (1 if run["row"] in stabilized_time else 4),
                "note_stabilized": stabilized_time.get(run["row"], (f"{run['row']} at Δt = T0/800",))[0],
            }
        )
    print(
        f"pendulum {run['row'][:26]:26s} {run['column']:5s} {run.get('variant') or '':10s} "
        f"{run['status']:8s} {run['seconds']:6.1f} s  "
        + (f"period error {run['error']:.2e}" if run["status"] == "ran" else run["note"][:80])
    )

swing_main = {(r["row"], r["column"]): r for r in pendulum_runs if r.get("variant") is None}
swing_substitute = {(r["row"], r["column"]): r for r in pendulum_runs if r.get("variant") == "stabilized"}
swing_hatch = {
    key
    for key, r in swing_main.items()
    if r["status"] != "ran" and swing_substitute.get(key, {}).get("status") == "ran"
}
swing_cell = {key: swing_substitute[key] if key in swing_hatch else r for key, r in swing_main.items()}
swing_columns = [f"{a}°" for a in amplitudes]
fig = plots.figure(1, 1, size=(11, 12))
plots.matrix(
    [
        [
            swing_cell[(row, c)]["error"] if swing_cell[(row, c)]["status"] == "ran" else np.nan
            for c in swing_columns
        ]
        for row in rows
    ],
    rows,
    swing_columns,
    scale="log",
    label="relative period error",
    text=[
        [
            (
                f"{swing_cell[(row, c)]['error']:.1e}"
                if swing_cell[(row, c)]["status"] == "ran"
                else swing_cell[(row, c)]["status"]
            )
            for c in swing_columns
        ]
        for row in rows
    ],
    hatched=[[(row, c) in swing_hatch for c in swing_columns] for row in rows],
    notes=(
        ["Hatched: the declared integrator cannot keep the undamped swing at Δt = T0/200; colour and value show its counterpart:"]
        + sorted({f"  {key[0]}: {swing_substitute[key]['note_stabilized']}." for key in swing_hatch})
    ),
    title=f"Pendulum period against 4·√(L/g)·K(sin²(θ0/2)), Δt = T0/200",
    ax=fig[0],
)
plots.show()

# ## Nonlinear × linear solver on the pendulum
#
# The pendulum is nonlinear, so its implicit stages need a genuine nonlinear solve. Implicit midpoint
# steps of the 150° swing at $\Delta t=T_0/50$ are solved with every nonlinear solver and every linear
# solver for its Newton systems; the period must not depend on the solver as long as it converges.

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
        (name, linear_name): kind(linear_solver=linear)
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
        if linear_name not in ("sparse LU", "automatic")
    },
    ("fixed point", "none"): FixedPoint(),
    ("fixed point + Aitken", "none"): FixedPoint(acceleration="aitken"),
}
solver_runs = [dict(row=row, column=column, solver=solver) for (row, column), solver in nonlinear_solvers.items()]
for run in solver_runs:
    started = time.perf_counter()
    run.update(status="rejected", note="", error=np.nan)
    try:
        result = Simulation(
            problem=swings[150],
            mesh=mesh,
            numerics=Numerics(
                spatial=LumpedParameters(),
                time_integrator=TimeMethod("implicit_rk_midpoint"),
                nonlinear_solver=run["solver"],
            ),
        ).run(start_time=0.0, end_time=4 * periods[150], time_step=small_angle / 50)
        times, angles = result.field(pendulum.angle).time_series()
        run["error"] = abs(
            np.diff(crossing_times(times, angles, 0.0, direction="falling")).mean() / periods[150] - 1
        )
        run["status"] = "ran"
    except Exception as error:
        message = str(error).splitlines()[0]
        run["status"] = "failed" if "did not converge" in message else "rejected"
        run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    print(
        f"solver {run['row'][:22]:22s} {run['column']:15s} {run['status']:8s} {run['seconds']:6.2f} s  "
        + (f"period error {run['error']:.3e}" if run["status"] == "ran" else run["note"][:80])
    )
solver_rows = list(dict.fromkeys(r["row"] for r in solver_runs))
solver_columns = [*linear_solvers, "none"]
solver_grid = {(r["row"], r["column"]): r for r in solver_runs}
fig = plots.figure(1, 1, size=(13, 5))
plots.matrix(
    [
        [
            solver_grid[(row, c)]["seconds"]
            if (row, c) in solver_grid and solver_grid[(row, c)]["status"] == "ran"
            else np.nan
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
                f"{solver_grid[(row, c)]['seconds']:.1f} s\n{solver_grid[(row, c)]['error']:.2e}"
                if solver_grid[(row, c)]["status"] == "ran"
                else solver_grid[(row, c)]["status"]
            )
            if (row, c) in solver_grid
            else ""
            for c in solver_columns
        ]
        for row in solver_rows
    ],
    title="Implicit midpoint, 150° swing: wall time and period error per solver pair",
    ax=fig[0],
)
plots.show()

# ## Step check: Δt against Δt/2
#
# Representative integrators on the damped line at half the step. The error ratio shows the observed
# order: about 4 for a second-order method, 2 for a first-order one.

halved = {r["row"]: r for r in runs if r["matrix"] == "step check"}
print(f"{'integrator':26s} {'error Δt':>11s} {'error Δt/2':>11s} {'ratio':>7s} {'order':>6s}")
for name, r in halved.items():
    full = main[(name, "damped")]
    if r["status"] == "ran" and full["status"] == "ran":
        print(
            f"{name:26s} {full['error']:11.3e} {r['error']:11.3e} {full['error'] / r['error']:7.2f} "
            f"{np.log2(full['error'] / r['error']):6.2f}"
        )

# ## What the library refused, and what failed

everything = runs + pendulum_runs + solver_runs
for status in ("rejected", "failed"):
    selected = [r for r in everything if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(f"  {r['row'][:28]:28s} {str(r['column'])[:16]:16s} {r['note'][:100]}")
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in everything)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(everything)} runs ({sum(r.get('variant') == 'stabilized' for r in everything)} counterparts), "
    + f"{sum(r['seconds'] for r in everything) / 60:.1f} min in total"
)

# ## Methods that do not apply
#
# A lumped drivetrain has no spatial discretization, so there is no space axis, no flux and no mesh
# refinement; the step check takes its place. Conjugate gradients and algebraic multigrid need a
# symmetric positive definite stage matrix, which the first-order form $(\theta,\omega)$ does not
# have. Runge-Kutta-Chebyshev and -Legendre super-steps are for parabolic operators with a real
# spectrum; IMEX splits need a separate stiff and non-stiff part, which the declared model does not
# publish. JFNK never forms the Jacobian, so it has no matrix for a sparse direct (or automatic, which
# selects the direct) inner solve; it is paired with the Krylov solvers only.
#
# ## Going further
#
# Add gear ratios through `gear_ratios`, drive the first wheel through `applied_torque(0)`, or couple
# the line to a controller; the matrix code stays as it is.
