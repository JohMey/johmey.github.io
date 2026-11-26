"""OneSimulation example: a travelling and reflected sound pulse through every method combination OneSimulation offers.
Requires the installed, activated OneSimulation runtime.

One physics declaration, linear acoustics in pressure and velocity, is solved in a 1 m tube with rigid
ends on 400 cells with every applicable combination of discretization family, reconstruction,
interface flux, time integrator, nonlinear solver and linear solver. The staggered pressure-velocity
grid and the second-order pressure wave equation run as additional rows. Every run is graded against
d'Alembert's solution.
"""

# # A sound pulse in a tube: one model, every method
#
# Air ($\rho_0=1.21$ kg/m³, $c=343$ m/s) fills a tube of length 1 m closed by rigid walls. At rest, a
# Gaussian pressure pulse $p_0(x)=\exp(-((x-x_0)/\sigma)^2)$ Pa with $x_0=0.3$ m and $\sigma=0.03$ m is
# released. The linear acoustic equations
#
# $$\partial_t p+\rho_0c^2\,\partial_x u=0,\qquad \partial_t u+\frac1{\rho_0}\partial_x p=0$$
#
# are declared once. The pulse splits into two halves. The right half travels towards $x=1$ m; the
# left half is reflected at the rigid wall $x=0$ ($u=0$, pressure unchanged in sign) and follows it.
# At $t=1.2$ ms both halves travel to the right, 0.6 m apart.
#
# The same physics is also declared as the second-order pressure wave equation
# $\chi\,\partial_t^2p-\partial_x(\rho_0^{-1}\partial_x p)=0$ with $\chi=1/(\rho_0c^2)$, whose natural
# boundary condition is the rigid wall.
#
# **Reference.** d'Alembert's solution with the mirror image of the pulse at $x=-x_0$:
#
# $$p(x,t)=\tfrac12\big[p_0(x-ct)+p_0(ct-x)\big],\qquad u=\frac{p}{\rho_0 c}$$
#
# for $x\in[0,1]$ m and $t<(1-x_0-3\sigma)/c$; the left-travelling parts have left the tube.
#
# **Grading.** At $t=1.2$ ms every run is evaluated at the 400 cell centres:
# $L_1(p)=\sum_i|p_h(x_i)-p(x_i)|\,\Delta x$ and the same for $u$; the wall time covers validation,
# binding and time stepping.

import time

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.meshing.meshes.api import cartesian
from onesim.numerics import Numerics
from onesim.numerics.discretization import NodalInterpolation
from onesim.numerics.fdm import (
    BoundaryFluxDifference,
    CentralDifference,
    FiniteDifference,
    GlobalLaxFriedrichsSplit,
    PolynomialDifference,
    SummationByPartsDifference,
    WENO5Flux,
    WENOFluxDifference,
)
from onesim.numerics.fem import (
    CIP,
    GLS,
    SUPG,
    CharacteristicTau,
    DiscontinuousLagrange,
    FiniteElement,
    InterpolatedFluxDivergence,
    Lagrange,
)
from onesim.numerics.fluxes import HLL, HLLE, Rusanov
from onesim.numerics.fvm import (
    MUSCL,
    QUICK,
    CentralWeightedENO,
    Eno3Reconstructor,
    FiniteVolume,
    JiangShuWeights,
    MinmodLimiter,
    MonotonizedCentralLimiter,
    SuperbeeLimiter,
    VanLeerLimiter,
    Weno3Reconstructor,
    Weno5Reconstructor,
)
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
    StaggeredVerlet,
    TimeMethod,
    TimeStepLimits,
)
from onesim.physics.acoustics import (
    AcousticPressureWave,
    LinearAcoustics,
    StaggeredAcousticDifference,
)
from onesim.simulation import Problem, ReflectingWall, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# The first-order pressure-velocity model and the second-order pressure wave share the medium, the
# tube and the pulse. Nothing in this cell names a numerical method.

density, sound_speed, centre, width, end_time = 1.21, 343.0, 0.3, 0.03, 1.2e-3
cells, coarse_cells = 400, 200
tube = Line(start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0))
air = LinearAcoustics(domain=tube.line, density=density, sound_speed=sound_speed)
wave = AcousticPressureWave(
    domain=tube.line, density=density, compressibility=1 / (density * sound_speed**2)
)
problems = {
    "first order": Problem(
        name="Sound pulse, pressure and velocity",
        geometry=tube,
        physics=(air,),
        boundary_conditions=(ReflectingWall(physics=air, boundary=tube.boundaries.all),),
        initial_conditions=(InitialValue(air.pressure, lambda x: np.exp(-(((x - centre) / width) ** 2))), InitialValue(air.velocity, 0.0)),
    ),
    "second order": Problem(
        name="Sound pulse, pressure wave",
        geometry=tube,
        physics=(wave,),
        initial_conditions=(
            InitialValue(wave.pressure, lambda x: np.exp(-(((x - centre) / width) ** 2))),
            InitialValue(wave.pressure_rate, 0.0),
        ),
    ),
}
meshes = {
    **{("line", n): StructuredLineMesher(cells=n).generate(tube) for n in (cells, coarse_cells)},
    **{
        ("staggered", n): cartesian(shape=(n,), extents=(1.0,), geometry=tube, periodic=False).build()
        for n in (cells, coarse_cells)
    },
}
centres = {n: (np.arange(n) + 0.5) / n for n in (cells, coarse_cells)}
print(
    f"{meshes[('line', cells)].cell_count} cells of width {1000 / cells:.2f} mm "
    f"(coarse check: {coarse_cells} cells); the staggered grid has {cells} pressure cells "
    f"and {cells + 1} velocity faces"
)

# ## The exact solution

fine = np.linspace(0.0, 1.0, 4001)
exact = {
    key: np.array(
        [
            0.5
            * (
                np.exp(-(((x - sound_speed * end_time - centre) / width) ** 2))
                + np.exp(-(((sound_speed * end_time - x - centre) / width) ** 2))
            )
            * scale
            for scale in (1.0, 1 / (density * sound_speed))
        ]
    )
    for key, x in {**centres, "fine": fine}.items()
}
print(
    f"pulse halves at x = {centre + sound_speed * end_time:.4f} m and "
    f"{sound_speed * end_time - centre:.4f} m, peak 0.5 Pa and {0.5 / (density * sound_speed) * 1e3:.4f} mm/s"
)

# ## The method catalogue
#
# Three interface fluxes: Rusanov, HLL and HLLE. For linear acoustics the slowest and fastest
# signal speeds are exactly $-c$ and $+c$, so HLL and HLLE are the exact (Godunov) flux of the
# linear system, and so is Rusanov, whose dissipation speed is also $c$: the three columns must agree
# to rounding, which the matrix shows. HLLC is not offered (see the end). Slope limiters and
# higher-order reconstructions for finite volumes, discontinuous Galerkin of degree 0 to 4,
# continuous Galerkin of degree 1 to 3 with and without stabilization, and finite-difference
# stencils. The staggered pressure-velocity grid and the second-order pressure wave in continuous
# elements need no interface flux and form their own column.

fluxes = {"Rusanov": Rusanov(), "HLL": HLL(), "HLLE": HLLE()}
no_flux = "no interface flux"
limiters = {
    "minmod": MinmodLimiter(),
    "MC": MonotonizedCentralLimiter(),
    "van Leer": VanLeerLimiter(),
    "superbee": SuperbeeLimiter(),
}
reconstructions = {
    "ENO3": Eno3Reconstructor(),
    "WENO3": Weno3Reconstructor(),
    "WENO5": Weno5Reconstructor(),
    "CWENO3": CentralWeightedENO(weights=JiangShuWeights()),
    "QUICK": QUICK(),
}
spatial = {}
for flux_name, flux in fluxes.items():
    spatial[("FVM first order", flux_name)] = FiniteVolume(convection=flux)
    spatial[("FVM MUSCL, unlimited", flux_name)] = FiniteVolume(
        convection=flux, reconstruction=MUSCL()
    )
    for limiter_name, limiter in limiters.items():
        spatial[(f"FVM MUSCL + {limiter_name}", flux_name)] = FiniteVolume(
            convection=flux, reconstruction=MUSCL(), limiter=limiter
        )
    for reconstruction_name, reconstruction in reconstructions.items():
        spatial[(f"FVM {reconstruction_name}", flux_name)] = FiniteVolume(
            convection=flux, reconstruction=reconstruction
        )
    for degree in (0, 1, 2, 3, 4):
        spatial[(f"FEM DG P{degree}", flux_name)] = FiniteElement(
            element=DiscontinuousLagrange(degree), convection=flux
        )
    for degree in (1, 2, 3):
        spatial[(f"FEM CG P{degree} Galerkin", flux_name)] = FiniteElement(
            element=Lagrange(degree), convection=flux
        )
    spatial[("FEM CG P2 + continuous interior penalty", flux_name)] = FiniteElement(
        element=Lagrange(2),
        convection=flux,
        stabilization=CIP(penalty=0.01),
        initialization=NodalInterpolation(),
    )
    for name, kind in (("SUPG", SUPG), ("GLS", GLS)):
        spatial[(f"FEM CG P1 + {name}", flux_name)] = FiniteElement(
            element=Lagrange(1),
            convection=flux,
            stabilization=kind(tau=CharacteristicTau(), strong_residual=InterpolatedFluxDivergence()),
            initialization=NodalInterpolation(),
        )
    spatial[("FDM central, 2nd order", flux_name)] = FiniteDifference(
        gradient=BoundaryFluxDifference(derivative=CentralDifference(), boundary_flux=flux)
    )
    spatial[("FDM polynomial, 4th order", flux_name)] = FiniteDifference(
        gradient=BoundaryFluxDifference(
            derivative=PolynomialDifference(accuracy_order=4), boundary_flux=flux
        )
    )
    for order in (2, 4):
        spatial[(f"FDM summation by parts, order {order}", flux_name)] = FiniteDifference(
            gradient=BoundaryFluxDifference(
                derivative=SummationByPartsDifference(accuracy_order=order), boundary_flux=flux
            )
        )
spatial[("FDM WENO5 flux splitting", no_flux)] = FiniteDifference(
    gradient=WENOFluxDifference(flux_splitting=GlobalLaxFriedrichsSplit(), reconstruction=WENO5Flux())
)
spatial[("staggered pressure-velocity grid", no_flux)] = StaggeredAcousticDifference()
for degree in (1, 2, 3):
    spatial[(f"pressure wave, FEM CG P{degree}", no_flux)] = FiniteElement(
        element=Lagrange(degree),
        stability=SpectralEnclosureStability(enclosure="imaginary_axis", spectral_radius="arnoldi"),
    )
# SUPG and GLS weight the residual with the flux Jacobian $A$. For the pressure-velocity form
# $A=\begin{pmatrix}0&\rho_0c^2\\1/\rho_0&0\end{pmatrix}$ is not symmetric, and in SI units its two
# fields differ by twelve orders of magnitude, so no single scalar $\tau$ fits both. Both
# stabilizations therefore take the characteristic matrix parameter
# $\tau=\tfrac{h}{2}\,|A|^{-1}$ (`CharacteristicTau`), formed from the eigen-decomposition of $A$:
# the streamline test matrix becomes $\tau A=\tfrac{h}{2}\,\mathrm{sign}(A)$, so each sound wave gets
# the scalar SUPG weighting, and the added streamline diffusion is $\tfrac{h}{2}|A|=\tfrac{hc}{2}I$
# for pressure and velocity alike.
form = {
    key: (
        ("second order", "line")
        if key[0].startswith("pressure wave")
        else ("first order", "staggered" if key[0].startswith("staggered") else "line")
    )
    for key in spatial
}

# Time integrators: every explicit one-step and multistep method, staggered Verlet, implicit
# Runge-Kutta, Adams-Moulton and BDF methods, Rosenbrock methods and the adaptive integrators.
# Implicit stages take Newton with a coloured finite-difference Jacobian and a sparse direct solve;
# the rate of the pressure wave contains the inverse of the consistent mass matrix, so its Jacobian
# is dense and its implicit stages take the matrix-free JFNK with GMRES. Its Rosenbrock stages
# instead solve the mass-weighted stage matrix [[I, −cT], [cK, M + cD]], which is sparse, with the
# sparse direct solver.
# Staggered Verlet is a kick-drift method for a partitioned system: it runs on the staggered grid
# (kick the face velocities, drift the cell pressures) and on the pressure wave (kick the pressure
# rate, drift the pressure), where it is the leapfrog scheme of the wave equation.

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
    "staggered Verlet": StaggeredVerlet(),
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
    "adaptive BDF": AdaptiveBDF(relative_tolerance=1e-3, absolute_tolerance=1e-6),
}
newton = Newton(linear_solver=SparseDirect(), jacobian="colored_finite_difference")
implicit_step = 1.5e-5
# Leapfrog is unstable on the dissipative upwind operators at every step; its counterpart adds the
# Robert-Asselin-Williams filter. Forward Euler and Adams-Bashforth 2 are stable on the upwind
# operators only far below the step of the higher-order methods; their counterpart is the same
# method at a reduced safety factor. On the non-dissipative operators (staggered grid, pressure
# wave) every eigenvalue lies on the imaginary axis, where forward Euler, Heun, SSP-RK2, explicit
# midpoint, Heun-Euler and Adams-Bashforth 2 are unstable at every step. On the pressure wave the
# spectral enclosure refuses them with that reason; the staggered grid refuses forward Euler and
# lets the others run, where the growth per step is of order $(\omega\Delta t)^4$ and stays small over
# this short run except for Adams-Bashforth 2. Cells that cannot have a stable step read
# "no stable step". BDF4 to BDF6 and Adams-Moulton 4 are not A-stable
# and the fixed-point iteration contracts only for small steps; their counterpart is the same method
# at a step it is stable for. The pressure wave takes its explicit step from the
# imaginary-axis spectral enclosure (Arnoldi estimate of the highest frequency) and the selected
# method's stability region; staggered Verlet takes its step from the same enclosure (stable for
# $\omega\Delta t\le2$, derived from the Verlet kernel itself).
stabilized_time = {
    "leapfrog": (
        "leapfrog + Robert-Asselin-Williams filter (ν = 0.2, α = 0.53; first order), safety factor 0.15",
        TimeMethod("leapfrog_raw"),
        0.15,
    ),
    "forward Euler": ("forward Euler at safety factor 0.05", ForwardEuler(), 0.05),
    "Adams-Bashforth 2": ("Adams-Bashforth 2 at safety factor 0.1", TimeMethod("adams_bashforth2"), 0.1),
}
stable_step = {
    "BDF4 (Gear)": 2.5e-6,
    "BDF5 (Gear)": 2.5e-6,
    "BDF6 (Gear)": 2.5e-6,
    "Adams-Moulton 3": 2.5e-6,
    "Adams-Moulton 4": 2.5e-6,
    "fixed point": 1e-6,
    "fixed point + Aitken": 1e-6,
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
    **{("JFNK", linear_name): JFNK(linear_solver=linear) for linear_name, linear in linear_solvers.items()},
    ("fixed point", "none"): FixedPoint(),
    ("fixed point + Aitken", "none"): FixedPoint(acceleration="aitken"),
}

# ## The test matrix
#
# 1. **Space × interface flux** — every spatial scheme with every flux, SSP-RK3 at the stable step
#    (safety factor 0.5).
# 2. **Time integrator × space** — every integrator on four representative schemes: MUSCL-MC
#    volumes and DG P2 with HLL, the staggered grid and the pressure wave in CG P2. Explicit
#    integrators run at their stable step, implicit ones at $\Delta t=1.5\cdot10^{-5}$ s (CFL about
#    2). Staggered Verlet runs on the two partitioned schemes.
# 3. **Nonlinear × linear solver** — backward Euler on MUSCL-MC-HLL volumes, $\Delta t=1.5\cdot10^{-5}$ s;
#    the Newton variants take the analytic Jacobian (the HLL face derivative is the upwind physical
#    Jacobian on one-sided faces and a difference quotient elsewhere).
#
# Representative combinations are repeated on 200 cells for the mesh check.

time_spatial = {
    "FVM MUSCL + MC, HLL": ("FVM MUSCL + MC", "HLL"),
    "FEM DG P2, HLL": ("FEM DG P2", "HLL"),
    "staggered grid": ("staggered pressure-velocity grid", no_flux),
    "pressure wave, CG P2": ("pressure wave, FEM CG P2", no_flux),
}
partitioned = ("staggered grid", "pressure wave, CG P2")
imaginary_unstable = (
    "forward Euler",
    "SSP-RK2",
    "Heun",
    "explicit midpoint",
    "Adams-Bashforth 2",
    "Heun-Euler (embedded)",
)
runs = [
    dict(matrix="space x interface flux", row=row, column=column, key=(row, column),
         time=TimeMethod("ssprk3"), nonlinear=None, step=end_time, cells=cells)
    for row, column in spatial
]
runs += [
    dict(
        matrix="time integrator x space",
        row=name,
        column=space,
        key=key,
        time=integrator,
        nonlinear=(
            (JFNK(linear_solver=GMRES()) if space == "pressure wave, CG P2" else newton)
            if getattr(integrator, "implicit", False)
            and "rosenbrock" not in getattr(integrator, "method_id", "")
            else None
        ),
        linear=SparseDirect() if "rosenbrock" in getattr(integrator, "method_id", "") else None,
        step=implicit_step if getattr(integrator, "implicit", False) else end_time,
        scheme=spatial[key],
        cells=cells,
    )
    for name, integrator in integrators.items()
    for space, key in time_spatial.items()
    if name != "staggered Verlet" or space in partitioned
]
runs += [
    dict(matrix="nonlinear x linear solver", row=row, column=column, key=time_spatial["FVM MUSCL + MC, HLL"],
         time=BackwardEuler(), nonlinear=solver, step=implicit_step, cells=cells)
    for (row, column), solver in nonlinear_solvers.items()
]
runs += [
    {
        **run,
        "variant": "stabilized",
        "time": stabilized_time[run["row"]][1],
        "safety": stabilized_time[run["row"]][2],
        "note_stabilized": stabilized_time[run["row"]][0],
    }
    for run in list(runs)
    if run["matrix"] == "time integrator x space" and run["row"] in stabilized_time
]
runs += [
    {
        **run,
        "variant": "stabilized",
        "step": stable_step[run["row"]],
        "note_stabilized": f"{run['row']} at Δt = {stable_step[run['row']]:.1e} s",
    }
    for run in list(runs)
    if run["matrix"] in ("time integrator x space", "nonlinear x linear solver")
    and run["row"] in stable_step
    and run.get("variant") is None
]
representative = [
    ("FVM first order", "HLL"),
    ("FVM MUSCL + MC", "HLL"),
    ("FVM WENO5", "HLLE"),
    ("FEM DG P2", "HLL"),
    ("FEM CG P2 Galerkin", "Rusanov"),
    ("FDM summation by parts, order 4", "HLL"),
    ("staggered pressure-velocity grid", no_flux),
    ("pressure wave, FEM CG P2", no_flux),
]
runs += [
    {**run, "matrix": "mesh check", "cells": coarse_cells}
    for run in runs
    if run["matrix"] == "space x interface flux" and run["key"] in representative
]
print(
    f"{len(runs)} runs: "
    + ", ".join(f"{sum(r['matrix'] == m for r in runs)} {m}" for m in dict.fromkeys(r["matrix"] for r in runs))
)

# ## Run everything

for run in runs:
    case, kind = form[run["key"]]
    model, n = problems[case].physics[0], run["cells"]
    simulation = Simulation(
        problem=problems[case],
        mesh=meshes[(kind, n)],
        numerics=Numerics(
            spatial=run.get("scheme", spatial[run["key"]]),
            linear_solver=(
                SparseDirect()
                if isinstance(run.get("scheme", spatial[run["key"]]), FiniteElement)
                else run.get("linear")
            ),
            time_integrator=run["time"],
            nonlinear_solver=run["nonlinear"],
            step_limits=TimeStepLimits(stability_safety_factor=run.get("safety", 0.5)),
        ),
    )
    started = time.perf_counter()
    check = simulation.validate()
    run.update(status="rejected", note="", errors=(np.nan,) * 2, steps=0)
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = simulation.run(start_time=0.0, end_time=end_time, time_step=run["step"])
            points = np.column_stack((centres[n], np.zeros((n, 2))))[:, : 1 if kind == "staggered" else 3]
            pressure = result.field(model.pressure).evaluate(points).reshape(-1)
            velocity = (
                result.field(model.velocity).evaluate(points).reshape(-1)
                if case == "first order"
                else np.full(n, np.nan)
            )
            run["profiles"] = np.array([pressure, velocity])
            run["errors"] = tuple(np.abs(run["profiles"] - exact[n]).sum(axis=1) / n)
            run["steps"] = len(result.times) - 1
            run["status"] = "ran" if np.isfinite(pressure).all() and np.abs(pressure).max() < 10 else "failed"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "no stable step"
                if "nonzero imaginary" in message or "declares time method" in message
                else "failed"
                if any(word in message for word in ("finite", "did not converge", "unstable", "clock resolution"))
                else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    if (
        run["status"] == "failed"
        and run["column"] in partitioned
        and run["row"] in imaginary_unstable
    ):
        run["status"] = "no stable step"
    run["seconds"] = time.perf_counter() - started
    print(
        f"{run['matrix'][:14]:14s} {run['row'][:40]:40s} {run['column'][:22]:22s} "
        f"{run.get('variant') or '':10s} {run['status']:8s} {run['seconds']:6.1f} s  "
        + (f"L1 p {run['errors'][0]:.2e}" if run["status"] == "ran" else run["note"][:90])
    )

# ## The matrices
#
# The pressure error of every combination. Hatched cells failed as declared; colour and number show
# the stabilized counterpart named in the legend.

main = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") is None}
substitute = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") == "stabilized"}
hatch = {
    key for key, r in main.items() if r["status"] != "ran" and substitute.get(key, {}).get("status") == "ran"
}
cell = {key: substitute[key] if key in hatch else r for key, r in main.items()}
layouts = {
    "space x interface flux": ([*fluxes, no_flux], (12, 15), "Hatched: the declared scheme failed; colour and value show:"),
    "time integrator x space": (
        list(time_spatial),
        (13, 12),
        "Hatched: the declared integrator is unstable at this step (not A-stable, undamped, or explicit "
        "Euler on a non-dissipative operator); colour and value show:",
    ),
    "nonlinear x linear solver": (
        [*linear_solvers, "none"],
        (13, 6),
        "Hatched: the fixed-point iteration cannot contract at Δt = 1.5·10⁻⁵ s; colour and value show:",
    ),
}
for key, (columns, size, head) in layouts.items():
    rows = list(dict.fromkeys(r["row"] for r in runs if r["matrix"] == key))
    fig = plots.figure(1, 1, size=size)
    plots.matrix(
        [[cell[(key, row, c)]["errors"][0] if (key, row, c) in cell else np.nan for c in columns] for row in rows],
        rows,
        columns,
        scale="log",
        label="L1 pressure error at t = 1.2 ms [Pa m]",
        text=[
            [
                (
                    f"{cell[(key, row, c)]['errors'][0]:.2e}"
                    if cell[(key, row, c)]["status"] == "ran"
                    else cell[(key, row, c)]["status"]
                )
                if (key, row, c) in cell
                else "–"
                for c in columns
            ]
            for row in rows
        ],
        hatched=[[(key, row, c) in hatch for c in columns] for row in rows],
        notes=[head]
        + sorted(
            {
                f"  {row}: {substitute[(key, row, c)]['note_stabilized']}."
                for row in rows
                for c in columns
                if (key, row, c) in hatch
            }
        ),
        title=f"{key[0].upper() + key[1:]}, {cells} cells",
        ax=fig[0],
    )
    plots.show()

# All measures for the combinations that ran, sorted by the pressure error.

for key in layouts:
    print(f"\n{key}")
    print(f"{'scheme':40s} {'column':22s} {'L1 p':>9s} {'L1 u':>9s} {'steps':>6s} {'time':>8s}")
    for r in sorted((r for k, r in cell.items() if k[0] == key and r["status"] == "ran"), key=lambda r: r["errors"][0]):
        print(
            f"{r['row']:40s} {r['column']:22s} {r['errors'][0]:9.2e} {r['errors'][1]:9.2e} "
            f"{r['steps']:6d} {r['seconds']:7.1f}s"
        )

# ## Profiles of representative combinations
#
# Pressure over the whole tube and a close-up of the reflected half, against d'Alembert's solution.

shown = {
    f"{r['row']}, {r['column']}": (centres[cells], r["profiles"][0])
    for k, r in main.items()
    if k[0] == "space x interface flux" and r["key"] in representative and r["status"] == "ran"
}
fig = plots.figure(1, 2, size=(15, 5))
for panel, window, title in (
    (fig[0], None, "Pressure at t = 1.2 ms"),
    (fig[1], (0.0, 0.25), "Reflected half, close-up"),
):
    plots.line(
        y=shown,
        reference={"d'Alembert (exact)": (fine, exact["fine"][0])},
        xlabel="position x [m]",
        ylabel="pressure p [Pa]",
        title=title,
        xlimits=window,
        legend="upper right" if window is None else False,
        ax=panel,
    )
plots.show()

# ## Mesh check: 400 against 200 cells
#
# The pulse is smooth, so the error falls with the design order of each scheme.

print(f"{'scheme':40s} {'column':18s} {'L1 p 200':>10s} {'L1 p 400':>10s} {'order':>6s}")
for r in (r for r in runs if r["matrix"] == "mesh check"):
    fine_run = main[("space x interface flux", r["row"], r["column"])]
    if r["status"] == "ran" and fine_run["status"] == "ran":
        print(
            f"{r['row']:40s} {r['column'][:18]:18s} {r['errors'][0]:10.3e} {fine_run['errors'][0]:10.3e} "
            f"{np.log2(r['errors'][0] / fine_run['errors'][0]):6.2f}"
        )

# ## What the library refused, and what failed

for status in ("rejected", "failed", "no stable step"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(f"  {r['matrix'][:24]:24s} {r['row'][:40]:40s} {r['column'][:22]:22s} {r.get('variant') or '':10s} {r['note'][:90]}")
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed", "no stable step"))
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.0f} min in total"
)

# ## Methods that do not apply to the sound pulse
#
# **HLLC** restores a contact wave between the two outer signals; linear acoustics has only the two
# sound waves $\mp c$, which HLL already resolves exactly. The library refuses HLLC for this law with a
# message naming HLL, HLLE and Rusanov. **Staggered Verlet** needs a partitioned system in which the
# rate of each partition depends only on the other: the staggered grid and the pressure wave are
# such systems, the upwind finite-volume, DG and finite-difference operators are not, because their
# interface dissipation couples pressure to pressure. **Limiters for invariant domains** and
# **graph viscosity** guard positivity and have nothing to guard in linear acoustics. Conjugate
# gradients and algebraic multigrid need a symmetric positive definite operator; Runge-Kutta-Chebyshev
# and -Legendre super-steps are for parabolic operators; IMEX splits need a stiff and a non-stiff part.
#
# ## Going further
#
# Make the pulse narrower until the coarse grid no longer resolves it, add `velocity_damping` to the
# model, or replace the rigid end at $x=1$ m with an impedance boundary; the matrix code stays as it is.
