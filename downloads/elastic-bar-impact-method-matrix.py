"""OneSimulation example: an elastic bar struck at one end, through every method combination.
Requires the installed, activated OneSimulation runtime.

One physics declaration, an axial elastic bar, is solved as a static bar under end traction and
body force, as a travelling and reflected stress wave after an impact, and as a vibrating rod.
The bar runs through both of its formulations, the second-order displacement equation and the
first-order velocity-stress system, with every spatial scheme, Riemann solver, time integrator
(including the structural Newmark, HHT-alpha and generalized-alpha methods), nonlinear and
linear solver that applies. Each run is graded against the closed-form solution.
"""

# # An elastic bar struck at one end: one model, every method
#
# A straight, homogeneous bar carries axial motion only (uniaxial stress). Its displacement $u$,
# velocity $v=\partial_t u$ and stress $\sigma=E\,\partial_x u$ obey
#
# $$\rho\,\partial_t v=\partial_x\sigma+f,\qquad \partial_t\sigma=E\,\partial_x v,\qquad
# \partial_t u=v,$$
#
# or, eliminating $v$ and $\sigma$, the wave equation $\rho\,\partial_t^2u=\partial_x(E\,\partial_xu)+f$
# with the bar wave speed $c=\sqrt{E/\rho}$. All quantities are scaled with the bar length $L$,
# the density $\rho$ and the wave speed $c$: $E=\rho=c=1$.
#
# The bar is declared once with `ElasticBar`. Everything numerical is an argument of `Numerics`,
# and that choice also picks the formulation:
#
# * a **continuous Lagrange element without an interface flux** solves the second-order
#   displacement equation: stationary, or in time with the structural Newmark, HHT-α and
#   generalized-α integrators or with any generic time method through the first-order reduction
#   $(u,v)$;
# * **every scheme with an interface flux** (finite volumes, finite differences, discontinuous
#   Galerkin, continuous Galerkin with a Riemann flux) solves the velocity-stress system
#   $\partial_t q+\partial_x F(q)=S$ with $q=(v,\sigma,u)$, $F=(-\sigma/\rho,\,-Ev,\,0)$ and
#   $S=(f/\rho,\,0,\,v)$, whose waves travel at $-c$, $0$ and $+c$.
#
# **Three cases.**
#
# 1. *Static bar*, $L=2$, clamped at $x=0$, end traction $\sigma_0=1$ at $x=2$ and the body force
#    $f=1/2$: $\sigma(x)=\sigma_0+f(2-x)$ and $u(x)=\big(\sigma_0x+f\,x(2-x/2)\big)/E$.
# 2. *Impact*, $L=1$: the bar is at rest; from $t=0$ the end $x=0$ moves with $v_0=1$, the end
#    $x=1$ is free. The stress front $\sigma=-\rho c v_0$ runs to the free end, reflects with the
#    opposite sign and doubles the velocity behind it. At $t=1.5$ (d'Alembert):
#    $\sigma=-1,\ v=1,\ u=1.5-x$ for $x<0.5$ and $\sigma=0,\ v=2,\ u=1$ for $x>0.5$.
# 3. *Vibrating rod*, $L=1$, both ends clamped, released from $u_0=\sin\pi x+\tfrac12\sin 5\pi x$:
#    $u=\sin\pi x\cos\omega_1t+\tfrac12\sin5\pi x\cos\omega_5t$ with $\omega_n=n\pi c/L$.
#
# **Grading.** The impact runs are evaluated at the 400 cell centres $x_i$ of the finest mesh:
# $L_1(q)=\frac1N\sum_i|q_h(x_i)-q(x_i)|$ for stress, velocity and displacement; the wall time
# covers validation, binding and time stepping.

import time

import numpy as np
from onesim.cad.d1 import Line
from onesim.materials.solids.isotropic import IsotropicSolid
from onesim.meshing import StructuredLineMesher
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
    DGJumpPenalty,
    DGPolynomialLimiter,
    DiscontinuousLagrange,
    FiniteElement,
    InterpolatedFluxDivergence,
    InvariantDomainGraphViscosity,
    Lagrange,
    RowSumLumping,
    VolumeCIP,
)
from onesim.numerics.fluxes import HLL, HLLC, HLLE, Rusanov
from onesim.numerics.fvm import (
    MUSCL,
    QUICK,
    BarthJespersenLimiter,
    CentralWeightedENO,
    Eno2Reconstructor,
    Eno3Reconstructor,
    FiniteVolume,
    JiangShuWeights,
    MinmodLimiter,
    MonotonizedCentralLimiter,
    SmoothnessDifference,
    SuperbeeLimiter,
    TvbLimiter,
    VanLeerLimiter,
    VenkatakrishnanLimiter,
    Weno3Reconstructor,
    Weno5Reconstructor,
    WenoZWeights,
)
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
    GeneralizedAlpha,
    HHTAlpha,
    Newmark,
    TimeMethod,
    TimeStepLimits,
)
from onesim.physics.solid_mechanics import (
    BarEndTraction,
    BarEndVelocity,
    BarFixedEnd,
    ElasticBar,
    ElasticBarInvariantDomain,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# Nothing in this cell names a numerical method. The material is a generic isotropic solid in
# scaled units; the bar uses its Young's modulus and density.

cells, coarse_cells, end_time, static_cells, rod_cells = 400, 200, 1.5, 100, 100
material = IsotropicSolid(
    density=1.0,
    specific_heat_capacity=1.0,
    thermal_conductivity=1.0,
    youngs_modulus=1.0,
    poisson_ratio=0.3,
    thermal_expansion_coefficient=0.0,
)
rod = Line(start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0))
bar = ElasticBar(domain=rod.line, material=material)
impact = Problem(
    name="bar impact",
    geometry=rod,
    physics=(bar,),
    boundary_conditions=(
        BarEndVelocity(boundary=rod.boundaries.start, physics=bar, value=1.0),
        BarEndTraction(boundary=rod.boundaries.end, physics=bar, value=0.0),
    ),
    initial_conditions=tuple(
        InitialValue(field, 0.0) for field in (bar.velocity, bar.stress, bar.displacement)
    ),
)
long_rod = Line(start=(0.0, 0.0, 0.0), end=(2.0, 0.0, 0.0))
loaded = ElasticBar(domain=long_rod.line, material=material, body_force=0.5)
static = Problem(
    name="bar under end traction and body force",
    geometry=long_rod,
    physics=(loaded,),
    boundary_conditions=(
        BarFixedEnd(boundary=long_rod.boundaries.start, physics=loaded),
        BarEndTraction(boundary=long_rod.boundaries.end, physics=loaded, value=1.0),
    ),
)
vibrating = ElasticBar(domain=rod.line, material=material)
vibration = Problem(
    name="clamped rod released from two modes",
    geometry=rod,
    physics=(vibrating,),
    boundary_conditions=(
        BarFixedEnd(boundary=rod.boundaries.all, physics=vibrating),
    ),
    initial_conditions=(
        InitialValue(
            vibrating.displacement,
            lambda x: np.sin(np.pi * x) + 0.5 * np.sin(5 * np.pi * x),
        ),
        InitialValue(vibrating.velocity, 0.0),
        InitialValue(vibrating.stress, lambda x: np.pi * np.cos(np.pi * x) + 2.5 * np.pi * np.cos(5 * np.pi * x)),
    ),
)
meshes = {n: StructuredLineMesher(cells=n).generate(rod) for n in (cells, coarse_cells)}
static_mesh = StructuredLineMesher(cells=static_cells).generate(long_rod)
rod_mesh = StructuredLineMesher(cells=rod_cells).generate(rod)
centres = {n: mesh.cell_centers[:, 0] for n, mesh in meshes.items()}
print(
    f"impact: {meshes[cells].cell_count} cells of width {1 / cells:.4f} "
    f"(mesh check: {meshes[coarse_cells].cell_count} cells); static bar: "
    f"{static_mesh.cell_count} elements on L = 2; vibrating rod: {rod_mesh.cell_count} elements; "
    f"wave speed c = {bar.wave_speed:.3f}, impedance rho*c = {bar.impedance:.3f}"
)

# ## The exact solutions
#
# Written out with NumPy: the static parabola, the d'Alembert solution of the impact at $t=1.5$
# (incident front plus its reflection from the free end) and the two standing modes.

fine = np.linspace(0.0, 1.0, 2001)
exact = {
    key: np.array(
        (
            np.where(points < 0.5, -1.0, 0.0),
            np.where(points < 0.5, 1.0, 2.0),
            np.where(points < 0.5, end_time - points, 1.0),
        )
    )
    for key, points in {**centres, "fine": fine}.items()
}
static_points = np.linspace(0.0, 2.0, 801)
static_exact = (
    1.0 * static_points + 0.5 * static_points * (2 - static_points / 2),
    1.0 + 0.5 * (2 - static_points),
)
rod_end_time = 4.0
rod_points = np.linspace(0.0, 1.0, 401)
rod_exact = np.sin(np.pi * rod_points) * np.cos(np.pi * rod_end_time) + 0.5 * np.sin(
    5 * np.pi * rod_points
) * np.cos(5 * np.pi * rod_end_time)
print(
    f"impact at t = {end_time}: front of the reflected wave at x = {2 - end_time:.2f}; "
    f"rod: omega_1 = {np.pi:.4f}, omega_5 = {5 * np.pi:.4f}, two periods of mode 1 by t = {rod_end_time}"
)

# ## The static bar: element degree × linear solver
#
# The stationary displacement equation $K u=F$ with the clamped end eliminated symmetrically, so
# the stiffness matrix stays symmetric positive definite and conjugate gradients apply. Quadratic
# and cubic elements contain the exact parabola; linear elements are exact at the nodes.

static_solvers = {
    "sparse LU": SparseDirect(),
    "CG": ConjugateGradient(maximum_iterations=4000),
    "GMRES(400)": GMRES(restart=400, maximum_iterations=4000),
    "GMRES + ILU": GMRES(preconditioner=ILU()),
    "GMRES + AMG": GMRES(preconditioner=AlgebraicMultigrid()),
    "right GMRES(400)": RightGMRES(restart=400, maximum_iterations=4000),
    "automatic": AutomaticLinearSolver(),
}
static_runs = [
    dict(degree=degree, column=name, solver=solver, cells=n)
    for degree in (1, 2, 3)
    for name, solver in static_solvers.items()
    for n in (static_cells, static_cells // 2)
]
for run in static_runs:
    started = time.perf_counter()
    run.update(status="rejected", note="", errors=(np.nan, np.nan))
    try:
        simulation = Simulation(
            problem=static,
            mesh=static_mesh if run["cells"] == static_cells else StructuredLineMesher(cells=run["cells"]).generate(long_rod),
            numerics=Numerics(
                spatial=FiniteElement(element=Lagrange(run["degree"])),
                linear_solver=run["solver"],
            ),
        )
        check = simulation.validate()
        if not check.valid:
            run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
        else:
            result = simulation.solve_stationary()
            samples = np.column_stack((static_points, np.zeros((len(static_points), 2))))
            run["profiles"] = (
                result.field(loaded.displacement).evaluate(samples).reshape(-1),
                result.field(loaded.stress).evaluate(samples).reshape(-1),
            )
            run["errors"] = tuple(
                np.abs(value - reference).max()
                for value, reference in zip(run["profiles"], static_exact)
            )
            run["status"] = "ran"
    except Exception as error:
        run["note"] = str(error).splitlines()[0]
    run["seconds"] = time.perf_counter() - started
    print(
        f"static P{run['degree']} {run['column']:20s} {run['cells']:4d} {run['status']:8s} "
        + (
            f"max |u error| {run['errors'][0]:.2e}  max |stress error| {run['errors'][1]:.2e}"
            if run["status"] == "ran"
            else run["note"][:90]
        )
    )

static_grid = {(r["degree"], r["column"]): r for r in static_runs if r["cells"] == static_cells}
fig = plots.figure(1, 2, size=(16, 4.5))
for panel, index, label in ((fig[0], 0, "displacement"), (fig[1], 1, "stress")):
    plots.matrix(
        [
            [
                static_grid[(degree, c)]["errors"][index] + 1e-16
                if static_grid[(degree, c)]["status"] == "ran"
                else np.nan
                for c in static_solvers
            ]
            for degree in (1, 2, 3)
        ],
        [f"FEM CG P{degree}" for degree in (1, 2, 3)],
        list(static_solvers),
        scale="log",
        label=f"max |{label} error|",
        text=[
            [
                f"{static_grid[(degree, c)]['errors'][index]:.1e}"
                if static_grid[(degree, c)]["status"] == "ran"
                else static_grid[(degree, c)]["status"]
                for c in static_solvers
            ]
            for degree in (1, 2, 3)
        ],
        title=f"Static bar, {static_cells} elements: maximum {label} error",
        ax=panel,
    )
plots.show()

# Mesh check for the static bar: linear elements converge in the stress at first order (the
# projected stress of a linear element is piecewise constant before projection); quadratic and
# cubic elements reproduce the exact solution on both meshes up to round-off.

for degree in (1, 2, 3):
    pair = [
        r
        for r in static_runs
        if r["degree"] == degree and r["column"] == "sparse LU" and r["status"] == "ran"
    ]
    print(
        f"P{degree}: "
        + ", ".join(
            f"{r['cells']} elements: max |u err| {r['errors'][0]:.2e}, max |stress err| {r['errors'][1]:.2e}"
            for r in pair
        )
    )

# ## The method catalogue for the impact
#
# The velocity-stress system is linear, so every Riemann solver is exact on it once the signal
# speeds are $\pm c$: Rusanov and HLL coincide with the Godunov flux for velocity and stress; HLLC
# additionally restores the stationary middle wave across which only the displacement jumps, so it
# carries no numerical diffusion on $u$. The catalogue is the one of the Sod notebook: slope
# limiters, ENO/WENO reconstructions, discontinuous elements of degree 0 to 4 with and without
# limiting or jump penalty, continuous elements with a Riemann flux and graph viscosity, interior
# penalty, SUPG, GLS or no stabilization, and finite differences including summation-by-parts.
# The DG jump penalty $\gamma/h\,[u][v]$ uses $\gamma=c\,\Delta x/2$, the scale of one upwind flux.
# The graph viscosity takes `ElasticBarInvariantDomain`: every finite state of a linear elastic bar
# is admissible.

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
    "WENO3-Z": Weno3Reconstructor(
        weights=WenoZWeights(global_indicator=SmoothnessDifference())
    ),
    "WENO5-Z": Weno5Reconstructor(
        weights=WenoZWeights(global_indicator=SmoothnessDifference())
    ),
    "CWENO3": CentralWeightedENO(weights=JiangShuWeights()),
    "CWENO3-Z": CentralWeightedENO(
        weights=WenoZWeights(global_indicator=SmoothnessDifference(first=1, last=2))
    ),
    "QUICK": QUICK(),
}
graph_viscosity = InvariantDomainGraphViscosity(
    mass=RowSumLumping(), invariant_domain=ElasticBarInvariantDomain()
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
    for reconstruction_name, reconstruction in reconstructions.items():
        spatial[(f"FVM {reconstruction_name}", flux_name)] = FiniteVolume(
            convection=flux, reconstruction=reconstruction
        )
    spatial[("FEM DG P0", flux_name)] = FiniteElement(
        element=DiscontinuousLagrange(0), convection=flux
    )
    for degree in (1, 2, 3, 4):
        spatial[(f"FEM DG P{degree}, unlimited", flux_name)] = FiniteElement(
            element=DiscontinuousLagrange(degree), convection=flux
        )
        spatial[(f"FEM DG P{degree} + minmod", flux_name)] = FiniteElement(
            element=DiscontinuousLagrange(degree),
            convection=flux,
            limiter=DGPolynomialLimiter(slope_limiter=MinmodLimiter()),
        )
    spatial[("FEM DG P1 + jump penalty", flux_name)] = FiniteElement(
        element=DiscontinuousLagrange(1),
        convection=flux,
        stabilization=DGJumpPenalty(coefficient=0.5 / cells),
    )
    for degree in (1, 2, 3):
        spatial[(f"FEM CG P{degree} + graph viscosity", flux_name)] = FiniteElement(
            element=Lagrange(degree), convection=flux, stabilization=graph_viscosity
        )
        spatial[(f"FEM CG P{degree} Galerkin", flux_name)] = FiniteElement(
            element=Lagrange(degree), convection=flux
        )
    for degree, penalty in ((1, 0.1), (2, 0.01)):
        spatial[(f"FEM CG P{degree} + continuous interior penalty", flux_name)] = (
            FiniteElement(
                element=Lagrange(degree),
                convection=flux,
                stabilization=CIP(penalty=penalty),
                initialization=NodalInterpolation(),
            )
        )
    spatial[("FEM CG P1 + volume interior penalty", flux_name)] = FiniteElement(
        element=Lagrange(1),
        convection=flux,
        stabilization=VolumeCIP(diffusivity=2e-3),
        initialization=NodalInterpolation(),
    )
    for name, kind in (("SUPG", SUPG), ("GLS", GLS)):
        spatial[(f"FEM CG P1 + {name}", flux_name)] = FiniteElement(
            element=Lagrange(1),
            convection=flux,
            stabilization=kind(tau=0.5 / cells, strong_residual=InterpolatedFluxDivergence()),
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
                derivative=SummationByPartsDifference(accuracy_order=order),
                boundary_flux=flux,
            )
        )
spatial[("FDM WENO5 flux splitting", "Lax-Friedrichs splitting")] = FiniteDifference(
    gradient=WENOFluxDifference(
        flux_splitting=GlobalLaxFriedrichsSplit(), reconstruction=WENO5Flux()
    )
)
# The second-order displacement formulation: continuous elements without an interface flux.
for degree in (1, 2, 3):
    spatial[(f"FEM CG P{degree} displacement", "second-order form")] = FiniteElement(
        element=Lagrange(degree)
    )

# Time integrators: the generic catalogue of the Sod notebook plus the structural integrators for
# second-order systems. `Newmark()` is the average-acceleration (trapezoidal) rule,
# `Newmark(beta=1/6)` the linear-acceleration rule (conditionally stable), `Newmark(beta=0)` the
# explicit central difference and `Newmark(beta=0.3025, gamma=0.6)` a numerically damped variant;
# the kick-drift methods (symplectic Euler, Stoermer-Verlet) advance the displacement with the new
# velocity and need the same position/velocity pair;
# HHT-α and generalized-α damp the highest modes with second-order accuracy, controlled by α and by
# the spectral radius at infinity.

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
    **{f"BDF{k} (Gear)": TimeMethod(f"bdf{k}_gear") for k in (2, 3, 4, 5, 6)},
    "Rosenbrock-Euler": TimeMethod("rosenbrock_euler"),
    "Rosenbrock-Wanner Euler": TimeMethod("rosenbrock_wanner_euler"),
    "adaptive BDF": AdaptiveBDF(relative_tolerance=1e-3, absolute_tolerance=1e-6),
    "symplectic Euler (kick-drift)": TimeMethod("symplectic_euler"),
    "Stoermer-Verlet (kick-drift)": TimeMethod("staggered_verlet"),
    "Newmark, average acceleration": Newmark(),
    "Newmark, linear acceleration": Newmark(beta=1 / 6),
    "Newmark, central difference": Newmark(beta=0.0),
    "Newmark, damped (gamma = 0.6)": Newmark(beta=0.3025, gamma=0.6),
    "HHT-alpha (alpha = -0.05)": HHTAlpha(alpha=-0.05),
    "HHT-alpha (alpha = -1/3)": HHTAlpha(alpha=-1 / 3),
    **{
        f"generalized-alpha (rho_inf = {rho})": GeneralizedAlpha(spectral_radius=rho)
        for rho in (1.0, 0.8, 0.5, 0.0)
    },
}
structural = {name for name, item in integrators.items() if getattr(item, "second_order", False)}
second_order_only = structural | {
    "symplectic Euler (kick-drift)",
    "Stoermer-Verlet (kick-drift)",
}
newton = Newton(linear_solver=SparseDirect(), jacobian="colored_finite_difference")
implicit_step = 5e-3

# ## Schemes that must fail, and their stabilized counterparts
#
# The undamped displacement formulation has a purely imaginary spectrum $\pm i\omega_k$. An
# explicit method whose stability region contains no piece of the imaginary axis (forward Euler,
# two-stage Runge-Kutta, Adams-Bashforth 2) amplifies every mode at every step, and the library
# refuses it before stepping; BDF3 to BDF6 and Adams-Moulton 3 lose stability on the part of the
# imaginary axis reached by the highest element modes at $\Delta t=5\cdot10^{-3}$. Their stabilized
# counterpart adds exactly one ingredient: a volume viscosity $\rho\nu\,\partial_x^2 v$ with
# $\nu=c\,\Delta x/2$ (`VolumeCIP` on the displacement element), which moves the spectrum into the
# left half-plane. Adams-Moulton 4 has a bounded stability region and runs instead at the step
# $\Delta t=0.1\,\Delta x/c$ it is stable for. On the velocity-stress system the Riemann
# flux already dissipates; there leapfrog (no damping at all) is unstable and takes the
# Robert-Asselin-Williams filter as in the Sod notebook. Fixed-point stage iterations contract
# only below the inverse spectral radius and run at a smaller step.

viscous = {
    degree: FiniteElement(
        element=Lagrange(degree), stabilization=VolumeCIP(diffusivity=0.5 / cells)
    )
    for degree in (1, 2, 3)
}
stabilized_time = {
    "leapfrog": (
        "leapfrog + Robert-Asselin-Williams filter (ν = 0.2, α = 0.53; first order), safety factor 0.15",
        TimeMethod("leapfrog_raw"),
        0.15,
    ),
}
stable_step = {"fixed point": 5e-4, "fixed point + Aitken": 5e-4}
imaginary_unstable = {
    "forward Euler",
    "SSP-RK2",
    "Heun",
    "explicit midpoint",
    "Adams-Bashforth 2",
    "Heun-Euler (embedded)",
    "Adams-Moulton 3",
    "BDF3 (variable step)",
    *{f"BDF{k} (Gear)" for k in (3, 4, 5, 6)},
}
bounded_implicit = {"Adams-Moulton 4"}
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
        (name, linear_name): kind(linear_solver=linear, jacobian="colored_finite_difference")
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
structural_solvers = {
    "sparse LU": SparseDirect(),
    "CG": ConjugateGradient(maximum_iterations=4000),
    "GMRES(400)": GMRES(restart=400, maximum_iterations=4000),
    "GMRES + ILU": GMRES(preconditioner=ILU()),
    "GMRES + AMG": GMRES(preconditioner=AlgebraicMultigrid()),
    "right GMRES(400)": RightGMRES(restart=400, maximum_iterations=4000),
    "automatic": AutomaticLinearSolver(),
}

# ## The test matrix
#
# 1. **Space × Riemann solver**, SSP-RK3 at the stable step (safety factor 0.5) for the
#    velocity-stress schemes; the displacement elements run with classical RK4 at their spectral
#    step in the extra column "second-order form".
# 2. **Time integrator × space**: every integrator on first-order and MUSCL-minmod volumes with
#    HLLC and on linear and quadratic displacement elements. Explicit methods at their stable step,
#    implicit generic and structural methods at $\Delta t=5\cdot10^{-3}$ (two cells per step).
# 3. **Nonlinear × linear solver**: backward Euler on MUSCL-minmod-HLLC, $\Delta t=5\cdot10^{-3}$.
# 4. **Structural integrator × linear solver**: the average-acceleration Newmark rule on quadratic
#    displacement elements; its effective matrix $M+\beta\Delta t^2K$ is symmetric positive definite.
# 5. **Vibrating rod**: element degree × structural integrator, $\Delta t=10^{-2}$, two periods of
#    the fundamental mode.
#
# Representative combinations are repeated on 200 cells for the mesh check.

time_spatial = {
    "FVM first order, HLLC": spatial[("FVM first order", "HLLC")],
    "FVM MUSCL + minmod, HLLC": spatial[("FVM MUSCL + minmod", "HLLC")],
    "FEM CG P1 displacement": spatial[("FEM CG P1 displacement", "second-order form")],
    "FEM CG P2 displacement": spatial[("FEM CG P2 displacement", "second-order form")],
}
runs = [
    dict(
        matrix="space x Riemann solver",
        row=row,
        column=column,
        cells=cells,
        spatial=scheme,
        time=ClassicalRungeKutta4() if column == "second-order form" else TimeMethod("ssprk3"),
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
            and name not in structural
            and "rosenbrock" not in getattr(integrator, "method_id", "")
            else None
        ),
        linear=(
            SparseDirect()
            if "rosenbrock" in getattr(integrator, "method_id", "") or name in structural
            else None
        ),
        step=(
            implicit_step
            if getattr(integrator, "implicit", False)
            and getattr(integrator, "unconditionally_stable", True)
            else end_time
        ),
    )
    for name, integrator in integrators.items()
    for space, scheme in time_spatial.items()
    if name not in second_order_only or "displacement" in space
]
runs += [
    dict(
        matrix="nonlinear x linear solver",
        row=row,
        column=column,
        cells=cells,
        spatial=time_spatial["FVM MUSCL + minmod, HLLC"],
        time=BackwardEuler(),
        nonlinear=solver,
        step=implicit_step,
    )
    for (row, column), solver in nonlinear_solvers.items()
]
runs += [
    dict(
        matrix="structural integrator x linear solver",
        row="Newmark, average acceleration",
        column=name,
        cells=cells,
        spatial=time_spatial["FEM CG P2 displacement"],
        time=Newmark(),
        nonlinear=None,
        linear=solver,
        step=implicit_step,
    )
    for name, solver in structural_solvers.items()
]
runs += [
    {
        **run,
        "variant": "stabilized",
        "spatial": viscous[int(run["column"].split("P")[1][0])],
        "note_stabilized": "+ volume viscosity ν = cΔx/2 (VolumeCIP)",
    }
    for run in list(runs)
    if run["matrix"] == "time integrator x space"
    and "displacement" in run["column"]
    and run["row"] in imaginary_unstable
]
runs += [
    {
        **run,
        "variant": "stabilized",
        "step": 0.1 / cells,
        "note_stabilized": f"same method at Δt = {0.1 / cells:.1e}",
    }
    for run in list(runs)
    if run["matrix"] == "time integrator x space"
    and "displacement" in run["column"]
    and run["row"] in bounded_implicit
    and run.get("variant") is None
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
    if run["matrix"] == "time integrator x space"
    and run["row"] in stabilized_time
    and "FVM" in run["column"]
    and run.get("variant") is None
]
runs += [
    {
        **run,
        "variant": "stabilized",
        "step": stable_step[run["row"]],
        "note_stabilized": f"{run['row']} at Δt = {stable_step[run['row']]:.1e}",
    }
    for run in list(runs)
    if run["matrix"] == "nonlinear x linear solver"
    and run["row"] in stable_step
    and run.get("variant") is None
]
representative = [
    ("FVM first order", "Rusanov"),
    ("FVM first order", "HLLC"),
    ("FVM MUSCL + minmod", "HLLC"),
    ("FVM WENO5", "HLLC"),
    ("FEM DG P2 + minmod", "HLLC"),
    ("FEM CG P1 + graph viscosity", "HLLC"),
    ("FDM summation by parts, order 4", "HLLC"),
    ("FEM CG P2 displacement", "second-order form"),
]
runs += [
    {**run, "matrix": "mesh check", "cells": coarse_cells}
    for run in runs
    if run["matrix"] == "space x Riemann solver"
    and (run["row"], run["column"]) in representative
]
print(
    f"{len(runs)} impact runs: "
    + ", ".join(
        f"{sum(r['matrix'] == m for r in runs)} {m}"
        for m in dict.fromkeys(r["matrix"] for r in runs)
    )
)

# ## Run everything
#
# One loop, the same problem object every time. A combination the library refuses is recorded with
# its validation message; a run whose method is unstable for the operator, or whose stage iteration
# does not converge, is recorded as failed.

for run in runs:
    simulation = Simulation(
        problem=impact,
        mesh=meshes[run["cells"]],
        numerics=Numerics(
            spatial=run["spatial"],
            linear_solver=(
                run.get("linear")
                if run.get("linear") is not None
                else (
                    SparseDirect()
                    if isinstance(run["spatial"], FiniteElement) and run["nonlinear"] is None
                    else None
                )
            ),
            time_integrator=run["time"],
            nonlinear_solver=run["nonlinear"],
            step_limits=TimeStepLimits(stability_safety_factor=run.get("safety", 0.5)),
        ),
    )
    started = time.perf_counter()
    check = simulation.validate()
    run.update(status="rejected", note="", errors=(np.nan,) * 3, steps=0)
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = simulation.run(start_time=0.0, end_time=end_time, time_step=run["step"])
            points = np.column_stack((centres[run["cells"]], np.zeros((run["cells"], 2))))
            run["profiles"] = np.array(
                [
                    result.field(field).evaluate(points).reshape(-1)
                    for field in (bar.stress, bar.velocity, bar.displacement)
                ]
            )
            run["errors"] = tuple(
                np.abs(run["profiles"] - exact[run["cells"]]).sum(axis=1) / run["cells"]
            )
            run["steps"] = len(result.times) - 1
            run["status"] = (
                "ran"
                if np.isfinite(run["profiles"]).all() and run["errors"][0] < 1.0
                else "failed"
            )
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "failed"
                if "unstable" in message
                or "finite" in message
                or "did not converge" in message
                or "no stable region" in message
                else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    print(
        f"{run['matrix'][:12]:12s} {run['row'][:34]:34s} {run['column'][:24]:24s} "
        f"{(run.get('variant') or '')[:4]:4s} {run['status']:8s} {run['seconds']:7.1f} s  "
        + (
            f"L1 stress {run['errors'][0]:.2e}  L1 u {run['errors'][2]:.2e}"
            if run["status"] == "ran"
            else run["note"][:90]
        )
    )

# ## Space × Riemann solver
#
# The stress error of every spatial scheme with every Riemann solver. Linear schemes of higher
# than first order (unlimited MUSCL and QUICK, unlimited DG, Galerkin CG, central and SBP finite
# differences) overshoot at the two fronts, as Godunov's theorem demands, but a linear elastic
# state has no positivity limit, so they run; the overshoot shows in the error and in the
# profiles below.

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

matrix = [r for r in runs if r["matrix"] == "space x Riemann solver" and r.get("variant") is None]
space_key = "space x Riemann solver"
rows = list(dict.fromkeys(r["row"] for r in matrix))
columns = [*riemann, "Lax-Friedrichs splitting", "second-order form"]
grid = {(r["row"], r["column"]): r for r in matrix}
fig = plots.figure(1, 1, size=(14, 22))
plots.matrix(
    [
        [
            cell[(space_key, row, c)]["errors"][0] if (space_key, row, c) in cell else np.nan
            for c in columns
        ]
        for row in rows
    ],
    rows,
    columns,
    scale="log",
    label=f"L1 stress error at t = {end_time}",
    text=[
        [
            (
                f"{cell[(space_key, row, c)]['errors'][0]:.2e}"
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
    title=f"Space x Riemann solver, {cells} cells, SSP-RK3 (displacement elements: RK4)",
    ax=fig[0],
)
plots.show()

# All three measures for the combinations that ran, sorted by the stress error.

print(
    f"{'scheme':40s} {'Riemann solver':24s} {'L1 stress':>10s} {'L1 v':>10s} {'L1 u':>10s} "
    f"{'steps':>6s} {'time':>8s}"
)
for r in sorted((r for r in matrix if r["status"] == "ran"), key=lambda r: r["errors"][0]):
    print(
        f"{r['row']:40s} {r['column']:24s} {r['errors'][0]:10.2e} {r['errors'][1]:10.2e} "
        f"{r['errors'][2]:10.2e} {r['steps']:6d} {r['seconds']:7.1f}s"
    )

# ## Profiles of representative combinations
#
# Stress, velocity and displacement at $t=1.5$ against the d'Alembert solution, and a close-up of
# the reflected front.

shown = [r for r in matrix if (r["row"], r["column"]) in representative and r["status"] == "ran"]
fig = plots.figure(2, 2, size=(15, 9))
for panel, index, name, window in (
    (fig[0], 0, "stress", None),
    (fig[1], 1, "velocity", None),
    (fig[2], 2, "displacement", None),
    (fig[3], 0, "stress at the reflected front", (0.4, 0.6)),
):
    plots.line(
        y={
            f"{r['row']}, {r['column']}": (centres[cells], r["profiles"][index])
            for r in shown
        },
        reference={"d'Alembert": (fine, exact["fine"][index])},
        xlabel="position x / L",
        ylabel=name.split(" ")[0],
        title=f"{name.capitalize()} at t = {end_time}",
        xlimits=window,
        legend="lower right" if index == 0 and window is None else False,
        ax=panel,
    )
plots.show()

# ## Time integrator × space
#
# A dash marks a combination outside this cut: the structural and kick-drift integrators need a
# second-order (displacement, velocity) system and do not apply to the first-order velocity-stress
# volumes.

timing = [r for r in runs if r["matrix"] == "time integrator x space" and r.get("variant") is None]
time_key = "time integrator x space"
time_rows, time_columns = list(integrators), list(time_spatial)
time_hatched = [[(time_key, row, c) in hatch for c in time_columns] for row in time_rows]
time_notes = [
    "Hatched: the declared integrator is unstable for this operator; colour and value show its "
    "stabilized counterpart:",
    "  displacement elements: + volume viscosity ν = cΔx/2 (VolumeCIP), the integrator unchanged;",
    f"  Adams-Moulton 4 (bounded stability region) on displacement elements: Δt = {0.1 / cells:.1e}.",
] + [
    f"  {row}, volumes: {description}."
    for row, (description, _, _) in stabilized_time.items()
    if any((time_key, row, c) in hatch for c in time_columns)
]
fig = plots.figure(1, 2, size=(18, 16))
for panel, measure, label, title in (
    (fig[0], "errors", "L1 stress error", "L1 stress error"),
    (fig[1], "seconds", "wall time [s]", "Wall time and accepted steps"),
):
    plots.matrix(
        [
            [
                (
                    (cell[(time_key, row, c)]["errors"][0] if measure == "errors" else cell[(time_key, row, c)]["seconds"])
                    if (time_key, row, c) in cell and cell[(time_key, row, c)]["status"] == "ran"
                    else np.nan
                )
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
                        f"{cell[(time_key, row, c)]['errors'][0]:.2e}"
                        if measure == "errors"
                        else f"{cell[(time_key, row, c)]['seconds']:.0f} s, {cell[(time_key, row, c)]['steps']} steps"
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
        notes=time_notes if measure == "errors" else None,
        title=title,
        ax=panel,
    )
plots.show()

# ## Nonlinear × linear solver
#
# Backward Euler on MUSCL-minmod-HLLC. The discrete solution does not depend on the solver as long
# as each one converges; the wall time does.

solving = [r for r in runs if r["matrix"] == "nonlinear x linear solver" and r.get("variant") is None]
solver_key = "nonlinear x linear solver"
solver_rows = list(dict.fromkeys(r["row"] for r in solving))
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
                    f"{cell[(solver_key, row, c)]['seconds']:.0f} s\n"
                    f"{cell[(solver_key, row, c)]['errors'][0]:.3e}"
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
        "Hatched: the fixed-point (Picard) iteration cannot contract at Δt = 5·10⁻³; colour and "
        "value show the same iteration at the step it converges for:"
    ]
    + [
        f"  {row}: Δt = {stable_step[row]:.1e}."
        for row in stable_step
        if any((solver_key, row, c) in hatch for c in solver_columns)
    ],
    title="Backward Euler: wall time and L1 stress error per solver pair",
    ax=fig[0],
)
plots.show()

# ## Structural integrator × linear solver
#
# The Newmark effective matrix on quadratic displacement elements, solved by direct, Krylov and
# multigrid solvers. Conjugate gradients apply because the matrix is symmetric positive definite.

structural_runs = [r for r in runs if r["matrix"] == "structural integrator x linear solver"]
print(f"{'linear solver':22s} {'status':8s} {'L1 stress':>10s} {'L1 u':>10s} {'time':>8s}")
for r in structural_runs:
    print(
        f"{r['column']:22s} {r['status']:8s} "
        + (
            f"{r['errors'][0]:10.3e} {r['errors'][2]:10.3e} {r['seconds']:7.1f}s"
            if r["status"] == "ran"
            else r["note"][:80]
        )
    )

# ## The vibrating rod: element degree × structural integrator
#
# Two clamped-end modes, $n=1$ and $n=5$, released from rest and followed for two periods of the
# fundamental mode with $\Delta t=10^{-2}$ ($\omega_5\Delta t\approx0.16$). The trapezoidal rule and
# generalized-α with $\rho_\infty=1$ conserve the modal energy; HHT-α and generalized-α with
# $\rho_\infty<1$ damp the fifth mode slightly; the central difference runs at its stability limit
# $\Delta t\le2/\omega_{max}$ of the discrete operator.

rod_integrators = {name: integrators[name] for name in integrators if name in second_order_only}
rod_runs = [
    dict(degree=degree, row=name, time=integrator)
    for name, integrator in rod_integrators.items()
    for degree in (1, 2, 3)
]
for run in rod_runs:
    started = time.perf_counter()
    run.update(status="rejected", note="", error=np.nan, steps=0)
    try:
        result = Simulation(
            problem=vibration,
            mesh=rod_mesh,
            numerics=Numerics(
                spatial=FiniteElement(element=Lagrange(run["degree"])),
                time_integrator=run["time"],
                linear_solver=SparseDirect(),
                step_limits=TimeStepLimits(stability_safety_factor=0.9),
            ),
        ).run(start_time=0.0, end_time=rod_end_time, time_step=1e-2)
        values = result.field(vibrating.displacement).evaluate(
            np.column_stack((rod_points, np.zeros((len(rod_points), 2))))
        ).reshape(-1)
        run["error"] = np.abs(values - rod_exact).max()
        run["steps"] = len(result.times) - 1
        run["status"] = "ran"
    except Exception as error:
        run["note"] = str(error).splitlines()[0]
    run["seconds"] = time.perf_counter() - started
    print(
        f"rod P{run['degree']} {run['row'][:36]:36s} {run['status']:8s} {run['seconds']:6.1f} s  "
        + (
            f"max |u error| {run['error']:.2e}, {run['steps']} steps"
            if run["status"] == "ran"
            else run["note"][:90]
        )
    )
rod_grid = {(r["row"], r["degree"]): r for r in rod_runs}
fig = plots.figure(1, 1, size=(11, 7))
plots.matrix(
    [
        [rod_grid[(name, degree)]["error"] for degree in (1, 2, 3)]
        for name in rod_integrators
    ],
    list(rod_integrators),
    [f"FEM CG P{degree}" for degree in (1, 2, 3)],
    scale="log",
    label=f"max |u error| at t = {rod_end_time}",
    text=[
        [
            (
                f"{rod_grid[(name, degree)]['error']:.1e}, {rod_grid[(name, degree)]['steps']} steps"
                if rod_grid[(name, degree)]["status"] == "ran"
                else rod_grid[(name, degree)]["status"]
            )
            for degree in (1, 2, 3)
        ]
        for name in rod_integrators
    ],
    title=f"Vibrating rod, {rod_cells} elements: structural integrator x element degree",
    ax=fig[0],
)
plots.show()

# ## Mesh check: 400 against 200 cells
#
# The representative combinations on half as many cells. For the discontinuous fronts the $L_1$
# error falls roughly like $\Delta x^{1/2}$ (first order) to $\Delta x^{2/3}$ (second order).

coarse = {(r["row"], r["column"]): r for r in runs if r["matrix"] == "mesh check"}
print(
    f"{'scheme':40s} {'Riemann solver':24s} {'L1 stress 200':>14s} {'L1 stress 400':>14s} {'ratio':>6s}"
)
for key, r in coarse.items():
    fine_run = grid[key]
    if r["status"] == "ran" and fine_run["status"] == "ran":
        print(
            f"{key[0]:40s} {key[1]:24s} {r['errors'][0]:14.3e} {fine_run['errors'][0]:14.3e} "
            f"{r['errors'][0] / fine_run['errors'][0]:6.2f}"
        )

# ## What the library refused, and what failed
#
# A refused combination is caught by validation or binding before any time step; a failed one
# started, or was refused as unstable for this operator, and has its stabilized counterpart above.

everything = runs + static_runs + rod_runs
for status in ("rejected", "failed"):
    selected = [r for r in everything if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(
            f"  {r.get('matrix', 'static/rod')[:12]:12s} {str(r.get('row', r.get('degree')))[:34]:34s} "
            f"{str(r.get('column', ''))[:24]:24s} {(r.get('variant') or '')[:4]:4s} {r['note'][:100]}"
        )
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in everything)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(everything)} runs, "
    + f"{sum(r['seconds'] for r in everything) / 60:.0f} min in total"
)

# ## Methods that do not apply to the elastic bar
#
# * The **structural and kick-drift integrators** (Newmark, HHT-α, generalized-α, symplectic
#   Euler, Stoermer-Verlet) advance a second-order system
#   $M\ddot u+C\dot u+Ku=F$; the velocity-stress volumes, differences and DG elements discretize a
#   first-order system and have no mass-stiffness pair, so the library refuses the combination
#   with that reason.
# * The **static bar** is posed in the displacement; the velocity-stress system does not contain
#   the displacement as an independent unknown at rest ($v$ is constant and $\sigma$ follows from
#   one traction condition), so the stationary cut uses the displacement elements only. The
#   volume viscosity acts on the velocity and is not consumed by a stationary solve.
# * **Conjugate gradients** need a symmetric positive definite matrix and preconditioner: the
#   stage matrix of a generic implicit method on the first-order system is not symmetric, so CG
#   appears only for the static stiffness and the Newmark effective matrix, and there without the
#   (nonsymmetric) ILU and two-level AMG preconditioners, which run with GMRES instead.
# * Positivity limiters and hydrostatic reconstruction protect a positive density or depth that
#   the elastic bar does not have; Scharfetter-Gummel fluxes are for drift-diffusion;
#   Runge-Kutta-Chebyshev and -Legendre super-steps are for parabolic operators; IMEX splits need a
#   stiff and a non-stiff part.
#
# ## Going further
#
# Swap the free end for `BarFixedEnd` to reflect the front with the same sign, give the bar a
# `damping` rate, or prescribe a time-dependent end velocity `value=lambda t: ...`; the matrix code
# stays as it is.
