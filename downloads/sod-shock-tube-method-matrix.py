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
# (finite volumes, finite differences, discontinuous and continuous finite elements, and smoothed
# particle hydrodynamics), the reconstruction and its limiter, the Riemann solver, the time integrator and, for implicit steps,
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
from onesim.numerics.discretization import NodalInterpolation
from onesim.numerics.fdm import (
    BoundaryFluxDifference,
    CentralDifference,
    DissipativeFluxDifference,
    FiniteDifference,
    JSTDissipation,
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
    InterpolatedFluxDivergence,
    VolumeCIP,
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
    CentralWeightedENO,
    JiangShuWeights,
    SmoothnessDifference,
    WenoZWeights,
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
from onesim.numerics.discretization.spatial.execution.backends.boundaries.boundary_particle import (
    SPHParticleCloud,
)
from onesim.physics.hydrodynamics.incompressible.binding.runtime.particle_components import (
    SPHDiscretization,
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
    HydrodynamicPublicApi,
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
# Four Riemann solvers; seven slope limiters plus the unlimited slope; higher-order reconstructions
# including WENO3, WENO5, their WENO-Z variants and central WENO (CWENO3, CWENO3-Z); finite elements
# (FEM) with discontinuous elements of degree 0 to 4, with and without limiting or a jump penalty,
# and continuous elements of degree 1 to 3 with graph viscosity, continuous interior penalty, volume
# interior penalty, SUPG, GLS or no stabilization; and finite-difference stencils including
# summation-by-parts operators of order 2 and 4. Each is an object; the matrix is built from them
# with plain dictionaries.
#
# Limited discontinuous Galerkin runs with all four Riemann solvers for degrees 1 to 4; the limiter
# is minmod on the slopes plus the Euler invariant domain (positive density and pressure at every
# node).

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
    spatial[("FEM DG P0", flux_name)] = FiniteElement(
        element=DiscontinuousLagrange(0), convection=flux
    )
    for degree in (1, 2, 3, 4):
        spatial[(f"FEM DG P{degree}, unlimited", flux_name)] = FiniteElement(
            element=DiscontinuousLagrange(degree), convection=flux
        )
        spatial[(f"FEM DG P{degree} + minmod + invariant domain", flux_name)] = FiniteElement(
            element=DiscontinuousLagrange(degree), convection=flux, limiter=dg_limiter
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
    spatial[("FEM DG P1 + jump penalty", flux_name)] = FiniteElement(
        element=DiscontinuousLagrange(1),
        convection=flux,
        stabilization=DGJumpPenalty(coefficient=1.0),
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
# ## Schemes that must fail at a shock, and their stabilized counterparts
#
# Some declared schemes are linear and of higher than first order: MUSCL and QUICK without a
# limiter, discontinuous Galerkin without limiting or with only a jump penalty, continuous Galerkin
# without stabilization or with a linear SUPG/GLS term, and central or summation-by-parts finite
# differences. By Godunov's theorem every such scheme oscillates at a discontinuity;
# on Sod's problem the oscillation drives density or pressure negative. The library reports that
# honestly instead of clipping the state. For each of these schemes the matrix also runs the
# stabilized counterpart, which adds exactly one ingredient; its result is shown behind a grey
# hatching in the same cell.

invariant_dg = DGPolynomialLimiter(invariant_domain=EulerInvariantDomain())
stabilized_space = {
    "FVM MUSCL, unlimited": (
        "MUSCL unlimited + invariant-domain positivity limiter at the faces",
        lambda flux: FiniteVolume(
            convection=flux, reconstruction=MUSCL(), positivity=ConvexInvariantDomain()
        ),
    ),
    "FVM QUICK": (
        "QUICK + invariant-domain positivity limiter at the faces",
        lambda flux: FiniteVolume(
            convection=flux, reconstruction=QUICK(), positivity=ConvexInvariantDomain()
        ),
    ),
    **{
        f"FEM DG P{degree}, unlimited": (
            f"FEM DG P{degree} + Euler invariant-domain scaling (no slope limiter)",
            lambda flux, degree=degree: FiniteElement(
                element=DiscontinuousLagrange(degree), convection=flux, limiter=invariant_dg
            ),
        )
        for degree in (1, 2, 3, 4)
    },
    "FDM central, 2nd order": (
        "central differences + Jameson-Schmidt-Turkel artificial dissipation",
        lambda flux: FiniteDifference(
            gradient=DissipativeFluxDifference(
                flux_difference=BoundaryFluxDifference(
                    derivative=CentralDifference(), boundary_flux=flux
                ),
                dissipation=JSTDissipation(),
            )
        ),
    ),
    "FDM polynomial, 4th order": (
        "4th-order differences + Jameson-Schmidt-Turkel artificial dissipation",
        lambda flux: FiniteDifference(
            gradient=DissipativeFluxDifference(
                flux_difference=BoundaryFluxDifference(
                    derivative=PolynomialDifference(accuracy_order=4), boundary_flux=flux
                ),
                dissipation=JSTDissipation(),
            )
        ),
    ),
    **{
        f"FEM CG P1 + {name}": (
            f"CG P1 + invariant-domain graph viscosity instead of {name}",
            lambda flux: FiniteElement(
                element=Lagrange(1), convection=flux, stabilization=graph_viscosity
            ),
        )
        for name in ("SUPG", "GLS")
    },
    "FEM DG P1 + jump penalty": (
        "DG P1 + Euler invariant-domain scaling instead of the jump penalty",
        lambda flux: FiniteElement(
            element=DiscontinuousLagrange(1), convection=flux, limiter=invariant_dg
        ),
    ),
    **{
        f"FDM summation by parts, order {order}": (
            f"SBP order {order} with SAT boundary flux + Jameson-Schmidt-Turkel artificial dissipation",
            lambda flux, order=order: FiniteDifference(
                gradient=DissipativeFluxDifference(
                    flux_difference=BoundaryFluxDifference(
                        derivative=SummationByPartsDifference(accuracy_order=order),
                        boundary_flux=flux,
                    ),
                    dissipation=JSTDissipation(),
                )
            ),
        )
        for order in (2, 4)
    },
    **{
        f"FEM CG P{degree} Galerkin": (
            f"FEM CG P{degree} + invariant-domain graph viscosity"
            + (" (positive Bernstein basis)" if degree > 1 else ""),
            lambda flux, degree=degree: FiniteElement(
                element=Lagrange(degree), convection=flux, stabilization=graph_viscosity
            ),
        )
        for degree in (1, 2, 3)
    },
}

spatial[("FDM WENO5 flux splitting", "Lax-Friedrichs splitting")] = FiniteDifference(
    gradient=WENOFluxDifference(
        flux_splitting=GlobalLaxFriedrichsSplit(), reconstruction=WENO5Flux()
    )
)

# Time integrators: explicit one-step methods (forward Euler, Heun, explicit midpoint, SSP-RK2,
# SSP-RK3, RK4 and others) and multistep methods, implicit Runge-Kutta, Adams-Moulton
# and BDF methods, Rosenbrock methods and the adaptive integrators. Implicit stages take a Newton
# solver with a coloured finite-difference Jacobian and a sparse direct solve; the linearly implicit
# Rosenbrock stages take the sparse direct solve alone. The nonlinear and linear solvers are then
# varied on their own.

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
}
newton = Newton(linear_solver=SparseDirect(), jacobian="colored_finite_difference")
# BDF4 to BDF6 and Adams-Moulton 4 are not A-stable, and a fixed-point (Picard) iteration contracts
# only while the time step stays below the inverse spectral radius of the stage operator. At
# Δt = 2·10⁻³ they must fail; their stabilized counterpart is the same method at a step it is stable
# for, shown behind the hatching.
# Leapfrog has no damping: on the dissipative upwind operator it is unstable at every step. Its
# stabilized counterpart adds the Robert-Asselin-Williams time filter and runs at a reduced
# stability safety factor.
stabilized_time = {
    "leapfrog": (
        "leapfrog + Robert-Asselin-Williams filter (ν = 0.2, α = 0.53; first order), safety factor 0.15",
        TimeMethod("leapfrog_raw"),
        0.15,
    ),
}
stable_step = {
    "BDF4 (Gear)": 5e-4,
    "BDF5 (Gear)": 5e-4,
    "BDF6 (Gear)": 5e-4,
    "Adams-Moulton 4": 5e-4,
    "fixed point": 2.5e-4,
    "fixed point + Aitken": 2.5e-4,
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
#    $\Delta t=2\cdot10^{-3}$.
#
# A fourth cut, **SPH × time integrator**, follows below: the same problem as a particle method.
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
runs += [
    dict(
        matrix="space x Riemann solver",
        variant="stabilized",
        row=row,
        column=flux_name,
        cells=cells,
        spatial=make(flux),
        time=TimeMethod("ssprk2"),
        nonlinear=None,
        step=end_time,
        note_stabilized=description,
    )
    for row, (description, make) in stabilized_space.items()
    for flux_name, flux in riemann.items()
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
    if run["matrix"] in ("time integrator x space", "nonlinear x linear solver")
    and run["row"] in stable_step
    and run.get("variant") is None
]
representative = [
    ("FVM first order", "Rusanov"),
    ("FVM first order", "HLLC"),
    ("FVM MUSCL + minmod", "HLLC"),
    ("FVM MUSCL + superbee", "HLLC"),
    ("FVM WENO5", "HLLC"),
    ("FDM WENO5 flux splitting", "Lax-Friedrichs splitting"),
    ("FEM DG P2 + minmod + invariant domain", "HLLC"),
    ("FEM CG P1 + graph viscosity", "HLLC"),
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
            step_limits=TimeStepLimits(stability_safety_factor=run.get("safety", 0.5)),
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
                if "non-positive" in message
                or "finite" in message
                or "did not converge" in message
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

# Every cell shows the declared combination. Where it failed at the shock and a stabilized
# counterpart exists, the cell is hatched grey and its colour and number are those of the
# stabilized run; the legend below each matrix says what was added.

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
    "Hatched: the declared scheme fails at the shock (negative density or pressure, Godunov's "
    "theorem); colour and value show its stabilized counterpart:"
)

matrix = [r for r in runs if r["matrix"] == "space x Riemann solver" and r.get("variant") is None]
rows, columns = list(dict.fromkeys(r["row"] for r in matrix)), [
    *riemann,
    "Lax-Friedrichs splitting",
]
grid = {(r["row"], r["column"]): r for r in matrix}
space_key = "space x Riemann solver"
fig = plots.figure(1, 1, size=(13, 19))
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
    label="L1 density error at t = 0.2",
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
    notes=[legend_head]
    + [
        f"  {row}: {description}."
        for row, (description, _) in stabilized_space.items()
        if any((space_key, row, c) in hatch for c in columns)
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

timing = [r for r in runs if r["matrix"] == "time integrator x space" and r.get("variant") is None]
time_key = "time integrator x space"
time_rows, time_columns = list(dict.fromkeys(r["row"] for r in timing)), list(time_spatial)
time_grid = {(r["row"], r["column"]): r for r in timing}
fig = plots.figure(1, 2, size=(17, 11))
time_hatched = [[(time_key, row, c) in hatch for c in time_columns] for row in time_rows]
time_notes = (
    [
        "Hatched: the declared integrator is unstable at this step (not A-stable, or undamped on "
        "a dissipative operator); colour and value show its stabilized counterpart:"
    ]
    + [
        f"  {row}: same method at Δt = {stable_step[row]:.1e} instead of 2·10⁻³."
        for row in stable_step
        if any((time_key, row, c) in hatch for c in time_columns)
    ]
    + [
        f"  {row}: {description}."
        for row, (description, _, _) in stabilized_time.items()
        if any((time_key, row, c) in hatch for c in time_columns)
    ]
)
plots.matrix(
    [
        [
            cell[(time_key, row, c)]["errors"][0] if (time_key, row, c) in cell else np.nan
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
                f"{cell[(time_key, row, c)]['errors'][0]:.2e}"
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
    notes=time_notes,
    title="L1 density error",
    ax=fig[0],
)
plots.matrix(
    [
        [
            (
                cell[(time_key, row, c)]["seconds"]
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
    label="wall time [s]",
    text=[
        [
            (
                f"{cell[(time_key, row, c)]['seconds']:.0f} s, {cell[(time_key, row, c)]['steps']} steps"
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
    title="Wall time and accepted steps",
    ax=fig[1],
)
plots.show()

# ## Nonlinear × linear solver
#
# Backward Euler on MUSCL-minmod-HLLC. The discrete solution should not depend on the solver as long
# as each one converges; the wall time does.

solving = [
    r for r in runs if r["matrix"] == "nonlinear x linear solver" and r.get("variant") is None
]
solver_key = "nonlinear x linear solver"
solver_rows = list(dict.fromkeys(r["row"] for r in solving))
solver_columns = [*linear_solvers, "none"]
solver_grid = {(r["row"], r["column"]): r for r in solving}
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
        "Hatched: the fixed-point (Picard) iteration cannot contract at Δt = 2·10⁻³; colour and "
        "value show the same iteration at the step it converges for:"
    ]
    + [
        f"  {row}: Δt = {stable_step[row]:.1e}."
        for row in stable_step
        if any((solver_key, row, c) in hatch for c in solver_columns)
    ],
    title="Backward Euler: wall time and L1 density error per solver pair",
    ax=fig[0],
)
plots.show()

# ## Particles: smoothed-particle hydrodynamics × time integrator × artificial viscosity
#
# The same gas as moving particles instead of cells: 800 equally spaced particles in the periodic
# box $[-0.5, 1.5)$, i.e. 400 particles in the graded window $[0, 1]$, the same resolution as the
# 400 cells. Each particle carries the mass of its cell; the smoothing length is 2.5 spacings and
# the shock is captured by Monaghan's artificial viscosity with the coefficients $\alpha, \beta$.
# Every time integrator of the grid matrix is passed to the particle solver unchanged. The error
# is the mean absolute density deviation from the exact solution at the particles in $[0, 1]$.
#
# Leapfrog has no damping; with any artificial viscosity it is unstable at every step (the
# library refuses it). Its stabilized counterpart is again the Robert-Asselin-Williams filtered
# leapfrog.

particles, box = 800, (-0.5, 1.5)
spacing = (box[1] - box[0]) / particles
start = box[0] + (np.arange(particles) + 0.5) * spacing
start_density = np.where(start < 0.5, 1.0, 0.125)
start_pressure = np.where(start < 0.5, 1.0, 0.1)
viscosities = {
    "Monaghan α = 1, β = 2": (1.0, 2.0),
    "Monaghan α = 0.5, β = 1": (0.5, 1.0),
    "no artificial viscosity": (0.0, 0.0),
}
sph_runs = [
    dict(matrix="SPH", row=name, column=column, time=integrator, viscosity=coefficients)
    for name, integrator in integrators.items()
    for column, coefficients in viscosities.items()
]
sph_runs += [
    {
        **run,
        "variant": "stabilized",
        "time": TimeMethod("leapfrog_raw"),
        "note_stabilized": "leapfrog + Robert-Asselin-Williams filter (ν = 0.2, α = 0.53)",
    }
    for run in list(sph_runs)
    if run["row"] == "leapfrog" and run["viscosity"] != (0.0, 0.0)
]
for run in sph_runs:
    started = time.perf_counter()
    run.update(status="rejected", note="", errors=(np.nan,) * 3, steps=0)
    try:
        result = HydrodynamicPublicApi.create_provider(
            "sph_hydrodynamics",
            positions=start,
            masses=start_density * spacing,
            velocities=np.zeros(particles),
            specific_internal_energy=start_pressure / ((gamma - 1) * start_density),
            smoothing_length=2.5 * spacing,
            gamma=gamma,
            viscosity_alpha=run["viscosity"][0],
            viscosity_beta=run["viscosity"][1],
            periodic_bounds=(box,),
            spatial=SPHDiscretization(SPHParticleCloud),
            final_time=end_time,
            time_step=1.0e-3,
            cfl=0.25,
            time_integrator=run["time"],
        ).run()
        window = (result.positions[:, 0] >= 0.0) & (result.positions[:, 0] <= 1.0)
        reference = np.array(
            [
                [
                    getattr(
                        exact_euler_pressure_trace(
                            *waves, similarity=(x - 0.5) / end_time
                        ).state,
                        name,
                    )
                    for x in result.positions[window, 0]
                ]
                for name in ("density", "velocity", "pressure")
            ]
        )
        values = np.array(
            (
                result.density[window],
                result.velocities[window, 0],
                result.pressure[window],
            )
        )
        run["errors"] = tuple(np.abs(values - reference).mean(axis=1))
        run["status"] = "ran" if np.isfinite(values).all() else "failed"
    except Exception as error:
        message = str(error).splitlines()[0]
        run["status"] = "failed" if "no stable region" in message else "rejected"
        run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    print(
        f"SPH {run['row'][:28]:28s} {run['column'][:26]:26s} {run.get('variant') or '':10s} "
        f"{run['status']:8s} {run['seconds']:6.1f} s  "
        + (f"L1 rho {run['errors'][0]:.2e}" if run["status"] == "ran" else run["note"][:90])
    )

sph_main = {(r["row"], r["column"]): r for r in sph_runs if r.get("variant") is None}
sph_substitute = {
    (r["row"], r["column"]): r for r in sph_runs if r.get("variant") == "stabilized"
}
sph_hatch = {
    key
    for key, r in sph_main.items()
    if r["status"] != "ran" and sph_substitute.get(key, {}).get("status") == "ran"
}
sph_cell = {key: sph_substitute[key] if key in sph_hatch else r for key, r in sph_main.items()}
sph_rows, sph_columns = list(integrators), list(viscosities)
fig = plots.figure(1, 2, size=(17, 12))
for panel, measure, label, title in (
    (fig[0], "errors", "L1 density error", "SPH: L1 density error, 400 particles in [0, 1]"),
    (fig[1], "seconds", "wall time [s]", "SPH: wall time"),
):
    plots.matrix(
        [
            [
                (
                    (sph_cell[(row, c)]["errors"][0] if measure == "errors" else sph_cell[(row, c)]["seconds"])
                    if sph_cell[(row, c)]["status"] == "ran"
                    else np.nan
                )
                for c in sph_columns
            ]
            for row in sph_rows
        ],
        sph_rows,
        sph_columns,
        scale="log",
        label=label,
        text=[
            [
                (
                    (
                        f"{sph_cell[(row, c)]['errors'][0]:.2e}"
                        if measure == "errors"
                        else f"{sph_cell[(row, c)]['seconds']:.1f} s"
                    )
                    if sph_cell[(row, c)]["status"] == "ran"
                    else sph_cell[(row, c)]["status"]
                )
                for c in sph_columns
            ]
            for row in sph_rows
        ],
        hatched=[[(row, c) in sph_hatch for c in sph_columns] for row in sph_rows],
        notes=(
            [
                "Hatched: leapfrog is unstable under any artificial viscosity; colour and value show "
                "leapfrog + Robert-Asselin-Williams filter (ν = 0.2, α = 0.53)."
            ]
            if sph_hatch and measure == "errors"
            else None
        ),
        title=title,
        ax=panel,
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
    selected = [r for r in runs + sph_runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(f"  {r['matrix'][:12]:12s} {r['row'][:34]:34s} {r['column'][:24]:24s} {r['note'][:100]}")
print(
    "\nsummary: "
    + ", ".join(
        f"{sum(r['status'] == s for r in runs + sph_runs)} {s}"
        for s in ("ran", "rejected", "failed")
    )
    + f" of {len(runs) + len(sph_runs)} runs, "
    + f"{sum(r['seconds'] for r in runs + sph_runs) / 60:.0f} min in total"
)

# ## Methods that do not apply to Sod's problem
#
# A few methods in the library are not in the matrix because the Euler system lacks what they are
# built for: conjugate gradients and algebraic multigrid need a symmetric positive definite operator;
# Runge-Kutta-Chebyshev and -Legendre super-steps are for parabolic operators; staggered Verlet needs
# a second-order system; IMEX splits need a stiff and a non-stiff part; local time stepping is for
# steady states; hydrostatic reconstruction is for shallow water and Scharfetter-Gummel fluxes are
# for drift-diffusion. Scalar convection fluxes (linear or monotone upwind, central) are refused for
# the Euler system with a message that names a Riemann solver instead.
#
# ## Going further
#
# Swap the Riemann problem for Lax's or the 123 problem by changing the two initial states; the
# matrix code stays as it is. Add a column by adding one object to a dictionary: a new limiter, a
# different flux, another integrator.
