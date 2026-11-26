"""OneSimulation example: Buckley-Leverett waterflooding through every method combination.
Requires the installed, activated OneSimulation runtime.

One physics declaration, water displacing oil in a porous core, is solved on 400 cells with every
applicable combination of discretization family, reconstruction, limiter, numerical flux, time
integrator, nonlinear solver and linear solver. Each run is graded against the exact self-similar
solution with Welge's shock construction.
"""

# # Buckley-Leverett waterflooding: one model, every method
#
# Water is injected at $x=0$ into a 1 m core of porosity $\phi=0.2$ that initially holds only oil.
# With a constant total Darcy velocity $v$ the water saturation $S$ obeys the scalar conservation
# law
#
# $$\phi\,\partial_t S+\partial_x\big(v\,f(S)\big)=0,\qquad
# f(S)=\frac{S^2}{S^2+m(1-S)^2},$$
#
# with Corey's quadratic relative permeabilities and the viscosity ratio $m=\mu_w/\mu_o=0.5$. The
# flux is non-convex: its S-shape makes the solution a rarefaction fan attached to a shock. The law
# is declared once; everything numerical is an argument of `Numerics`: the discretization family,
# the reconstruction and its limiter, the numerical flux, the time integrator and, for implicit
# steps, the nonlinear and the linear solver.
#
# **Reference.** The exact solution (Buckley and Leverett, *Trans. AIME* 146, 107, 1942) is
# self-similar in $\xi=x/t$. Welge's tangent construction gives the shock saturation $S^*$ from
# $f(S^*)=S^*f'(S^*)$; behind the shock each saturation $S\in[S^*,1]$ travels with
# $\xi=v f'(S)/\phi$, ahead of it $S=0$:
#
# $$S(x,t)=\begin{cases}(f')^{-1}\big(\phi x/(v t)\big), & x/t\le v f'(S^*)/\phi,\\
# 0, & x/t>v f'(S^*)/\phi.\end{cases}$$
#
# The core is followed until the shock has reached $x=0.6$ m.
#
# **Grading.** At the final time every run is evaluated at the 400 cell centres $x_i$:
# $L_1(S)=\sum_i|S_h(x_i)-S(x_i)|\,\Delta x$; the shock position is where the saturation crosses
# $S^*/2$; the wall time covers validation, binding and time stepping.

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
    GlobalLaxFriedrichsSplit,
    JSTDissipation,
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
    ScalarInterval,
    VolumeCIP,
)
from onesim.numerics.fluxes import HLL, HLLE, MonotoneUpwind, Rusanov
from onesim.numerics.fvm import (
    MUSCL,
    QUICK,
    BarthJespersenLimiter,
    CentralWeightedENO,
    ConvexInvariantDomain,
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
    TimeStepLimits,
)
from onesim.physics.geotechnics import (
    BuckleyLeverettFlow,
    QuadraticCoreyFractionalFlow,
    SaturationInlet,
    SaturationOutflow,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()

# ## The physics, declared once
#
# Nothing in this cell names a numerical method.

porosity, darcy_velocity, viscosity_ratio, shock_target = 0.2, 1.0e-5, 0.5, 0.6
cells, coarse_cells = 400, 200
core = Line(start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0))
corey = QuadraticCoreyFractionalFlow(viscosity_ratio=viscosity_ratio)
flood = BuckleyLeverettFlow(
    domain=core.line, porosity=porosity, darcy_velocity=darcy_velocity, fractional_flow=corey
)
waterflood = Problem(
    name="Buckley-Leverett waterflood",
    geometry=core,
    physics=(flood,),
    boundary_conditions=(
        SaturationInlet(physics=flood, boundary=core.boundaries.start, value=1.0),
        SaturationOutflow(physics=flood, boundary=core.boundaries.end),
    ),
    initial_conditions=(InitialValue(flood.water_saturation, 0.0),),
)
meshes = {n: StructuredLineMesher(cells=n).generate(core) for n in (cells, coarse_cells)}
centres = {n: mesh.cell_centers[:, 0] for n, mesh in meshes.items()}

# ## The exact solution
#
# Welge's tangent from the origin to the fractional-flow curve gives the shock saturation; the fan
# behind it is the inverse of $f'$ on $[S^*,1]$, where $f'$ decreases monotonically.

front = brentq(lambda s: corey.fraction(s) - s * corey.fraction_derivative(s), 1e-3, 1 - 1e-9)
shock_speed = darcy_velocity * corey.fraction_derivative(front) / porosity
end_time = shock_target / shock_speed
fan = np.linspace(front, 1.0, 20001)
fan_speed = darcy_velocity * corey.fraction_derivative(fan) / porosity
fine = np.linspace(0.0, 1.0, 4001)
exact = {
    key: np.where(
        points / end_time > shock_speed,
        0.0,
        np.interp(points / end_time, fan_speed[::-1], fan[::-1]),
    )
    for key, points in {**centres, "fine": fine}.items()
}
fastest = darcy_velocity * np.max(corey.fraction_derivative(np.linspace(0, 1, 100001))) / porosity
print(
    f"{meshes[cells].cell_count} cells of {1e3 / cells:.1f} mm (coarse check: "
    f"{meshes[coarse_cells].cell_count} cells); Welge shock saturation S* = {front:.6f}, shock "
    f"speed {shock_speed:.4e} m/s, shock at x = {shock_target} m for t = {end_time:.0f} s "
    f"({end_time / 3600:.2f} h); fastest characteristic {fastest:.4e} m/s"
)

# ## The method catalogue
#
# Four numerical fluxes: Rusanov, HLL, HLLE and the monotone upwind (Godunov-type) flux of the scalar
# law (HLL and HLLE take their signal speeds from the extrema of $v f'(S)$ between the two traces,
# which the waterflood law supplies because the S-shaped flux makes the trace speeds alone too slow); seven slope limiters plus the unlimited slope; higher-order reconstructions
# including ENO, WENO3, WENO5, their WENO-Z variants and central WENO; finite elements with
# discontinuous elements of degree 0 to 4, with and without limiting or a jump penalty, and
# continuous elements of degree 1 to 3 with graph viscosity, continuous interior penalty, volume
# interior penalty, SUPG, GLS or no stabilization; and finite-difference stencils including
# summation-by-parts operators of order 2 and 4.
#
# The invariant domain of the saturation is the interval $[0,1]$; it is one object,
# `ScalarInterval`, given to the discontinuous-Galerkin limiter and to the graph viscosity.

fluxes = {
    "Rusanov": Rusanov(),
    "HLL": HLL(),
    "HLLE": HLLE(),
    "monotone upwind": MonotoneUpwind(),
}
saturation_range = ScalarInterval(field="water_saturation", lower=0.0, upper=1.0)
limiters = {
    "minmod": MinmodLimiter(),
    "MC": MonotonizedCentralLimiter(),
    "van Leer": VanLeerLimiter(),
    "superbee": SuperbeeLimiter(),
    "Barth-Jespersen": BarthJespersenLimiter(),
    "Venkatakrishnan": VenkatakrishnanLimiter(),
    "TVB (M = 50)": TvbLimiter(base_limiter=MinmodLimiter(), tvb_constant=50.0, mesh_width=1 / cells),
}
reconstructions = {
    "ENO2": Eno2Reconstructor(),
    "ENO3": Eno3Reconstructor(),
    "WENO3": Weno3Reconstructor(),
    "WENO5": Weno5Reconstructor(),
    "WENO3-Z": Weno3Reconstructor(weights=WenoZWeights(global_indicator=SmoothnessDifference())),
    "WENO5-Z": Weno5Reconstructor(weights=WenoZWeights(global_indicator=SmoothnessDifference())),
    "CWENO3": CentralWeightedENO(weights=JiangShuWeights()),
    "CWENO3-Z": CentralWeightedENO(
        weights=WenoZWeights(global_indicator=SmoothnessDifference(first=1, last=2))
    ),
    "QUICK": QUICK(),
}
dg_limiter = DGPolynomialLimiter(slope_limiter=MinmodLimiter(), invariant_domain=saturation_range)
bounded_dg = DGPolynomialLimiter(invariant_domain=saturation_range)
graph_viscosity = InvariantDomainGraphViscosity(mass=RowSumLumping(), invariant_domain=saturation_range)
spatial = {}
for flux_name, flux in fluxes.items():
    spatial[("FVM first order", flux_name)] = FiniteVolume(convection=flux)
    spatial[("FVM MUSCL, unlimited", flux_name)] = FiniteVolume(convection=flux, reconstruction=MUSCL())
    for limiter_name, limiter in limiters.items():
        spatial[(f"FVM MUSCL + {limiter_name}", flux_name)] = FiniteVolume(
            convection=flux, reconstruction=MUSCL(), limiter=limiter
        )
    for reconstruction_name, reconstruction in reconstructions.items():
        spatial[(f"FVM {reconstruction_name}", flux_name)] = FiniteVolume(
            convection=flux, reconstruction=reconstruction
        )
    spatial[("FEM DG P0", flux_name)] = FiniteElement(element=DiscontinuousLagrange(0), convection=flux)
    for degree in (1, 2, 3, 4):
        spatial[(f"FEM DG P{degree}, unlimited", flux_name)] = FiniteElement(
            element=DiscontinuousLagrange(degree), convection=flux
        )
        spatial[(f"FEM DG P{degree} + minmod + [0,1] bound", flux_name)] = FiniteElement(
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
        spatial[(f"FEM CG P{degree} + continuous interior penalty", flux_name)] = FiniteElement(
            element=Lagrange(degree),
            convection=flux,
            stabilization=CIP(penalty=penalty),
            initialization=NodalInterpolation(),
        )
    spatial[("FEM CG P1 + volume interior penalty", flux_name)] = FiniteElement(
        element=Lagrange(1),
        convection=flux,
        stabilization=VolumeCIP(diffusivity=2e-3 * fastest),
        initialization=NodalInterpolation(),
    )
    for name, kind in (("SUPG", SUPG), ("GLS", GLS)):
        spatial[(f"FEM CG P1 + {name}", flux_name)] = FiniteElement(
            element=Lagrange(1),
            convection=flux,
            stabilization=kind(
                tau=0.5 / (cells * fastest), strong_residual=InterpolatedFluxDivergence()
            ),
            initialization=NodalInterpolation(),
        )
    spatial[("FEM DG P1 + jump penalty", flux_name)] = FiniteElement(
        element=DiscontinuousLagrange(1), convection=flux, stabilization=DGJumpPenalty(coefficient=1.0)
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
spatial[("FDM WENO5 flux splitting", "Lax-Friedrichs splitting")] = FiniteDifference(
    gradient=WENOFluxDifference(flux_splitting=GlobalLaxFriedrichsSplit(), reconstruction=WENO5Flux())
)

# ## Schemes that must fail at a shock, and their stabilized counterparts
#
# Linear schemes of higher than first order oscillate at a discontinuity (Godunov's theorem): MUSCL
# and QUICK without a limiter, discontinuous Galerkin without limiting or with only a jump penalty,
# continuous Galerkin without stabilization or with a linear SUPG/GLS term, central and
# summation-by-parts differences. Essentially non-oscillatory reconstructions (ENO, WENO) and the
# smooth limiters (Venkatakrishnan, TVB) are not strictly bounded either: their small overshoots at
# the inlet jump already leave $[0,1]$. Outside $[0,1]$ the fractional-flow law has no meaning; the
# library stops the run instead of clipping it. For each such scheme the matrix also runs the
# counterpart that adds exactly one ingredient — for finite volumes the invariant-domain limiter
# that keeps every face state in $[0,1]$ — shown behind a grey hatching.

stabilized_space = {
    "FVM MUSCL, unlimited": (
        "MUSCL + invariant-domain face limiter ([0,1] at every face)",
        lambda flux: FiniteVolume(
            convection=flux, reconstruction=MUSCL(), positivity=ConvexInvariantDomain()
        ),
    ),
    **{
        f"FVM MUSCL + {limiter_name}": (
            f"MUSCL + {limiter_name} + invariant-domain face limiter",
            lambda flux, limiter=limiter: FiniteVolume(
                convection=flux,
                reconstruction=MUSCL(),
                limiter=limiter,
                positivity=ConvexInvariantDomain(),
            ),
        )
        for limiter_name, limiter in limiters.items()
    },
    **{
        f"FVM {reconstruction_name}": (
            f"{reconstruction_name} + invariant-domain face limiter",
            lambda flux, reconstruction=reconstruction: FiniteVolume(
                convection=flux, reconstruction=reconstruction, positivity=ConvexInvariantDomain()
            ),
        )
        for reconstruction_name, reconstruction in reconstructions.items()
    },
    **{
        f"FEM DG P{degree}, unlimited": (
            f"FEM DG P{degree} + [0,1] invariant-domain scaling (no slope limiter)",
            lambda flux, degree=degree: FiniteElement(
                element=DiscontinuousLagrange(degree), convection=flux, limiter=bounded_dg
            ),
        )
        for degree in (1, 2, 3, 4)
    },
    "FEM DG P1 + jump penalty": (
        "DG P1 + [0,1] invariant-domain scaling instead of the jump penalty",
        lambda flux: FiniteElement(
            element=DiscontinuousLagrange(1), convection=flux, limiter=bounded_dg
        ),
    ),
    **{
        f"FEM CG P{degree} Galerkin": (
            f"FEM CG P{degree} + invariant-domain graph viscosity",
            lambda flux, degree=degree: FiniteElement(
                element=Lagrange(degree), convection=flux, stabilization=graph_viscosity
            ),
        )
        for degree in (1, 2, 3)
    },
    **{
        f"FEM CG P1 + {name}": (
            f"CG P1 + invariant-domain graph viscosity instead of {name}",
            lambda flux: FiniteElement(
                element=Lagrange(1), convection=flux, stabilization=graph_viscosity
            ),
        )
        for name in ("SUPG", "GLS")
    },
    "FEM CG P1 + volume interior penalty": (
        "CG P1 + invariant-domain graph viscosity instead of the volume interior penalty",
        lambda flux: FiniteElement(
            element=Lagrange(1), convection=flux, stabilization=graph_viscosity
        ),
    ),
    **{
        f"FEM CG P{degree} + continuous interior penalty": (
            f"CG P{degree} + invariant-domain graph viscosity instead of the interior penalty",
            lambda flux, degree=degree: FiniteElement(
                element=Lagrange(degree), convection=flux, stabilization=graph_viscosity
            ),
        )
        for degree in (1, 2)
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
}

# Time integrators: the same catalogue as for every other conservation law. Implicit stages take a
# Newton solver with a coloured finite-difference Jacobian and a sparse direct solve; the linearly
# implicit Rosenbrock stages take the sparse direct solve alone. The implicit step is 1.4 times the
# explicit CFL limit of the fastest characteristic. Implicit stages use damped Newton: the S-shaped
# flux makes undamped Newton iterates overshoot $[0,1]$ at that step although the converged stage
# solution stays bounded (see the solver matrix).

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
implicit_step = 1.4 / (cells * fastest)
newton = DampedNewton(linear_solver=SparseDirect(), jacobian="colored_finite_difference")
# Only backward Euler keeps the saturation in $[0,1]$ at every step size (it is unconditionally
# strong-stability preserving). Crank-Nicolson, the Adams-Moulton methods and BDF3 to BDF6 preserve
# the bounds of the limited MUSCL scheme only below a CFL number of about one; at the implicit step
# they overshoot at the shock and must fail. Their counterpart is the same method at a quarter of the
# step. A fixed-point (Picard) iteration contracts only below the inverse spectral radius of the
# stage operator (counterpart: an eighth of the step); undamped Newton, quasi-Newton and JFNK
# iterations can overshoot $[0,1]$ before they converge (counterpart: half the step). Leapfrog is
# undamped and unstable on the dissipative upwind operator; its counterpart adds the
# Robert-Asselin-Williams filter at a reduced safety factor. The adaptive Dormand-Prince pair is not
# strong-stability preserving either (its stages combine with negative weights) and overshoots the
# inlet jump; no single added ingredient makes it bounded, so it is reported as failed where it
# does.
stabilized_time = {
    "leapfrog": (
        "leapfrog + Robert-Asselin-Williams filter (ν = 0.2, α = 0.53; first order), safety factor 0.15",
        TimeMethod("leapfrog_raw"),
        0.15,
    ),
}
stable_step = {
    **{
        name: implicit_step / 4
        for name in (
            "Crank-Nicolson",
            "Adams-Moulton 2",
            "Adams-Moulton 3",
            "Adams-Moulton 4",
            "BDF3 (variable step)",
            "BDF3 (Gear)",
            "BDF4 (Gear)",
            "BDF5 (Gear)",
            "BDF6 (Gear)",
        )
    },
    "fixed point": implicit_step / 8,
    "fixed point + Aitken": implicit_step / 8,
    **{name: implicit_step / 2 for name in ("Newton", "quasi-Newton", "JFNK")},
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
        (name, linear_name): kind(linear_solver=linear, jacobian="colored_finite_difference")
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
# 1. **Space × numerical flux** — every spatial scheme with every flux, advanced with SSP-RK2 at
#    the stable step (safety factor 0.5).
# 2. **Time integrator × space** — every integrator with the monotone upwind flux on first-order,
#    MUSCL-minmod and WENO5 volumes; implicit integrators on the first two at 1.4 times the CFL limit.
# 3. **Nonlinear × linear solver** — backward Euler on MUSCL-minmod with the monotone upwind flux.
#
# A few representative combinations are repeated on 200 cells for the mesh check.

time_spatial = {
    name: spatial[(name, "monotone upwind")]
    for name in ("FVM first order", "FVM MUSCL + minmod", "FVM WENO5")
}
runs = [
    dict(
        matrix="space x flux",
        row=row,
        column=column,
        cells=cells,
        spatial=scheme,
        time=TimeMethod("ssprk2"),
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
        step=implicit_step if getattr(integrator, "implicit", False) else end_time,
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
        step=implicit_step,
    )
    for (row, column), solver in nonlinear_solvers.items()
]
runs += [
    dict(
        matrix="space x flux",
        variant="stabilized",
        row=row,
        column=flux_name,
        cells=cells,
        spatial=make(flux),
        time=TimeMethod("ssprk2"),
        step=end_time,
        note_stabilized=description,
    )
    for row, (description, make) in stabilized_space.items()
    for flux_name, flux in fluxes.items()
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
        "note_stabilized": f"{run['row']} at Δt = {stable_step[run['row']]:.3g} s",
    }
    for run in list(runs)
    if run["matrix"] in ("time integrator x space", "nonlinear x linear solver")
    and run["row"] in stable_step
    and run.get("variant") is None
]
representative = [
    ("FVM first order", "Rusanov"),
    ("FVM first order", "monotone upwind"),
    ("FVM MUSCL + minmod", "monotone upwind"),
    ("FVM MUSCL + superbee", "monotone upwind"),
    ("FVM WENO5", "monotone upwind"),
    ("FVM MUSCL + van Leer", "HLL"),
    ("FEM DG P2 + minmod + [0,1] bound", "monotone upwind"),
    ("FEM CG P1 + graph viscosity", "monotone upwind"),
]
runs += [
    {**run, "matrix": "mesh check", "cells": coarse_cells}
    for run in runs
    if run["matrix"] == "space x flux"
    and run.get("variant") is None
    and (run["row"], run["column"]) in representative
]
print(
    f"{len(runs)} runs: "
    + ", ".join(
        f"{sum(r['matrix'] == m for r in runs)} {m}" for m in dict.fromkeys(r["matrix"] for r in runs)
    )
)

# ## Run everything
#
# A stabilized counterpart is run only where the declared combination did not run.
# One loop, the same problem object every time. A combination the library refuses is recorded with
# its validation message; a run that stops because the saturation left $[0,1]$ or a solver did not
# converge is recorded as failed.

for run in runs:
    if run.get("variant") == "stabilized" and any(
        other.get("variant") is None
        and other.get("status") == "ran"
        and (other["matrix"], other["row"], other["column"])
        == (run["matrix"], run["row"], run["column"])
        for other in runs
    ):
        run.update(status="not needed", note="", error=np.nan, shock=np.nan, steps=0)
        run["seconds"] = 0.0
        continue
    implicit = getattr(run["time"], "implicit", False)
    rosenbrock = "rosenbrock" in getattr(run["time"], "method_id", "")
    simulation = Simulation(
        problem=waterflood,
        mesh=meshes[run["cells"]],
        numerics=Numerics(
            spatial=run["spatial"],
            linear_solver=(
                SparseDirect() if isinstance(run["spatial"], FiniteElement) or rosenbrock else None
            ),
            time_integrator=run["time"],
            nonlinear_solver=run.get("nonlinear", newton) if implicit and not rosenbrock else None,
            step_limits=TimeStepLimits(stability_safety_factor=run.get("safety", 0.5)),
        ),
    )
    started = time.perf_counter()
    check = simulation.validate()
    run.update(status="rejected", note="", error=np.nan, shock=np.nan, steps=0)
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = simulation.run(start_time=0.0, end_time=end_time, time_step=run["step"])
            points = np.column_stack((centres[run["cells"]], np.zeros((run["cells"], 2))))
            run["profile"] = result.field(flood.water_saturation).evaluate(points).reshape(-1)
            run["error"] = np.abs(run["profile"] - exact[run["cells"]]).sum() / run["cells"]
            crossing = np.flatnonzero(run["profile"] > front / 2)[-1]
            run["shock"] = np.interp(
                front / 2,
                run["profile"][[crossing + 1, crossing]],
                centres[run["cells"]][[crossing + 1, crossing]],
            )
            run["steps"] = len(result.times) - 1
            run["status"] = "ran" if np.isfinite(run["profile"]).all() else "failed"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "failed"
                if "[0,1]" in message
                or "finite" in message
                or "did not converge" in message
                or "no stable region" in message
                or "maximum iteration" in message
                else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    print(
        f"{run['matrix'][:12]:12s} {run['row'][:36]:36s} {run['column'][:24]:24s} "
        f"{run.get('variant') or '':10s} {run['status']:8s} {run['seconds']:7.1f} s  "
        + (
            f"L1 {run['error']:.2e}  shock {run['shock'] - shock_target:+.1e}"
            if run["status"] == "ran"
            else run["note"][:90]
        )
    )

# ## Space × numerical flux
#
# The saturation error of every spatial scheme with every numerical flux. Where the declared scheme
# failed at the shock and a stabilized counterpart exists, the cell is hatched grey and shows the
# counterpart; the legend says what was added.

main = {
    (r["matrix"], r["row"], r["column"]): r
    for r in runs
    if r.get("variant") is None and r["matrix"] != "mesh check"
}
substitute = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") == "stabilized"}
hatch = {
    key
    for key, r in main.items()
    if r["status"] != "ran" and substitute.get(key, {}).get("status") == "ran"
}
cell = {key: substitute[key] if key in hatch else r for key, r in main.items()}

space_key = "space x flux"
matrix = [r for r in runs if r["matrix"] == space_key and r.get("variant") is None]
rows, columns = list(dict.fromkeys(r["row"] for r in matrix)), [*fluxes, "Lax-Friedrichs splitting"]
fig = plots.figure(1, 1, size=(14, 19))
plots.matrix(
    [
        [cell[(space_key, row, c)]["error"] if (space_key, row, c) in cell else np.nan for c in columns]
        for row in rows
    ],
    rows,
    columns,
    scale="log",
    label="L1 saturation error",
    text=[
        [
            (
                f"{cell[(space_key, row, c)]['error']:.2e}"
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
    notes=[
        "Hatched: the declared scheme oscillates at the shock and leaves [0,1] (Godunov's theorem); "
        "colour and value show its stabilized counterpart:"
    ]
    + [
        f"  {row}: {description}."
        for row, (description, _) in stabilized_space.items()
        if any((space_key, row, c) in hatch for c in columns)
    ],
    title=f"Space x numerical flux, {cells} cells, SSP-RK2",
    ax=fig[0],
)
plots.show()

# All measures for the combinations that ran, sorted by the saturation error.

print(f"{'scheme':38s} {'flux':24s} {'L1 S':>9s} {'shock err':>10s} {'steps':>6s} {'time':>8s}")
for r in sorted((r for r in matrix if r["status"] == "ran"), key=lambda r: r["error"]):
    print(
        f"{r['row']:38s} {r['column']:24s} {r['error']:9.2e} {r['shock'] - shock_target:+10.1e} "
        f"{r['steps']:6d} {r['seconds']:7.1f}s"
    )

# ## Profiles of representative combinations
#
# The saturation against the exact solution, and a close-up of the shock.

shown = [r for r in matrix if (r["row"], r["column"]) in representative and r["status"] == "ran"]
fig = plots.figure(1, 2, size=(15, 5))
for panel, xwindow, ywindow, title in (
    (fig[0], None, None, "Water saturation"),
    (fig[1], (0.5, 0.7), (0.0, 0.75), "Water saturation at the shock"),
):
    plots.line(
        y={f"{r['row']}, {r['column']}": (centres[cells], r["profile"]) for r in shown},
        reference={"exact (Welge)": (fine, exact["fine"])},
        xlabel="position x [m]",
        ylabel="water saturation S",
        title=title,
        xlimits=xwindow,
        ylimits=ywindow,
        legend="upper right" if xwindow is None else False,
        ax=panel,
    )
plots.show()

# ## Time integrator × space
#
# Explicit integrators at their stable step, implicit integrators at 1.4 times the CFL limit. A dash
# marks a combination outside this cut (implicit WENO5).

time_key = "time integrator x space"
time_rows, time_columns = list(integrators), list(time_spatial)
fig = plots.figure(1, 2, size=(17, 11))
time_hatched = [[(time_key, row, c) in hatch for c in time_columns] for row in time_rows]
time_notes = (
    [
        "Hatched: the declared integrator is unstable at this step (not A-stable, or undamped on a "
        "dissipative operator); colour and value show its stabilized counterpart:"
    ]
    + [
        f"  {row}: same method at Δt = {stable_step[row]:.3g} s instead of {implicit_step:.3g} s."
        for row in stable_step
        if any((time_key, row, c) in hatch for c in time_columns)
    ]
    + [
        f"  {row}: {description}."
        for row, (description, _, _) in stabilized_time.items()
        if any((time_key, row, c) in hatch for c in time_columns)
    ]
)
for panel, measure, label, title in (
    (fig[0], "error", "L1 saturation error", "L1 saturation error"),
    (fig[1], "seconds", "wall time [s]", "Wall time and accepted steps"),
):
    plots.matrix(
        [
            [
                (
                    cell[(time_key, row, c)][measure]
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
                        f"{cell[(time_key, row, c)]['error']:.2e}"
                        if measure == "error"
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
        notes=time_notes if measure == "error" and len(time_notes) > 1 else None,
        title=title,
        ax=panel,
    )
plots.show()

# ## Nonlinear × linear solver
#
# Backward Euler on MUSCL-minmod with the monotone upwind flux. The discrete solution should not
# depend on the solver as long as each one converges; the wall time does.

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
                    f"{cell[(solver_key, row, c)]['seconds']:.0f} s\n"
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
        "Hatched: the declared solver fails at the implicit step; colour and value show the same "
        "solver at a smaller step:"
    ]
    + [
        f"  {row}: Δt = {stable_step[row]:.3g} s instead of {implicit_step:.3g} s."
        for row in solver_rows
        if row in stable_step and any((solver_key, row, c) in hatch for c in solver_columns)
    ],
    title="Backward Euler: wall time and L1 saturation error per solver pair",
    ax=fig[0],
)
plots.show()

# ## Mesh check: 400 against 200 cells
#
# The representative combinations on half as many cells. For a solution with a shock the $L_1$
# error falls roughly like $\Delta x^{1/2}$ to $\Delta x$; the shock position converges.

coarse = {(r["row"], r["column"]): r for r in runs if r["matrix"] == "mesh check"}
grid = {(r["row"], r["column"]): r for r in matrix}
print(
    f"{'scheme':38s} {'flux':24s} {'L1 200':>9s} {'L1 400':>9s} {'ratio':>6s} "
    f"{'shock 200':>10s} {'shock 400':>10s}"
)
for key, r in coarse.items():
    fine_run = grid[key]
    if r["status"] == "ran" and fine_run["status"] == "ran":
        print(
            f"{key[0]:38s} {key[1]:24s} {r['error']:9.2e} {fine_run['error']:9.2e} "
            f"{r['error'] / fine_run['error']:6.2f} {r['shock'] - shock_target:+10.1e} "
            f"{fine_run['shock'] - shock_target:+10.1e}"
        )

# ## What the library refused, and what failed
#
# A refused combination is caught by validation or binding before any time step; a failed one
# started and left the admissible saturation range.

for status in ("rejected", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(
            f"  {r['matrix'][:12]:12s} {r['row'][:36]:36s} {r['column'][:24]:24s} "
            f"{r.get('variant') or '':10s} {r['note'][:90]}"
        )
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.0f} min in total"
)

# ## Methods that do not apply to the waterflood
#
# HLLC restores a contact wave between its two outer signals; a scalar conservation law has no
# contact wave, and the library refuses HLLC with a message that names HLL, HLLE and Rusanov. The
# linear upwind flux and the central flux need a linear (or explicitly linearized) flux and are
# refused for the S-shaped fractional flow with a message that names a monotone flux instead.
# Four groups still fail without a running counterpart and are reported as failed: finite-
# difference WENO5 with global Lax-Friedrichs flux splitting has no bounded (invariant-domain)
# variant yet; the Jameson-Schmidt-Turkel dissipation added to the 4th-order and summation-by-parts
# differences is not strong enough to keep the inlet jump inside $[0,1]$; WENO5 and WENO5-Z with the
# Rusanov flux overshoot even with the face limiter at the default step; and the adaptive
# Dormand-Prince pair is not strong-stability preserving.
# Conjugate gradients and algebraic multigrid need a symmetric positive definite operator; the
# upwinded transport operator is not symmetric. Runge-Kutta-Chebyshev and -Legendre super-steps are
# for parabolic operators; hydrostatic reconstruction is for shallow water and Scharfetter-Gummel
# fluxes are for drift-diffusion.
#
# ## Going further
#
# Change the viscosity ratio: a more viscous oil ($m<1$ smaller) lowers the shock saturation and
# speeds up the front. Add capillary diffusion or gravity by replacing the fractional-flow object;
# the matrix code stays as it is.
