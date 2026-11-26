"""OneSimulation example: the shallow-water dam break through every method combination OneSimulation offers.
Requires the installed, activated OneSimulation runtime.

One physics declaration, the Saint-Venant equations with a dam-break initial state, is solved on 400
cells with every applicable combination of discretization family, reconstruction, limiter, Riemann
solver, time integrator, nonlinear solver and linear solver. The wet-bed case is graded against
Stoker's exact solution, the dry-bed case against Ritter's.
"""

# # Shallow-water dam break: one model, every method
#
# A channel of length 10 m holds water 1 m deep behind a gate at $x_0=5$ m. In front of the gate the
# water is 0.1 m deep (wet bed) or absent (dry bed). The one-dimensional shallow-water
# (Saint-Venant) equations for the depth $h$ and the discharge $q=hu$,
#
# $$\partial_t\begin{pmatrix}h\\ q\end{pmatrix}
# +\partial_x\begin{pmatrix}q\\ q^2/h+\tfrac12 g h^2\end{pmatrix}=0,\qquad g=9.81\ \mathrm{m/s^2},$$
#
# are declared once. When the gate is removed, a rarefaction runs back into the reservoir and a bore
# (wet bed) or a thin front (dry bed) runs downstream. The waves do not reach the reflecting walls at
# $x=0$ and $x=10$ m before $t=0.6$ s.
#
# The flat-bed model `ShallowWater` requires a strictly positive depth, so the dry bed is declared with
# `TopographicShallowWater` (flat bed, `bed_elevation=0`), whose hydrostatic providers handle dry
# cells: the hydrostatic reconstruction for finite volumes, a hydrostatic flux difference for finite
# differences and a hydrostatic graph viscosity for continuous finite elements.
#
# **Reference.** J. J. Stoker, *Water Waves*, Interscience 1957, ch. 10 (wet bed): the middle depth
# $h_m$ joins the reservoir rarefaction to the bore,
# $2(c_0-c_m)=(h_m-h_1)\sqrt{g(h_m+h_1)/(2h_mh_1)}$ with $c=\sqrt{gh}$; A. Ritter, *Z. Vereines
# Deutscher Ingenieure* 36, 947 (1892) (dry bed): $h=(2c_0-\xi)^2/(9g)$ for $-c_0<\xi<2c_0$,
# $\xi=(x-x_0)/t$. In the rarefaction $u=\tfrac23(c_0+\xi)$.
#
# **Grading.** At $t=0.6$ s every run is evaluated at the 400 cell centres:
# $L_1(h)=\sum_i|h_h(x_i)-h(x_i)|\,\Delta x$ and the same for the discharge; the wall time covers
# validation, binding and time stepping.

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
    HydrostaticFluxDifference,
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
    DGPolynomialLimiter,
    DiscontinuousLagrange,
    FiniteElement,
    HydrostaticGraphViscosity,
    InterpolatedFluxDivergence,
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
    ConvexInvariantDomain,
    Eno2Reconstructor,
    Eno3Reconstructor,
    FiniteVolume,
    HydrostaticReconstruction,
    JiangShuWeights,
    MinmodLimiter,
    MonotonizedCentralLimiter,
    SmoothnessDifference,
    SuperbeeLimiter,
    TvbLimiter,
    VanLeerLimiter,
    VenkatakrishnanLimiter,
    WenoZWeights,
    Weno3Reconstructor,
    Weno5Reconstructor,
    WetDryActiveSet,
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
from onesim.physics.hydrodynamics import (
    ShallowWater,
    ShallowWaterInvariantDomain,
    TopographicShallowWater,
    WetDryShallowWaterDomain,
)
from onesim.simulation import Problem, ReflectingWall, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()

# ## The physics, declared once
#
# Two problems: the wet bed with the flat-bed model and the dry bed with the topographic model.
# Nothing in this cell names a numerical method.

g, gate, deep, shallow, end_time = 9.81, 5.0, 1.0, 0.1, 0.6
cells, coarse_cells = 400, 200
channel = Line(start=(0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0))
wet_water = ShallowWater(domain=channel.line, gravity=g)
dry_water = TopographicShallowWater(domain=channel.line, gravity=g, bed_elevation=0.0)
models = {"wet": wet_water, "dry": dry_water}
released = {
    "wet": lambda x: np.where(x < gate, deep, shallow),
    "dry": lambda x: np.where(x < gate, deep, 0.0),
}
problems = {
    case: Problem(
        name=f"Dam break, {case} bed",
        geometry=channel,
        physics=(model,),
        boundary_conditions=(
            ReflectingWall(physics=model, boundary=channel.boundaries.all),
        ),
        initial_conditions=(
            InitialValue(model.depth, released[case]),
            InitialValue(model.discharge, 0.0),
        ),
    )
    for case, model in models.items()
}
meshes = {n: StructuredLineMesher(cells=n).generate(channel) for n in (cells, coarse_cells)}
centres = {n: mesh.cell_centers[:, 0] for n, mesh in meshes.items()}
print(
    f"{meshes[cells].cell_count} cells of width {10 / cells:.4f} m "
    f"(coarse check: {meshes[coarse_cells].cell_count} cells)"
)

# ## The exact solutions
#
# Stoker's middle depth is the root of the bore condition; the self-similar depth and discharge
# follow from $\xi=(x-x_0)/t$.

c0 = np.sqrt(g * deep)
middle = brentq(
    lambda h: 2 * (c0 - np.sqrt(g * h))
    - (h - shallow) * np.sqrt(g * (h + shallow) / (2 * h * shallow)),
    shallow,
    deep,
)
middle_velocity = 2 * (c0 - np.sqrt(g * middle))
bore_speed = middle * middle_velocity / (middle - shallow)
fine = np.linspace(0.0, 10.0, 4001)
similarity = {key: (x - gate) / end_time for key, x in {**centres, "fine": fine}.items()}
exact = {
    ("wet", key): np.array(
        [
            np.select(
                [xi < -c0, xi < middle_velocity - np.sqrt(g * middle), xi < bore_speed],
                [deep, (2 * c0 - xi) ** 2 / (9 * g), middle],
                shallow,
            ),
            np.select(
                [xi < -c0, xi < middle_velocity - np.sqrt(g * middle), xi < bore_speed],
                [0.0, (2 * c0 - xi) ** 2 * 2 * (c0 + xi) / (27 * g), middle * middle_velocity],
                0.0,
            ),
        ]
    )
    for key, xi in similarity.items()
}
exact.update(
    {
        ("dry", key): np.array(
            [
                np.select([xi < -c0, xi < 2 * c0], [deep, (2 * c0 - xi) ** 2 / (9 * g)], 0.0),
                np.select(
                    [xi < -c0, xi < 2 * c0],
                    [0.0, (2 * c0 - xi) ** 2 * 2 * (c0 + xi) / (27 * g)],
                    0.0,
                ),
            ]
        )
        for key, xi in similarity.items()
    }
)
print(
    f"Stoker: middle depth {middle:.5f} m, middle velocity {middle_velocity:.5f} m/s, "
    f"bore speed {bore_speed:.5f} m/s; Ritter front speed {2 * c0:.5f} m/s"
)

# ## The method catalogue
#
# Four Riemann solvers: Rusanov, HLL, HLLE and HLLC. HLL takes the shallow-water signal speeds
# $u\mp c$ of both sides (and the dry-front speeds $u\pm2c$ next to a dry cell), HLLE the Einfeldt
# bounds from the Roe average, and HLLC adds Toro's middle wave with the star depths and the contact
# speed. Slope limiters, higher-order reconstructions, discontinuous Galerkin of degree 0 to 3 with
# and without the shallow-water invariant-domain limiter, continuous Galerkin with graph viscosity,
# interior penalty, SUPG, GLS or no stabilization, and finite-difference stencils.

riemann = {"Rusanov": Rusanov(), "HLL": HLL(), "HLLE": HLLE(), "HLLC": HLLC()}
limiters = {
    "minmod": MinmodLimiter(),
    "MC": MonotonizedCentralLimiter(),
    "van Leer": VanLeerLimiter(),
    "superbee": SuperbeeLimiter(),
    "Barth-Jespersen": BarthJespersenLimiter(),
    "Venkatakrishnan": VenkatakrishnanLimiter(),
    "TVB (M = 50)": TvbLimiter(
        base_limiter=MinmodLimiter(), tvb_constant=50.0, mesh_width=10 / cells
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
    "QUICK": QUICK(),
}
water_domain = ShallowWaterInvariantDomain()
dg_limiter = DGPolynomialLimiter(slope_limiter=MinmodLimiter(), invariant_domain=water_domain)
graph_viscosity = InvariantDomainGraphViscosity(mass=RowSumLumping(), invariant_domain=water_domain)
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
    for degree in (1, 2, 3):
        spatial[(f"FEM DG P{degree}, unlimited", flux_name)] = FiniteElement(
            element=DiscontinuousLagrange(degree), convection=flux
        )
        spatial[(f"FEM DG P{degree} + minmod + invariant domain", flux_name)] = FiniteElement(
            element=DiscontinuousLagrange(degree), convection=flux, limiter=dg_limiter
        )
    for degree in (1, 2):
        spatial[(f"FEM CG P{degree} + graph viscosity", flux_name)] = FiniteElement(
            element=Lagrange(degree), convection=flux, stabilization=graph_viscosity
        )
        spatial[(f"FEM CG P{degree} Galerkin", flux_name)] = FiniteElement(
            element=Lagrange(degree), convection=flux
        )
    spatial[("FEM CG P1 + continuous interior penalty", flux_name)] = FiniteElement(
        element=Lagrange(1),
        convection=flux,
        stabilization=CIP(penalty=0.1),
        initialization=NodalInterpolation(),
    )
    for name, kind in (("SUPG", SUPG), ("GLS", GLS)):
        spatial[(f"FEM CG P1 + {name}", flux_name)] = FiniteElement(
            element=Lagrange(1),
            convection=flux,
            stabilization=kind(tau=0.5 * 10 / cells, strong_residual=InterpolatedFluxDivergence()),
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

# ## Schemes that must fail at the bore, and their stabilized counterparts
#
# Linear schemes of higher than first order oscillate at a discontinuity (Godunov's theorem). Ahead
# of the bore the water is only 0.1 m deep; the undershoot drives the depth to zero or below and the
# library stops instead of clipping. For each such scheme the matrix also runs a counterpart that
# adds exactly one ingredient; its result is shown behind a grey hatching.

stabilized_space = {
    "FVM MUSCL, unlimited": (
        "MUSCL + minmod slope limiter",
        lambda flux: FiniteVolume(
            convection=flux, reconstruction=MUSCL(), limiter=MinmodLimiter()
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
            f"FEM DG P{degree} + DG polynomial limiter (minmod slopes, shallow-water invariant domain)",
            lambda flux, degree=degree: FiniteElement(
                element=DiscontinuousLagrange(degree), convection=flux, limiter=dg_limiter
            ),
        )
        for degree in (1, 2, 3)
    },
    **{
        row: (
            f"CG P1 + invariant-domain graph viscosity instead of {row.split(' + ')[-1] if '+' in row else 'plain Galerkin'}",
            lambda flux: FiniteElement(
                element=Lagrange(1), convection=flux, stabilization=graph_viscosity
            ),
        )
        for row in ("FEM CG P1 Galerkin", "FEM CG P1 + SUPG", "FEM CG P1 + GLS")
    },
    "FDM central, 2nd order": (
        "central differences + Jameson-Schmidt-Turkel artificial dissipation",
        lambda flux: FiniteDifference(
            gradient=DissipativeFluxDifference(
                flux_difference=BoundaryFluxDifference(derivative=CentralDifference(), boundary_flux=flux),
                dissipation=JSTDissipation(),
            )
        ),
    ),
    "FDM polynomial, 4th order": (
        "4th-order differences + Jameson-Schmidt-Turkel artificial dissipation",
        lambda flux: FiniteDifference(
            gradient=DissipativeFluxDifference(
                flux_difference=BoundaryFluxDifference(derivative=PolynomialDifference(accuracy_order=4), boundary_flux=flux),
                dissipation=JSTDissipation(),
            )
        ),
    ),
    **{
        f"FDM summation by parts, order {order}": (
            f"SBP order {order} with SAT boundary flux + Jameson-Schmidt-Turkel artificial dissipation",
            lambda flux, order=order: FiniteDifference(
            gradient=DissipativeFluxDifference(
                flux_difference=BoundaryFluxDifference(derivative=SummationByPartsDifference(accuracy_order=order), boundary_flux=flux),
                dissipation=JSTDissipation(),
            )
        ),
        )
        for order in (2, 4)
    },
}

# ## The dry bed: hydrostatic providers
#
# On the dry bed every spatial family runs through its hydrostatic provider, again with every
# Riemann solver: the Audusse hydrostatic reconstruction (first order and MUSCL with three limiters
# and the positivity limiter, and WENO3 and WENO5 with the positivity limiter), the hydrostatic flux
# difference with summation-by-parts stencils of order 2 and 4, and continuous P1 and P2 elements
# with the hydrostatic graph viscosity.

wet_dry = WetDryShallowWaterDomain()
dry_spatial = {}
for flux_name, flux in riemann.items():
    dry_spatial[("FVM hydrostatic, first order", flux_name)] = FiniteVolume(
        convection=flux, provider=HydrostaticReconstruction()
    )
    for limiter_name in ("minmod", "MC", "van Leer"):
        dry_spatial[(f"FVM hydrostatic MUSCL + {limiter_name} + positivity", flux_name)] = (
            FiniteVolume(
                convection=flux,
                provider=HydrostaticReconstruction(),
                reconstruction=MUSCL(),
                limiter=limiters[limiter_name],
                positivity=ConvexInvariantDomain(),
            )
        )
    dry_spatial[("FVM hydrostatic WENO3 + positivity", flux_name)] = FiniteVolume(
        convection=flux,
        provider=HydrostaticReconstruction(),
        reconstruction=Weno3Reconstructor(),
        positivity=ConvexInvariantDomain(),
    )
    dry_spatial[("FVM hydrostatic WENO5 + positivity", flux_name)] = FiniteVolume(
        convection=flux,
        provider=HydrostaticReconstruction(),
        reconstruction=Weno5Reconstructor(),
        positivity=ConvexInvariantDomain(),
    )
    for order in (2, 4):
        dry_spatial[(f"FDM hydrostatic SBP, order {order}", flux_name)] = FiniteDifference(
            gradient=SummationByPartsDifference(accuracy_order=order),
            provider=HydrostaticFluxDifference(flux=flux, invariant_domain=wet_dry),
        )
    for degree in (1, 2):
        dry_spatial[(f"FEM CG P{degree} + hydrostatic graph viscosity", flux_name)] = (
            FiniteElement(
                element=Lagrange(degree),
                convection=flux,
                stabilization=HydrostaticGraphViscosity(
                    mass=RowSumLumping(), invariant_domain=wet_dry
                ),
                initialization=NodalInterpolation(),
            )
        )

# Time integrators: every explicit one-step and multistep method, implicit Runge-Kutta,
# Adams-Moulton and BDF methods, Rosenbrock methods and the adaptive integrators. Implicit stages
# take Newton with a coloured finite-difference Jacobian and a sparse direct solve. On the dry bed
# the implicit stages use the hydrostatic reconstruction with the wet/dry active set, which gives
# Newton the generalized Jacobian at dry cells, and the Rusanov flux, which supplies the analytic
# face derivative that the active set needs.

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
# BDF4 to BDF6 and Adams-Moulton 4 are not A-stable, and a fixed-point iteration contracts only
# for small steps: at Δt = 10⁻² they must fail; their counterpart is the same method at a stable
# step. Leapfrog has no damping; its counterpart adds the Robert-Asselin-Williams filter.
stabilized_time = {
    "leapfrog": (
        "leapfrog + Robert-Asselin-Williams filter (ν = 0.2, α = 0.53; first order), safety factor 0.15",
        TimeMethod("leapfrog_raw"),
        0.15,
    ),
}
implicit_step = 1e-2
stable_step = {
    "BDF4 (Gear)": 2.5e-3,
    "BDF5 (Gear)": 2.5e-3,
    "BDF6 (Gear)": 2.5e-3,
    "Adams-Moulton 4": 2.5e-3,
    "fixed point": 1e-3,
    "fixed point + Aitken": 1e-3,
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
    **{
        ("JFNK", linear_name): JFNK(linear_solver=linear)
        for linear_name, linear in linear_solvers.items()
    },
    ("fixed point", "none"): FixedPoint(),
    ("fixed point + Aitken", "none"): FixedPoint(acceleration="aitken"),
}

# ## The test matrix
#
# 1. **Wet bed: space × Riemann solver** — every spatial scheme with every Riemann solver, SSP-RK2
#    at the stable step (safety factor 0.5).
# 2. **Wet bed: time integrator × space** — every integrator with HLLC fluxes on first-order,
#    MUSCL-minmod and WENO5 volumes; implicit integrators at $\Delta t=10^{-2}$ s (CFL about 1.7)
#    on the first two.
# 3. **Wet bed: nonlinear × linear solver** — backward Euler on MUSCL-minmod-HLLC, $\Delta t=10^{-2}$ s.
# 4. **Dry bed: space × Riemann solver** — every hydrostatic provider with every Riemann solver.
# 5. **Dry bed: time integrator** — every integrator on the first-order hydrostatic reconstruction
#    (HLL for explicit, Rusanov with the wet/dry active set for implicit stages) and on the
#    hydrostatic MUSCL-minmod scheme for the explicit ones.
#
# A few representative combinations are repeated on 200 cells for the mesh check.

time_spatial = {
    name: spatial[(name, "HLLC")]
    for name in ("FVM first order", "FVM MUSCL + minmod", "FVM WENO5")
}
dry_time_spatial = {
    "FVM hydrostatic, first order": (
        dry_spatial[("FVM hydrostatic, first order", "HLL")],
        FiniteVolume(
            convection=Rusanov(),
            provider=HydrostaticReconstruction(dry_linearization=WetDryActiveSet()),
        ),
    ),
    "FVM hydrostatic MUSCL + minmod + positivity": (
        dry_spatial[("FVM hydrostatic MUSCL + minmod + positivity", "HLL")],
        None,
    ),
}
runs = [
    dict(case="wet", matrix="wet: space x Riemann solver", row=row, column=column,
         spatial=scheme, time=TimeMethod("ssprk2"), nonlinear=None, step=end_time)
    for (row, column), scheme in spatial.items()
]
runs += [
    dict(case="dry", matrix="dry: space x Riemann solver", row=row, column=column,
         spatial=scheme, time=TimeMethod("ssprk2"), nonlinear=None, step=end_time)
    for (row, column), scheme in dry_spatial.items()
]
runs += [
    dict(
        case="wet",
        matrix="wet: time integrator x space",
        row=name,
        column=space,
        spatial=scheme,
        time=integrator,
        nonlinear=newton if getattr(integrator, "implicit", False) and not "rosenbrock" in getattr(integrator, "method_id", "") else None,
        linear=SparseDirect() if "rosenbrock" in getattr(integrator, "method_id", "") else None,
        step=implicit_step if getattr(integrator, "implicit", False) else end_time,
    )
    for name, integrator in integrators.items()
    for space, scheme in time_spatial.items()
    if space != "FVM WENO5" or not getattr(integrator, "implicit", False)
]
runs += [
    dict(
        case="dry",
        matrix="dry: time integrator x space",
        row=name,
        column=space,
        spatial=schemes[1] if getattr(integrator, "implicit", False) else schemes[0],
        time=integrator,
        nonlinear=(
            Newton(linear_solver=SparseDirect(), jacobian=WetDryActiveSet())
            if getattr(integrator, "implicit", False) and not "rosenbrock" in getattr(integrator, "method_id", "")
            else None
        ),
        linear=SparseDirect() if "rosenbrock" in getattr(integrator, "method_id", "") else None,
        step=implicit_step if getattr(integrator, "implicit", False) else end_time,
    )
    for name, integrator in integrators.items()
    for space, schemes in dry_time_spatial.items()
    if not getattr(integrator, "implicit", False) or schemes[1] is not None
]
runs += [
    dict(case="wet", matrix="wet: nonlinear x linear solver", row=row, column=column,
         spatial=time_spatial["FVM MUSCL + minmod"], time=BackwardEuler(), nonlinear=solver,
         step=implicit_step)
    for (row, column), solver in nonlinear_solvers.items()
]
runs += [
    dict(case="wet", matrix="wet: space x Riemann solver", variant="stabilized", row=row,
         column=flux_name, spatial=make(flux), time=TimeMethod("ssprk2"), nonlinear=None,
         step=end_time, note_stabilized=description)
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
    if "time integrator" in run["matrix"]
    and run["row"] in stabilized_time
    and run.get("variant") is None
]
runs += [
    {
        **run,
        "variant": "stabilized",
        "step": stable_step[run["row"]],
        "note_stabilized": f"{run['row']} at Δt = {stable_step[run['row']]:.1e} s",
    }
    for run in list(runs)
    if ("time integrator" in run["matrix"] or "nonlinear" in run["matrix"])
    and run["row"] in stable_step
    and run.get("variant") is None
]
representative = [
    ("FVM first order", "Rusanov"),
    ("FVM first order", "HLLC"),
    ("FVM MUSCL + minmod", "HLLC"),
    ("FVM WENO5", "HLLC"),
    ("FEM DG P2 + minmod + invariant domain", "HLLC"),
    ("FEM CG P1 + graph viscosity", "HLLC"),
    ("FDM WENO5 flux splitting", "Lax-Friedrichs splitting"),
    ("FVM hydrostatic, first order", "HLL"),
    ("FVM hydrostatic MUSCL + minmod + positivity", "HLLC"),
    ("FDM hydrostatic SBP, order 2", "HLLE"),
    ("FEM CG P1 + hydrostatic graph viscosity", "HLL"),
]
runs += [
    {**run, "matrix": f"{run['case']}: mesh check", "cells": coarse_cells}
    for run in runs
    if "space x Riemann" in run["matrix"]
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
# One loop over the two problem objects. A combination the library refuses is recorded with its
# validation message; a run that stops because the depth left the physical state space is recorded
# as failed.

for run in runs:
    model, n = models[run["case"]], run.get("cells", cells)
    simulation = Simulation(
        problem=problems[run["case"]],
        mesh=meshes[n],
        numerics=Numerics(
            spatial=run["spatial"],
            linear_solver=(
                SparseDirect() if isinstance(run["spatial"], FiniteElement) else run.get("linear")
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
            points = np.column_stack((centres[n], np.zeros((n, 2))))
            run["profiles"] = np.array(
                [
                    result.field(field).evaluate(points).reshape(-1)
                    for field in (model.depth, model.discharge)
                ]
            )
            run["errors"] = tuple(
                np.abs(run["profiles"] - exact[(run["case"], n)]).sum(axis=1) * 10 / n
            )
            run["steps"] = len(result.times) - 1
            run["status"] = "ran" if np.isfinite(run["profiles"]).all() else "failed"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "failed"
                if any(
                    word in message
                    for word in ("positive depth", "nonnegative depth", "finite", "did not converge",
                                 "clock resolution", "non-positive")
                )
                else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    print(
        f"{run['matrix'][:18]:18s} {run['row'][:40]:40s} {run['column'][:24]:24s} "
        f"{run.get('variant') or '':10s} {run['status']:8s} {run['seconds']:7.1f} s  "
        + (f"L1 h {run['errors'][0]:.2e}  L1 q {run['errors'][1]:.2e}" if run["status"] == "ran"
           else run["note"][:90])
    )

# ## The matrices
#
# Every cell shows the declared combination. Where it failed and a stabilized counterpart exists,
# the cell is hatched grey and its colour and number are those of the counterpart; the legend below
# each matrix says what was added.

main = {
    (r["matrix"], r["row"], r["column"]): r
    for r in runs
    if r.get("variant") is None and "mesh check" not in r["matrix"]
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
layouts = {
    "wet: space x Riemann solver": ([*riemann, "Lax-Friedrichs splitting"], (13, 18),
        "Hatched: the declared scheme fails at the bore (depth driven to zero, Godunov's theorem); "
        "colour and value show its stabilized counterpart:"),
    "dry: space x Riemann solver": (list(riemann), (11, 6),
        "Hatched: the declared scheme fails at the dry front; colour and value show:"),
    "wet: time integrator x space": (list(time_spatial), (11, 11),
        "Hatched: the declared integrator is unstable at this step (not A-stable, or undamped); "
        "colour and value show:"),
    "dry: time integrator x space": (list(dry_time_spatial), (11, 11),
        "Hatched: the declared integrator is unstable at this step; colour and value show:"),
    "wet: nonlinear x linear solver": ([*linear_solvers, "none"], (13, 6),
        "Hatched: the fixed-point iteration cannot contract at Δt = 10⁻² s; colour and value show:"),
}
for key, (columns, size, head) in layouts.items():
    rows = list(dict.fromkeys(r["row"] for r in runs if r["matrix"] == key))
    fig = plots.figure(1, 1, size=size)
    plots.matrix(
        [[cell[(key, row, c)]["errors"][0] if (key, row, c) in cell else np.nan for c in columns]
         for row in rows],
        rows,
        columns,
        scale="log",
        label="L1 depth error at t = 0.6 s [m²]",
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

# All measures for the combinations that ran, sorted by the depth error.

for key in layouts:
    print(f"\n{key}")
    print(f"{'scheme':44s} {'column':24s} {'L1 h':>9s} {'L1 q':>9s} {'steps':>6s} {'time':>8s}")
    for r in sorted((r for k, r in cell.items() if k[0] == key and r["status"] == "ran"),
                    key=lambda r: r["errors"][0]):
        print(f"{r['row']:44s} {r['column']:24s} {r['errors'][0]:9.2e} {r['errors'][1]:9.2e} "
              f"{r['steps']:6d} {r['seconds']:7.1f}s")

# ## Profiles of representative combinations
#
# Depth and discharge at $t=0.6$ s against the exact solutions, wet bed above, dry bed below.

fig = plots.figure(2, 2, size=(15, 9))
for panel, case, index, name in (
    (fig[0], "wet", 0, "depth h [m]"),
    (fig[1], "wet", 1, "discharge q [m²/s]"),
    (fig[2], "dry", 0, "depth h [m]"),
    (fig[3], "dry", 1, "discharge q [m²/s]"),
):
    plots.line(
        y={
            f"{r['row']}, {r['column']}": (centres[cells], r["profiles"][index])
            for key, r in main.items()
            if r["case"] == case
            and "space x Riemann" in key[0]
            and (r["row"], r["column"]) in representative
            and r["status"] == "ran"
        },
        reference={("Stoker" if case == "wet" else "Ritter") + " (exact)": (fine, exact[(case, "fine")][index])},
        xlabel="position x [m]",
        ylabel=name,
        title=f"{case.capitalize()} bed, {name.split(' [')[0]} at t = {end_time} s",
        legend="upper right" if index == 0 else False,
        ax=panel,
    )
plots.show()

# ## Mesh check: 400 against 200 cells
#
# For a discontinuous solution the $L_1$ error falls roughly like $\Delta x^{1/2}$ to $\Delta x$.

print(f"{'scheme':44s} {'column':12s} {'L1 h 200':>10s} {'L1 h 400':>10s} {'ratio':>6s}")
for r in (r for r in runs if "mesh check" in r["matrix"]):
    fine_run = main[(f"{r['case']}: space x Riemann solver", r["row"], r["column"])]
    if r["status"] == "ran" and fine_run["status"] == "ran":
        print(f"{r['row']:44s} {r['column'][:12]:12s} {r['errors'][0]:10.3e} "
              f"{fine_run['errors'][0]:10.3e} {r['errors'][0] / fine_run['errors'][0]:6.2f}")

# ## What the library refused, and what failed

for status in ("rejected", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(f"  {r['matrix'][:30]:30s} {r['row'][:40]:40s} {r['column'][:16]:16s} "
              f"{r.get('variant') or '':10s} {r['note'][:90]}")
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.0f} min in total"
)

# ## Methods that do not apply to the dam break
#
# Conjugate gradients and algebraic multigrid need a symmetric positive definite operator;
# Runge-Kutta-Chebyshev and -Legendre super-steps are for parabolic operators; staggered Verlet needs
# a partitioned second-order system; IMEX splits need a stiff and a non-stiff part. Scalar convection
# fluxes (monotone upwind, central) are refused for the coupled system with a message that names a
# Riemann solver. On the dry bed, discontinuous Galerkin and plain (non-hydrostatic) schemes are not
# offered: the flat-bed model requires a positive depth, and the topographic model accepts only its
# hydrostatic providers, which keep the lake at rest and the dry cells exact. Implicit stages on the
# dry bed use the Rusanov flux because the wet/dry active set needs an analytic face derivative,
# which the HLL family does not provide.
#
# One dry-bed integrator fails without a counterpart. The adaptive Dormand-Prince RK45 method is not
# strong-stability preserving: its intermediate stages leave the set of nonnegative depths at the
# dry front, whatever the tolerance. The hydrostatic WENO3/WENO5 rows rely on the positivity limiter
# in the form of Zhang and Shu: it retracts the face depths toward the cell mean until the faces and
# the interior state of the mean are nonnegative (the explicit step is small enough for that), and
# the retracted faces keep the reconstructed velocity, so nearly dry faces carry no spurious
# discharge and every dry cell keeps exactly zero discharge.
#
# ## Going further
#
# Raise the downstream depth or give the bed a slope with `bed_elevation`; the matrix code stays as
# it is. Add a column by adding one object to a dictionary.
