"""OneSimulation example: a tubular reactor through every method combination OneSimulation offers.
Requires the installed, activated OneSimulation runtime.

One physics declaration, the axial-dispersion reactor A -> B with Danckwerts inlet and outlet
conditions, is solved on 400 cells with every applicable combination of discretization family,
reconstruction, limiter, interface flux, stabilization, time integrator, nonlinear and linear
solver, both as a stationary problem and as a transient from plug flow to the steady state. Each run is
graded against the closed-form solution of Danckwerts and Wehner & Wilhelm.
"""

# # Tubular reactor with axial dispersion: one model, every method
#
# A first-order reaction $A\to B$ runs in a tube of length $L$ in which the fluid is partly
# back-mixed. Both species are transported by the mean velocity $u$ and the axial dispersion
# coefficient $D$:
#
# $$\partial_t c_A+u\,\partial_x c_A=D\,\partial_x^2c_A-k\,c_A,\qquad
#   \partial_t c_B+u\,\partial_x c_B=D\,\partial_x^2c_B+k\,c_A .$$
#
# The reaction is declared once as a reaction network (`ReactionIR`), not as a decay rate, so
# both species are coupled through their reaction source. The Danckwerts conditions are total
# outward fluxes: the feed $u\,c_0$ of pure $A$ enters at $x=0$, and only advection leaves at
# $x=L$.
#
# **Reference.** P. V. Danckwerts, *Continuous flow systems*, Chem. Eng. Sci. 2 (1953) 1;
# J. F. Wehner & R. H. Wilhelm, Chem. Eng. Sci. 6 (1956) 89. With $\mathrm{Pe}=uL/D$,
# $\mathrm{Da}=kL/u$, $a=\sqrt{1+4\,\mathrm{Da}/\mathrm{Pe}}$ and $z=x/L$
#
# $$\frac{c_A(z)}{c_0}=\frac{2\,e^{\mathrm{Pe}\,z/2}\Big[(1+a)\,e^{a\mathrm{Pe}(1-z)/2}-(1-a)\,e^{-a\mathrm{Pe}(1-z)/2}\Big]}
#   {(1+a)^2e^{a\mathrm{Pe}/2}-(1-a)^2e^{-a\mathrm{Pe}/2}} .$$
#
# Both species have the same dispersion coefficient and $B$ is not fed, so $c_A+c_B$ obeys the
# reaction-free balance with the feed $c_0$ at the inlet: in the steady state $c_B=c_0-c_A$
# exactly.
#
# **Grading.** Every run is evaluated at the cell centres $x_i$:
# $L_1(c_A)=\sum_i|c_{A,h}(x_i)-c_A(x_i)|\,\Delta x$, and the outlet conversion
# $1-c_A(L)/c_0$. The wall time covers validation, binding and the solve.

import time
import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import (
    CentralDifference,
    CentralLaplacian,
    FiniteDifference,
    GlobalLaxFriedrichsSplit,
    PolynomialDifference,
    PolynomialLaplacian,
    SummationByPartsDifference,
    WENO5Flux,
    WENOFluxDifference,
)
from onesim.numerics.fem import (
    CIP,
    GLS,
    SUPG,
    DGJumpPenalty,
    DiscontinuousLagrange,
    FiniteElement,
    Lagrange,
    SymmetricInteriorPenalty,
    VolumeCIP,
)
from onesim.numerics.fluxes import HLL, HLLE, MonotoneUpwind, Rusanov
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
    TwoPointDiffusion,
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
from onesim.physics.chemistry import ReactionIR, SpeciesTransport, TransportSpecies
from onesim.simulation import OutwardFluxBoundary, Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# A 1 m tube, $u=1$ m/s, $\mathrm{Pe}=20$ and $\mathrm{Da}=2$. Nothing in this cell names a
# numerical method. The initial state is the plug-flow profile $c_A=c_0e^{-\mathrm{Da}\,z}$,
# $c_B=c_0-c_A$: the transients relax from it to the dispersed steady state, and the stationary
# solves start their Newton iteration from it. The kinetics are linear, so the concentrations
# are declared as signed states (`nonnegative=False`): an undershoot is graded as an error, not
# stopped as a domain violation.

length, velocity, rate, feed, peclet = 1.0, 1.0, 2.0, 1.0, 20.0
cells, coarse_cells, time_cells = 400, 200, 100
dispersion = velocity * length / peclet
tube = Line(start=(0.0, 0.0, 0.0), end=(length, 0.0, 0.0))
reactor = SpeciesTransport(
    domain=tube.line,
    velocity=velocity,
    nonnegative=False,
    species=(
        TransportSpecies("A", diffusivity=dispersion),
        TransportSpecies("B", diffusivity=dispersion),
    ),
    reactions=(
        ReactionIR(
            "A -> B", (("A", 1),), (("B", 1),), {"kind": "constant", "unit": "1/s", "value": rate}
        ),
    ),
)
plug_flow = Problem(
    name="Danckwerts tubular reactor",
    geometry=tube,
    physics=(reactor,),
    boundary_conditions=(
        OutwardFluxBoundary(
            physics=reactor,
            boundary=tube.boundaries.start,
            flux=lambda **kw: np.array([-velocity * feed, 0.0]),
            jacobian=lambda **kw: np.zeros((2, 2)),
        ),
        OutwardFluxBoundary(
            physics=reactor,
            boundary=tube.boundaries.end,
            flux=lambda interior, **kw: velocity * interior,
            jacobian=lambda **kw: velocity * np.eye(2),
            wave_speed=velocity,
        ),
    ),
    initial_conditions=(
        InitialValue(reactor.concentration("A"), lambda x: feed * np.exp(-rate * x / velocity)),
        InitialValue(reactor.concentration("B"), lambda x: feed * (1 - np.exp(-rate * x / velocity))),
    ),
)
meshes = {n: StructuredLineMesher(cells=n).generate(tube) for n in (cells, coarse_cells, time_cells)}
centres = {n: mesh.cell_centers[:, 0] for n, mesh in meshes.items()}
print(
    f"{meshes[cells].cell_count} cells of width {length / cells:.4f} m, cell Peclet number "
    f"{velocity * length / cells / dispersion:.3f} (mesh check: {meshes[coarse_cells].cell_count} cells, "
    f"time-integrator cut: {meshes[time_cells].cell_count} cells)"
)

# ## The exact solution
#
# The closed form on every grid and on a fine line for the figures.

damkoehler = rate * length / velocity
a = np.sqrt(1 + 4 * damkoehler / peclet)
fine = np.linspace(0.0, length, 2001)
exact = {
    key: feed
    * 2
    * np.exp(peclet * x / length / 2)
    * ((1 + a) * np.exp(a * peclet * (1 - x / length) / 2) - (1 - a) * np.exp(-a * peclet * (1 - x / length) / 2))
    / ((1 + a) ** 2 * np.exp(a * peclet / 2) - (1 - a) ** 2 * np.exp(-a * peclet / 2))
    for key, x in {**centres, "fine": fine}.items()
}
exact_conversion = 1 - exact["fine"][-1] / feed
print(
    f"Pe = {peclet:g}, Da = {damkoehler:g}: exact inlet c_A/c0 = {exact['fine'][0]:.6f}, outlet conversion "
    f"{exact_conversion:.6f} (stirred tank {1 - 1 / (1 + damkoehler):.4f}, plug flow {1 - np.exp(-damkoehler):.4f})"
)

# ## The method catalogue
#
# Interface fluxes for the finite volumes and discontinuous elements: Rusanov, monotone upwind, HLL
# and HLLE. Finite volumes with first-order states, unlimited and limited MUSCL (six slope
# limiters), ENO, WENO, WENO-Z, central WENO and QUICK; discontinuous Galerkin of degree 1 to 4
# with symmetric interior-penalty diffusion, with and without a jump penalty; continuous
# Galerkin of degree 1 to 3, plain and with SUPG, GLS, continuous interior penalty or the volume
# interior penalty $h\,\nu\,[u'][v']$ (with $\nu=\gamma u\Delta x$ it is the same operator as the
# continuous interior penalty, written with a numerical diffusivity, so the two rows must agree);
# finite-difference stencils of order 2 and 4, summation-by-parts operators and WENO5 flux
# splitting. Continuous elements and finite differences take no interface flux; they sit in
# their own column.

fluxes = {"Rusanov": Rusanov(), "monotone upwind": MonotoneUpwind(), "HLL": HLL(), "HLLE": HLLE()}
limiters = {
    "minmod": MinmodLimiter(),
    "MC": MonotonizedCentralLimiter(),
    "van Leer": VanLeerLimiter(),
    "superbee": SuperbeeLimiter(),
    "Barth-Jespersen": BarthJespersenLimiter(),
    "Venkatakrishnan": VenkatakrishnanLimiter(),
}
reconstructions = {
    "ENO2": Eno2Reconstructor(),
    "ENO3": Eno3Reconstructor(),
    "WENO3": Weno3Reconstructor(),
    "WENO5": Weno5Reconstructor(),
    "WENO3-Z": Weno3Reconstructor(weights=WenoZWeights(global_indicator=SmoothnessDifference())),
    "WENO5-Z": Weno5Reconstructor(weights=WenoZWeights(global_indicator=SmoothnessDifference())),
    "CWENO3": CentralWeightedENO(weights=JiangShuWeights()),
    "QUICK": QUICK(),
}
no_flux = "no interface flux"
spatial = {}
for flux_name, flux in fluxes.items():
    spatial[("FVM first order", flux_name)] = FiniteVolume(convection=flux, diffusion=TwoPointDiffusion())
    spatial[("FVM MUSCL, unlimited", flux_name)] = FiniteVolume(
        convection=flux, diffusion=TwoPointDiffusion(), reconstruction=MUSCL()
    )
    for limiter_name, limiter in limiters.items():
        spatial[(f"FVM MUSCL + {limiter_name}", flux_name)] = FiniteVolume(
            convection=flux, diffusion=TwoPointDiffusion(), reconstruction=MUSCL(), limiter=limiter
        )
    for reconstruction_name, reconstruction in reconstructions.items():
        spatial[(f"FVM {reconstruction_name}", flux_name)] = FiniteVolume(
            convection=flux, diffusion=TwoPointDiffusion(), reconstruction=reconstruction
        )
    for degree in (1, 2, 3, 4):
        spatial[(f"FEM DG P{degree}", flux_name)] = FiniteElement(
            element=DiscontinuousLagrange(degree), convection=flux, diffusion=SymmetricInteriorPenalty()
        )
    spatial[("FEM DG P1 + jump penalty", flux_name)] = FiniteElement(
        element=DiscontinuousLagrange(1),
        convection=flux,
        diffusion=SymmetricInteriorPenalty(),
        stabilization=DGJumpPenalty(coefficient=1.0),
    )
for degree in (1, 2, 3):
    spatial[(f"FEM CG P{degree} Galerkin", no_flux)] = FiniteElement(element=Lagrange(degree))
for name, kind in (("SUPG", SUPG), ("GLS", GLS)):
    spatial[(f"FEM CG P1 + {name}", no_flux)] = FiniteElement(element=Lagrange(1), stabilization=kind(tau=0.5 / cells))
for degree, penalty in ((1, 0.1), (2, 0.01)):
    spatial[(f"FEM CG P{degree} + continuous interior penalty", no_flux)] = FiniteElement(
        element=Lagrange(degree), stabilization=CIP(penalty=penalty)
    )
    spatial[(f"FEM CG P{degree} + volume interior penalty", no_flux)] = FiniteElement(
        element=Lagrange(degree), stabilization=VolumeCIP(diffusivity=penalty * velocity * length / cells)
    )
spatial[("FDM central, 2nd order", no_flux)] = FiniteDifference(gradient=CentralDifference(), diffusion=CentralLaplacian())
spatial[("FDM polynomial, 4th order", no_flux)] = FiniteDifference(
    gradient=PolynomialDifference(accuracy_order=4), diffusion=PolynomialLaplacian(accuracy_order=4)
)
for order in (2, 4):
    spatial[(f"FDM summation by parts, order {order}", no_flux)] = FiniteDifference(
        gradient=SummationByPartsDifference(accuracy_order=order), diffusion=PolynomialLaplacian(accuracy_order=order)
    )
spatial[("FDM WENO5 flux splitting", no_flux)] = FiniteDifference(
    gradient=WENOFluxDifference(flux_splitting=GlobalLaxFriedrichsSplit(), reconstruction=WENO5Flux()),
    diffusion=CentralLaplacian(),
)

# Time integrators: the explicit one-step and multistep methods, implicit Runge-Kutta,
# Adams-Moulton and BDF methods, Rosenbrock methods and the adaptive integrators. Implicit stages
# take Newton with the analytic Jacobian and a sparse direct solve; Rosenbrock stages take the
# sparse direct solve alone.

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
    "adaptive BDF": AdaptiveBDF(relative_tolerance=1e-6, absolute_tolerance=1e-9),
}
newton = Newton(linear_solver=SparseDirect(), jacobian="analytic")
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

# ## Schemes that must fail, and their stabilized counterparts
#
# Leapfrog has no damping: on the dissipative transport operator it is unstable at every step;
# its counterpart adds the Robert-Asselin-Williams filter. Adams-Moulton 3 and 4 have bounded
# stability intervals, and a fixed-point (Picard) iteration contracts only while the step stays below
# the inverse spectral radius of the stage operator; at the implicit step of the time cut they must
# fail, and their counterpart is the same method at a step it is stable for. On quadratic continuous
# elements that step lies below about $10^{-4}$ s for Adams-Moulton 4, so this cell stays marked as
# failed. The fixed-step Gear methods of order 3 to 6 start their history with extrapolated
# backward-Euler steps (`TimeMethod("bdf<k>_gear", startup_policy="implicit_extrapolation")`), so
# they run at the implicit step on every spatial family.

end_time, implicit_step = 2.5, 0.02
stabilized_time = {
    "leapfrog": (
        "leapfrog + Robert-Asselin-Williams filter (ν = 0.2, α = 0.53; first order), safety factor 0.15",
        TimeMethod("leapfrog_raw"),
        0.15,
    ),
}
stable_step = {
    "Adams-Moulton 3": 2e-3,
    "Adams-Moulton 4": 2e-3,
    "fixed point": 2e-4,
    "fixed point + Aitken": 2e-4,
}

# ## The test matrix
#
# 1. **Space × interface flux** — every spatial scheme, stationary solve with Newton on 400 cells.
# 2. **Time integrator × space** — every integrator from plug flow to $t=2.5$ s (the slowest
#    mode decays at least at the rate $k+u^2/4D=7$ 1/s, so the profile is steady to $10^{-7}$) on 100 cells. Explicit
#    integrators run at their stable step (safety factor 0.5) on first-order and WENO5 volumes
#    and fourth-order differences; implicit integrators run with $\Delta t=0.02$ s, beyond every
#    explicit limit, on first-order and MUSCL-minmod volumes, fourth-order differences and
#    quadratic continuous elements.
# 3. **Nonlinear × linear solver** — backward Euler on MUSCL-minmod-Rusanov volumes, 100 cells,
#    $\Delta t=0.02$ s up to $t=0.5$ s, graded by the deviation from the Newton/LU run; and the
#    stationary problem on quadratic elements, solved by Jacobian-free Newton-Krylov with each inner
#    linear solver and by each linear solver alone (the reaction is linear, so the stationary system
#    is affine).
#
# Representative combinations are repeated on 200 cells for the mesh check.

time_spatial = {
    "FVM first order": spatial[("FVM first order", "Rusanov")],
    "FVM MUSCL + minmod": spatial[("FVM MUSCL + minmod", "Rusanov")],
    "FVM WENO5": spatial[("FVM WENO5", "Rusanov")],
    "FDM polynomial, 4th order": spatial[("FDM polynomial, 4th order", no_flux)],
    "FEM CG P2": spatial[("FEM CG P2 Galerkin", no_flux)],
}
explicit_columns = ("FVM first order", "FVM WENO5", "FDM polynomial, 4th order")
implicit_columns = ("FVM first order", "FVM MUSCL + minmod", "FDM polynomial, 4th order", "FEM CG P2")
runs = [
    dict(matrix="space x flux", row=row, column=column, cells=cells, spatial=scheme, study="stationary", nonlinear=newton)
    for (row, column), scheme in spatial.items()
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
        end=end_time,
        nonlinear=newton if getattr(integrator, "implicit", False) and "rosenbrock" not in getattr(integrator, "method_id", "") else None,
        linear=SparseDirect() if "rosenbrock" in getattr(integrator, "method_id", "") or isinstance(time_spatial[space], FiniteElement) else None,
        step=implicit_step if getattr(integrator, "implicit", False) else end_time,
    )
    for name, integrator in integrators.items()
    for space in (implicit_columns if getattr(integrator, "implicit", False) else explicit_columns)
]
runs += [
    dict(
        matrix="nonlinear x linear solver",
        row=row,
        column=column,
        cells=time_cells,
        spatial=time_spatial["FVM MUSCL + minmod"],
        study="transient",
        time=BackwardEuler(),
        end=0.5,
        nonlinear=solver,
        step=implicit_step,
    )
    for (row, column), solver in nonlinear_solvers.items()
]
runs += [
    dict(
        matrix="nonlinear x linear solver",
        row="stationary JFNK",
        column=name,
        cells=cells,
        spatial=time_spatial["FEM CG P2"],
        study="stationary",
        nonlinear=JFNK(linear_solver=solver),
    )
    for name, solver in linear_solvers.items()
]
runs += [
    dict(
        matrix="nonlinear x linear solver",
        row="stationary, linear solve only",
        column=name,
        cells=cells,
        spatial=time_spatial["FEM CG P2"],
        study="stationary",
        nonlinear=None,
        linear=solver,
    )
    for name, solver in linear_solvers.items()
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
    {**run, "variant": "stabilized", "step": stable_step[run["row"]], "note_stabilized": f"{run['row']} at Δt = {stable_step[run['row']]:.1e}"}
    for run in list(runs)
    if run["matrix"] in ("time integrator x space", "nonlinear x linear solver")
    and run["row"] in stable_step
    and run.get("variant") is None
]
# Unpreconditioned GMRES variants stall on the stationary transport system within their default
# 1000 iterations; their counterpart adds an incomplete-LU preconditioner.
preconditioned = {
    "GMRES": GMRES(preconditioner=ILU()),
    "right GMRES": RightGMRES(preconditioner=ILU()),
    "flexible GMRES": FlexibleGMRES(preconditioner=ILU()),
}
runs += [
    {**run, "variant": "stabilized", "linear": preconditioned[run["column"]], "note_stabilized": f"{run['column']} + ILU preconditioner"}
    for run in list(runs)
    if run["matrix"] == "nonlinear x linear solver" and run["nonlinear"] is None and run["column"] in preconditioned
]
representative = [
    ("FVM first order", "Rusanov"),
    ("FVM MUSCL + minmod", "Rusanov"),
    ("FVM WENO5", "Rusanov"),
    ("FEM DG P2", "Rusanov"),
    ("FEM CG P1 Galerkin", no_flux),
    ("FEM CG P2 Galerkin", no_flux),
    ("FDM central, 2nd order", no_flux),
    ("FDM polynomial, 4th order", no_flux),
]
runs += [
    {**run, "matrix": "mesh check", "cells": coarse_cells}
    for run in runs
    if run["matrix"] == "space x flux" and (run["row"], run["column"]) in representative
]
print(f"{len(runs)} runs: " + ", ".join(f"{sum(r['matrix'] == m for r in runs)} {m}" for m in dict.fromkeys(r["matrix"] for r in runs)))

# ## Run everything
#
# One loop. A combination the library refuses is recorded with its validation message; a run
# that diverges is recorded as failed.

for run in runs:
    simulation = Simulation(
        problem=plug_flow,
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
    run.update(status="rejected", note="", error=np.nan, conversion=np.nan, steps=0)
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = (
                simulation.solve_stationary()
                if run["study"] == "stationary"
                else simulation.run(start_time=0.0, end_time=run["end"], time_step=run["step"])
            )
            points = np.column_stack((centres[run["cells"]], np.zeros((run["cells"], 2))))
            run["profiles"] = np.array(
                [result.field(reactor.concentration(name)).evaluate(points).reshape(-1) for name in ("A", "B")]
            )
            run["outlet"] = float(result.field(reactor.concentration("A")).evaluate(np.array([[length, 0.0, 0.0]])).reshape(-1)[0])
            run["error"] = float(np.abs(run["profiles"][0] - exact[run["cells"]]).sum() * length / run["cells"])
            run["balance"] = float(np.abs(run["profiles"].sum(axis=0) - feed).max())
            run["conversion"] = 1 - run["outlet"] / feed
            run["steps"] = 0 if run["study"] == "stationary" else len(result.times) - 1
            # A run that ends farther from the solution than the feed itself has diverged.
            run["status"] = "ran" if np.isfinite(run["profiles"]).all() and run["error"] < feed * length else "failed"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "failed"
                if any(word in message for word in ("non-positive", "nonnegative", "finite", "did not converge", "maximum iteration", "unstable", "no stable"))
                else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    print(
        f"{run['matrix'][:12]:12s} {run['row'][:34]:34s} {run['column'][:26]:26s} {run.get('variant') or '':10s} "
        f"{run['status']:8s} {run['seconds']:7.1f} s  "
        + (f"L1 c_A {run['error']:.2e}  conversion {run['conversion']:.6f}" if run["status"] == "ran" else run["note"][:90])
    )

# The Newton/LU run of the solver cut is the discrete reference of that cut.

main = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") is None and r["matrix"] != "mesh check"}
substitute = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") == "stabilized"}
hatch = {key for key, r in main.items() if r["status"] != "ran" and substitute.get(key, {}).get("status") == "ran"}
cell = {key: substitute[key] if key in hatch else r for key, r in main.items()}
solver_reference = main[("nonlinear x linear solver", "Newton", "sparse LU")]
for r in runs:
    if r["matrix"] == "nonlinear x linear solver" and r["status"] == "ran":
        r["deviation"] = (
            float(np.abs(r["profiles"][0] - solver_reference["profiles"][0]).max())
            if r["study"] == "transient" and r["cells"] == solver_reference["cells"]
            else r["error"]
        )

# ## Space × interface flux
#
# The $L_1$ error of $c_A$ for every spatial scheme with every interface flux. Continuous elements
# and finite differences take no interface flux.

space_key = "space x flux"
matrix = [r for r in runs if r["matrix"] == space_key and r.get("variant") is None]
rows, columns = list(dict.fromkeys(r["row"] for r in matrix)), [*fluxes, no_flux]
grid = {(r["row"], r["column"]): r for r in matrix}
fig = plots.figure(1, 1, size=(13, 15))
plots.matrix(
    [[cell[(space_key, row, c)]["error"] if (space_key, row, c) in cell else np.nan for c in columns] for row in rows],
    rows,
    columns,
    scale="log",
    label="L1 error of c_A, stationary",
    text=[
        [
            (f"{cell[(space_key, row, c)]['error']:.2e}" if cell[(space_key, row, c)]["status"] == "ran" else cell[(space_key, row, c)]["status"])
            if (space_key, row, c) in cell
            else "–"
            for c in columns
        ]
        for row in rows
    ],
    title=f"Space x interface flux, {cells} cells, stationary Newton solve",
    ax=fig[0],
)
plots.show()

# All measures for the combinations that ran, sorted by the error. The species balance
# $\max|c_A+c_B-c_0|$ shows that every scheme carries the reaction source of both species
# consistently.

print(f"{'scheme':40s} {'flux':18s} {'L1 c_A':>9s} {'conversion':>11s} {'|cA+cB-c0|':>11s} {'time':>8s}")
for r in sorted((r for r in matrix if r["status"] == "ran"), key=lambda r: r["error"]):
    print(f"{r['row']:40s} {r['column']:18s} {r['error']:9.2e} {r['conversion']:11.6f} {r['balance']:11.1e} {r['seconds']:7.2f}s")

# ## Profiles of representative combinations
#
# Both species along the tube against the closed form, and a close-up of the inlet jump.

shown = [r for r in matrix if (r["row"], r["column"]) in representative and r["status"] == "ran"]
fig = plots.figure(1, 2, size=(16, 6))
plots.line(
    y={f"{r['row']}, {r['column']}": (centres[cells], r["profiles"][0]) for r in shown},
    reference={"exact c_A": (fine, exact["fine"]), "exact c_B": (fine, feed - exact["fine"])},
    xlabel="position x [m]",
    ylabel="concentration / c0",
    title="c_A along the tube (dashed: exact)",
    legend="upper right",
    ax=fig[0],
)
plots.line(
    y={f"{r['row']}, {r['column']}": (centres[cells], r["profiles"][0]) for r in shown},
    reference={"exact c_A": (fine, exact["fine"])},
    xlabel="position x [m]",
    ylabel="c_A / c0",
    xlimits=(0.0, 0.05),
    ylimits=(0.85, 0.95),
    title="Inlet jump (Danckwerts condition)",
    legend=False,
    ax=fig[1],
)
plots.show()

# ## Time integrator × space
#
# From plug flow to the steady state. Explicit integrators at their stable step, implicit
# integrators at $\Delta t=0.02$ s. A dash marks a combination outside this cut (explicit steps on
# MUSCL volumes or continuous elements, implicit steps on WENO5).

time_key = "time integrator x space"
timing = [r for r in runs if r["matrix"] == time_key and r.get("variant") is None]
time_rows, time_columns = list(dict.fromkeys(r["row"] for r in timing)), list(time_spatial)
time_hatched = [[(time_key, row, c) in hatch for c in time_columns] for row in time_rows]
time_notes = (
    ["Hatched: the declared integrator is unstable at this step (not A-stable, or undamped); colour and value show its stabilized counterpart:"]
    + [f"  {row}: same method at Δt = {stable_step[row]:.1e} instead of {implicit_step}." for row in stable_step if any((time_key, row, c) in hatch for c in time_columns)]
    + [f"  {row}: {description}." for row, (description, _, _) in stabilized_time.items() if any((time_key, row, c) in hatch for c in time_columns)]
)
fig = plots.figure(1, 2, size=(18, 11))
plots.matrix(
    [[cell[(time_key, row, c)]["error"] if (time_key, row, c) in cell else np.nan for c in time_columns] for row in time_rows],
    time_rows,
    time_columns,
    scale="log",
    label="L1 error of c_A at t = 2.5 s",
    text=[
        [
            (f"{cell[(time_key, row, c)]['error']:.2e}" if cell[(time_key, row, c)]["status"] == "ran" else cell[(time_key, row, c)]["status"])
            if (time_key, row, c) in cell
            else "–"
            for c in time_columns
        ]
        for row in time_rows
    ],
    hatched=time_hatched,
    notes=time_notes,
    title=f"L1 error at t = {end_time:g} s, {time_cells} cells",
    ax=fig[0],
)
plots.matrix(
    [
        [cell[(time_key, row, c)]["seconds"] if (time_key, row, c) in cell and cell[(time_key, row, c)]["status"] == "ran" else np.nan for c in time_columns]
        for row in time_rows
    ],
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
# Backward Euler on MUSCL-minmod volumes: the discrete solution must not depend on the solver
# pair as long as it converges (value: largest deviation from the Newton/LU run at $t=0.5$ s);
# the wall time does. The last two rows solve the stationary problem on quadratic elements with
# Jacobian-free Newton-Krylov and with each linear solver alone (value: $L_1$ error against the
# closed form).

solver_key = "nonlinear x linear solver"
solving = [r for r in runs if r["matrix"] == solver_key and r.get("variant") is None]
solver_rows, solver_columns = list(dict.fromkeys(r["row"] for r in solving)), [*linear_solvers, "none"]
fig = plots.figure(1, 1, size=(14, 7))
plots.matrix(
    [
        [cell[(solver_key, row, c)]["seconds"] if (solver_key, row, c) in cell and cell[(solver_key, row, c)]["status"] == "ran" else np.nan for c in solver_columns]
        for row in solver_rows
    ],
    solver_rows,
    solver_columns,
    scale="log",
    label="wall time [s]",
    text=[
        [
            (f"{cell[(solver_key, row, c)]['seconds']:.1f} s\n{cell[(solver_key, row, c)]['deviation']:.1e}" if cell[(solver_key, row, c)]["status"] == "ran" else cell[(solver_key, row, c)]["status"])
            if (solver_key, row, c) in cell
            else ""
            for c in solver_columns
        ]
        for row in solver_rows
    ],
    hatched=[[(solver_key, row, c) in hatch for c in solver_columns] for row in solver_rows],
    notes=["Hatched: the declared pair does not converge; colour and value show its stabilized counterpart:"]
    + [f"  {row}: Δt = {stable_step[row]:.1e} instead of {implicit_step}." for row in ("fixed point", "fixed point + Aitken") if any((solver_key, row, c) in hatch for c in solver_columns)]
    + [f"  {column}: + incomplete-LU preconditioner." for column in preconditioned if any((solver_key, row, column) in hatch for row in solver_rows)],
    title="Wall time and deviation per solver pair",
    ax=fig[0],
)
plots.show()

# ## Mesh check: 400 against 200 cells
#
# The representative combinations on half as many cells. The error ratio estimates the order of
# accuracy as $\log_2$ of the ratio.

coarse = {(r["row"], r["column"]): r for r in runs if r["matrix"] == "mesh check"}
print(f"{'scheme':36s} {'flux':18s} {'L1 200':>10s} {'L1 400':>10s} {'order':>6s}")
for key, r in coarse.items():
    fine_run = grid[key]
    if r["status"] == "ran" and fine_run["status"] == "ran":
        print(f"{key[0]:36s} {key[1]:18s} {r['error']:10.3e} {fine_run['error']:10.3e} {np.log2(r['error'] / fine_run['error']):6.2f}")

# ## What the library refused, and what failed

for status in ("rejected", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(f"  {r['matrix'][:12]:12s} {r['row'][:34]:34s} {r['column'][:24]:24s} {r.get('variant') or '':10s} {r['note'][:100]}")
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.0f} min in total"
)

# ## Methods that do not apply to this reactor
#
# HLLC restores a contact wave between two outer signals; a scalar transport law has a single
# wave and no contact, and the library says so. `LinearUpwind` and `CentralDifference` from
# `onesim.numerics.fvm` are cell-face schemes of unstructured meshes (velocity-field upwinding and
# a corrected face gradient); on a line their counterparts are the MUSCL reconstruction with an
# upwind flux and two-point diffusion, both in the matrix. Discontinuous elements of degree 0
# carry no gradient for interior-penalty diffusion. Conjugate gradients and algebraic multigrid
# need a symmetric operator, which advection does not give. Staggered Verlet needs a
# second-order system; Runge-Kutta-Chebyshev super-steps are for purely parabolic operators.
#
# ## Going further
#
# Raise the Peclet number until the central schemes oscillate, or replace the reaction by a
# second-order or reversible network: the matrix code stays as it is.
