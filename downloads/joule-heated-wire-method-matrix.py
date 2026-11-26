"""OneSimulation example: a Joule-heated wire through every method combination.
Requires the installed, activated OneSimulation runtime.

Two physics declarations, current conduction in a stainless-steel wire and Fourier heat
conduction in the same wire, are coupled both ways: the Joule power density heats the wire, and
the temperature lowers the electrical conductivity. The coupled problem
is solved with every applicable combination of discretization family (finite differences, finite
volumes, continuous, Nitsche and discontinuous finite elements), coupling strategy (Gauss-Seidel
fixed point, monolithic Newton, quasi-stationary and partitioned time coupling), time integrator,
nonlinear solver and linear solver. Each run is graded against the exact steady temperature and a
fine reference of the transient; combinations the library refuses are reported as such.
"""

# # A Joule-heated wire: two coupled models, every method
#
# A stainless-steel wire of length $L=20\,$mm lies between two massive clamps that hold both ends
# at $T_0=300\,$K. At $t=0$ a voltage $U=50\,$mV is switched on across the clamps. The current
# heats the wire, the heat flows back into the clamps, and the hot steel conducts worse. Two physics
# models describe it:
#
# $$-\partial_x\big(\sigma(T)\,\partial_x\varphi\big)=0,\qquad \varphi(0)=U,\quad \varphi(L)=0,$$
#
# $$\rho c_p\,\partial_t T=\partial_x\big(k\,\partial_x T\big)+q,\qquad q=\sigma(T)\,|\partial_x\varphi|^2,
# \qquad T(0,t)=T(L,t)=T_0,\quad T(x,0)=T_0 .$$
#
# The potential obeys current continuity: `CurrentConduction` with a conductivity that follows the
# linear resistivity law of metals, $\rho_e(T)=\rho_0\,[1+\alpha\,(T-T_0)]$ with
# $\rho_0=1/\sigma_0$, $\sigma_0=1.4\cdot10^6\,$S/m and $\alpha=1.0\cdot10^{-3}\,$1/K (AISI 304). The law
# is declared as a material law, `ReciprocalLaw` of a `PolynomialLaw` in the temperature. Two
# `FieldCoupling`s close the loop: the temperature of the heat balance enters the temperature input
# of the conduction model, and its ohmic loss density $q=\sigma(T)\,|E|^2$ enters the heat balance.
# The steel has $k=16\,$W/(m K), $\rho=7900\,$kg/m³ and $c_p=500\,$J/(kg K). The electric field
# responds within nanoseconds, the temperature within seconds, so the potential is a stationary
# partner of the transient heat balance.
#
# Everything numerical is an argument of `Numerics` (one per model) and of the coupling solver.
#
# **Reference, steady.** The current density $J$ is uniform. With $\theta=1+\alpha(T-T_0)$ the heat
# balance $kT''+J^2\rho_e(T)=0$ becomes $\theta''+m^2\theta=0$, $m^2=\alpha\rho_0J^2/k$, with
# $\theta=1$ at both clamps, and $U=J\int_0^L\rho_e\,dx$ fixes $m$:
#
# $$\theta(x)=\frac{\cos m(x-L/2)}{\cos(mL/2)},\qquad \tan\frac{mL}{2}=\frac{U}{2}\sqrt{\frac{\alpha\sigma_0}{k}},
# \qquad T_s(x)=T_0+\frac{\theta(x)-1}{\alpha}.$$
#
# For $\alpha\to0$ this is the parabola $T_0+\sigma_0U^2x(L-x)/(2kL^2)$ with the peak rise $\sigma_0U^2/8k$,
# Kohlrausch's result that the rise depends on the voltage, not on the length (*Ann. Phys.* 1
# (1900) 132); the feedback lowers it.
#
# **Reference, transient.** The transient has no closed form. It is computed in the notebook on
# 2000 intervals with second-order differences and the BDF2 method at $\Delta t=0.5\,$ms, the Joule
# density extrapolated linearly in time and the current from the resistance
# $\int\rho_e\,dx$ by the trapezoidal rule. Both errors are below $10^{-5}\,$K, far below the
# errors of the graded runs.
#
# **Grading.** Every run is evaluated at the cell centres $x_i$: the mean absolute error of the
# temperature and its largest deviation, for the steady state and at $t=5\,$s (Fourier number
# $\alpha t/L^2\approx0.05$); the wall time covers validation, binding and solution.

import time

import numpy as np
from onesim.cad.d1 import Line
from onesim.materials import IsotropicSolid, PolynomialLaw, ReciprocalLaw
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.coupled_systems import (
    MonolithicStationaryCoupling,
    PartitionedCoupling,
    QuasiStationaryCoupling,
    StationaryPartitionedCoupling,
)
from onesim.numerics.fdm import (
    CentralDifference,
    CentralLaplacian,
    FiniteDifference,
    PolynomialDifference,
    PolynomialLaplacian,
)
from onesim.numerics.field_transfer import FieldInterpolation
from onesim.numerics.fem import (
    CIP,
    DiscontinuousLagrange,
    FiniteElement,
    Lagrange,
    Nitsche,
    SymmetricInteriorPenalty,
)
from onesim.numerics.fvm import CentralDifference as CellCentralDifference
from onesim.numerics.fvm import FiniteVolume, LeastSquaresGradient, TwoPointDiffusion
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
from onesim.physics.electromagnetics import CurrentConduction
from onesim.physics.electrostatics import FixedPotential
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_connections import FieldCoupling
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu

theme.set()

# ## The physics, declared once
#
# Nothing in this cell names a numerical method. Both models live on the same line and share the
# mesh.

length, voltage, conductivity, coefficient, ambient, end_time = 0.02, 0.05, 1.4e6, 1.0e-3, 300.0, 5.0
cells, coarse_cells, time_cells = 200, 100, 50
steel = IsotropicSolid(
    density=7900.0,
    specific_heat_capacity=500.0,
    thermal_conductivity=16.0,
    youngs_modulus=1.93e11,
    poisson_ratio=0.29,
    thermal_expansion_coefficient=1.6e-5,
)
diffusivity = steel.thermal_conductivity / (steel.density * steel.specific_heat_capacity)
resistivity = PolynomialLaw(
    (1 / conductivity, coefficient / conductivity), unit="Ohm*m", reference=ambient
)
wire = Line(start=(0.0, 0.0, 0.0), end=(length, 0.0, 0.0))
heat = HeatConduction(domain=wire.line, material=steel)
current = CurrentConduction(
    domain=wire.line, conductivity=ReciprocalLaw(resistivity, unit="S/m")
)
clamps = (wire.boundaries.start, wire.boundaries.end)
problem = Problem(
    name="Joule-heated wire",
    geometry=wire,
    physics=(heat, current),
    boundary_conditions=(
        *(FixedTemperature(physics=heat, boundary=clamp, value=ambient) for clamp in clamps),
        FixedPotential(physics=current, boundary=clamps[0], value=voltage),
        FixedPotential(physics=current, boundary=clamps[1], value=0.0),
    ),
    couplings=(
        FieldCoupling(
            source=current.loss_density, target=heat.heating, transfer=FieldInterpolation()
        ),
        FieldCoupling(
            source=heat.temperature, target=current.temperature, transfer=FieldInterpolation()
        ),
    ),
    initial_conditions=(InitialValue(heat.temperature, ambient),),
)
meshes = {
    n: StructuredLineMesher(cells=n).generate(wire) for n in (cells, coarse_cells, time_cells)
}
centres = {n: mesh.cell_centers for n, mesh in meshes.items()}
print(
    f"{meshes[cells].cell_count} cells of width {1e3 * length / cells:.2f} mm, "
    f"{meshes[cells].cell_count + 1} mesh nodes (mesh check: {meshes[coarse_cells].cell_count} "
    f"cells; time-integrator cut: {meshes[time_cells].cell_count} cells)"
)
print(
    f"Joule density at T0 {conductivity * (voltage / length) ** 2:.4e} W/m³, "
    f"diffusivity {diffusivity:.4e} m²/s, "
    f"Fourier number at t = {end_time} s: {diffusivity * end_time / length**2:.4f}"
)

# ## The reference solutions
#
# The closed form of the steady state at every cell centre, and the transient on the fine grid,
# interpolated to the cell centres.

wavenumber = 2 / length * np.arctan(
    voltage / 2 * np.sqrt(coefficient * conductivity / steel.thermal_conductivity)
)
fine = np.linspace(0.0, length, 2001)
points = {**{n: c[:, 0] for n, c in centres.items()}, "fine": fine}
steady = {
    key: ambient
    + (np.cos(wavenumber * (x - length / 2)) / np.cos(wavenumber * length / 2) - 1) / coefficient
    for key, x in points.items()
}
spacing, reference_step = fine[1], 5e-4
capacity = steel.density * steel.specific_heat_capacity
stiffness = steel.thermal_conductivity * diags(
    [1.0, -2.0, 1.0], [-1, 0, 1], shape=(len(fine) - 2, len(fine) - 2)
) / spacing**2
first_step = splu((capacity / reference_step * identity(len(fine) - 2) - stiffness).tocsc())
bdf2_step = splu((1.5 * capacity / reference_step * identity(len(fine) - 2) - stiffness).tocsc())
rise, previous = np.zeros(len(fine)), np.zeros(len(fine))
for n in range(round(end_time / reference_step)):
    guess = rise if n == 0 else 2 * rise - previous
    local_resistivity = (1 + coefficient * guess) / conductivity
    density = (voltage / np.trapezoid(local_resistivity, fine)) ** 2 * local_resistivity
    history = rise if n == 0 else 2 * rise - 0.5 * previous
    updated = np.zeros(len(fine))
    updated[1:-1] = (first_step if n == 0 else bdf2_step).solve(
        capacity / reference_step * history[1:-1] + density[1:-1]
    )
    previous, rise = rise, updated
transient = {key: ambient + np.interp(x, fine, rise) for key, x in points.items()}
print(
    f"steady peak rise {steady['fine'].max() - ambient:.4f} K "
    f"(without feedback σ0U²/8k = {conductivity * voltage**2 / (8 * steel.thermal_conductivity):.4f} K); "
    f"rise at the centre after {end_time} s: {np.interp(length / 2, fine, transient['fine']) - ambient:.4f} K"
)

# ## The method catalogue
#
# Both models take the same spatial scheme. Finite differences with central, fourth- and
# sixth-order polynomial Laplacians, each with the first derivative of the same order for the
# electric field and the heat flux; finite volumes with the two-point flux and with the corrected
# central flux; continuous Lagrange elements of degree 1 to 3 with strong boundary values,
# symmetric and nonsymmetric Nitsche terms and continuous interior penalty; discontinuous
# elements of degree 1 to 3 with the symmetric interior-penalty flux at two penalty levels.

spatial = {
    "FDM central, 2nd order": FiniteDifference(
        diffusion=CentralLaplacian(), gradient=CentralDifference()
    ),
    **{
        f"FDM polynomial, {order}th order": FiniteDifference(
            diffusion=PolynomialLaplacian(accuracy_order=order),
            gradient=PolynomialDifference(accuracy_order=order),
        )
        for order in (4, 6)
    },
    "FVM two-point flux": FiniteVolume(diffusion=TwoPointDiffusion()),
    "FVM corrected central flux": FiniteVolume(
        diffusion=CellCentralDifference(), gradient=LeastSquaresGradient()
    ),
    **{f"FEM CG P{degree}": FiniteElement(element=Lagrange(degree)) for degree in (1, 2, 3)},
    "FEM CG P2 + symmetric Nitsche": FiniteElement(
        element=Lagrange(2), boundary_enforcement=Nitsche()
    ),
    "FEM CG P2 + nonsymmetric Nitsche": FiniteElement(
        element=Lagrange(2), boundary_enforcement=Nitsche(symmetry=-1.0)
    ),
    "FEM CG P2 + continuous interior penalty": FiniteElement(
        element=Lagrange(2), stabilization=CIP(penalty=0.01)
    ),
    **{
        f"FEM DG P{degree} SIPG": FiniteElement(
            element=DiscontinuousLagrange(degree), diffusion=SymmetricInteriorPenalty()
        )
        for degree in (1, 2, 3)
    },
    "FEM DG P2 SIPG, penalty 40": FiniteElement(
        element=DiscontinuousLagrange(2),
        diffusion=SymmetricInteriorPenalty(penalty=40.0),
    ),
}

# Coupling strategies for the steady state: a Gauss-Seidel fixed point that solves one model after
# the other and passes the loss density and the temperature on (in both orders: heat first starts
# without a source), the monolithic Newton method on the stacked residuals of both models, and the
# same Newton method without the cross blocks (block Jacobi). The monolithic cross blocks hold the
# derivative of the current balance with respect to the temperature, through $d\sigma/dT$, and of
# the heat balance with respect to the potential; the share of the heat block that runs through
# the loss density, $\partial q/\partial T=|E|^2\,d\sigma/dT$, is added to the heat block. Both are
# assembled by colored differences.
#
# Coupling strategies in time: the potential as a quasi-stationary partner solved at every
# synchronization of the transient heat balance, and a partitioned step that iterates both models
# to a fixed point per step.

newton_tolerances = dict(relative_tolerance=1e-10, absolute_tolerance=1e-8)
stationary_couplings = {
    "Gauss-Seidel, potential first": StationaryPartitionedCoupling(
        stationary_systems=(current, heat),
        iteration=FixedPoint(relative_tolerance=1e-10, maximum_iterations=20),
    ),
    "Gauss-Seidel, heat first": StationaryPartitionedCoupling(
        stationary_systems=(heat, current),
        iteration=FixedPoint(relative_tolerance=1e-10, maximum_iterations=20),
    ),
    "monolithic Newton": MonolithicStationaryCoupling(
        stationary_systems=(heat, current),
        nonlinear_solver=Newton(linear_solver=SparseDirect(), **newton_tolerances),
    ),
    "block-Jacobi Newton": MonolithicStationaryCoupling(
        stationary_systems=(heat, current),
        nonlinear_solver=Newton(linear_solver=SparseDirect(), **newton_tolerances),
        coupling_jacobian="none",
    ),
}
time_couplings = {
    "quasi-stationary": QuasiStationaryCoupling(
        transient_systems=(heat,),
        stationary_systems=(current,),
        iteration=FixedPoint(maximum_iterations=20),
    ),
    "partitioned": PartitionedCoupling(
        transient_systems=(heat,),
        stationary_systems=(current,),
        iteration=FixedPoint(maximum_iterations=20),
    ),
}

# Time integrators of the heat balance: the explicit one-step and multistep methods, implicit
# Runge-Kutta, Adams-Moulton and BDF methods, Rosenbrock methods and the adaptive integrators.
# Implicit stages take a Newton solver with a sparse direct solve; the linearly implicit
# Rosenbrock stages take the sparse direct solve alone. Explicit methods advance at their stable
# step (safety factor 0.5), implicit ones at $\Delta t=0.05\,$s.

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
    "adaptive RK45": AdaptiveRungeKutta45(relative_tolerance=1e-5, absolute_tolerance=1e-6),
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
    "adaptive BDF": AdaptiveBDF(relative_tolerance=1e-4, absolute_tolerance=1e-3),
}
implicit_step = 0.05
newton = Newton(linear_solver=SparseDirect())

# ## Schemes that must fail, and their stabilized counterparts
#
# The heat operator has a real, negative spectrum reaching $-\lambda_{\max}\sim-4\alpha/h^2$ (more
# for higher orders and interior penalty). Adams-Moulton 3 and 4 are not A-stable: their stability
# intervals on the negative axis end at $\lambda\Delta t\approx6$ and $3$. Leapfrog has no damping:
# every decaying mode is amplified at any step. A fixed-point (Picard) iteration of an implicit
# step contracts only for $\lambda_{\max}\Delta t<1$. BDF5 and BDF6 are stable on the negative axis
# but not monotone at large $\lambda\Delta t$. Crank-Nicolson is A-stable but not L-stable: the
# stiffest modes are multiplied by almost $-1$ in every step. Each of these may fail; its stabilized
# counterpart, shown behind a grey hatching, is the same method at a step it is stable or damping
# for, or leapfrog with the Robert-Asselin-Williams filter. A counterpart runs only where the
# declared scheme actually failed.

stabilized_time = {
    "leapfrog": (
        "leapfrog + Robert-Asselin-Williams filter (ν = 0.2, α = 0.53), safety factor 0.1",
        TimeMethod("leapfrog_raw"),
        0.1,
    ),
}
stable_step = {
    "Adams-Moulton 3": 1e-2,
    "Adams-Moulton 4": 5e-3,
    "Crank-Nicolson": 5e-3,
    "BDF5 (Gear)": 5e-3,
    "BDF6 (Gear)": 5e-3,
}
linear_solvers = {
    "sparse LU": SparseDirect(),
    "conjugate gradients": ConjugateGradient(),
    "CG + algebraic multigrid": ConjugateGradient(preconditioner=AlgebraicMultigrid()),
    "GMRES": GMRES(),
    "GMRES + algebraic multigrid": GMRES(preconditioner=AlgebraicMultigrid()),
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
            ("JFNK", JFNK),
        )
        for linear_name, linear in linear_solvers.items()
    },
    ("fixed point", "none"): FixedPoint(maximum_iterations=200),
    ("fixed point + Aitken", "none"): FixedPoint(maximum_iterations=200, acceleration="aitken"),
}
stable_step.update({"fixed point": 5e-3, "fixed point + Aitken": 5e-3})

# ## The test matrix
#
# Five cuts through the space of combinations:
#
# 1. **Space × coupling strategy** — the steady state of every spatial scheme on 200 cells with the
#    four stationary coupling strategies.
# 2. **Time integrator × space** — the transient with every integrator on three spatial schemes
#    (central differences, two-point volumes and CG P1), the potential as quasi-stationary partner.
#    An explicit method on a diffusion operator must take $\Delta t\propto h^2$, so this cut uses
#    50 cells; the spatial accuracy is the subject of the first cut.
# 3. **Time coupling × integrator** — the three time-coupling strategies with four implicit
#    integrators on CG P2, 50 cells.
# 4. **Nonlinear × linear solver** — backward Euler on two-point finite volumes with 50 cells,
#    quasi-stationary coupling; the linear solver also solves the potential. Finite volumes, not
#    differences: with $\sigma(T)$ the difference form $\sigma\varphi''+\sigma'\varphi'$ of the
#    current balance is not symmetric, so conjugate gradients would not apply to its potential,
#    while the two-point flux keeps it symmetric positive definite.
# 5. **Mesh check** — the steady state of representative schemes with the monolithic Newton method
#    on 100 cells against 200 cells.

time_spatial = {
    name: spatial[name] for name in ("FDM central, 2nd order", "FVM two-point flux", "FEM CG P1")
}
coupling_integrators = {
    name: integrators[name] for name in ("backward Euler", "Crank-Nicolson", "BDF2 (Gear)", "SDIRK2")
}
runs = [
    dict(
        matrix="space x coupling",
        row=row,
        column=column,
        cells=cells,
        spatial=scheme,
        coupling=coupling,
        stationary=True,
    )
    for row, scheme in spatial.items()
    for column, coupling in stationary_couplings.items()
]
runs += [
    dict(
        matrix="time integrator x space",
        row=name,
        column=space,
        cells=time_cells,
        spatial=scheme,
        coupling=time_couplings["quasi-stationary"],
        time=integrator,
        step=implicit_step if getattr(integrator, "implicit", False) else end_time,
    )
    for name, integrator in integrators.items()
    for space, scheme in time_spatial.items()
]
runs += [
    dict(
        matrix="time coupling x integrator",
        row=name,
        column=column,
        cells=time_cells,
        spatial=spatial["FEM CG P2"],
        coupling=coupling,
        time=integrator,
        step=implicit_step,
    )
    for name, integrator in coupling_integrators.items()
    for column, coupling in time_couplings.items()
]
runs += [
    dict(
        matrix="nonlinear x linear solver",
        row=row,
        column=column,
        cells=time_cells,
        spatial=spatial["FVM two-point flux"],
        coupling=time_couplings["quasi-stationary"],
        time=BackwardEuler(),
        nonlinear=solver,
        linear=linear_solvers.get(column, SparseDirect()),
        step=implicit_step,
    )
    for (row, column), solver in nonlinear_solvers.items()
]
representative = [
    "FDM central, 2nd order",
    "FDM polynomial, 4th order",
    "FVM two-point flux",
    "FEM CG P1",
    "FEM CG P2",
    "FEM DG P1 SIPG",
    "FEM DG P2 SIPG",
]
runs += [
    {**run, "matrix": "mesh check", "cells": coarse_cells}
    for run in runs
    if run["matrix"] == "space x coupling"
    and run["row"] in representative
    and run["column"] == "monolithic Newton"
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
# its validation message. A run whose temperature leaves the interval between $T_0$ and the steady
# peak (plus 1 K) violates the maximum principle of the heated wire and is recorded as failed, as
# is a solver that does not converge. A failed run with a stabilized counterpart appends that
# counterpart to the list, so the same loop runs it next.

for run in runs:
    integrator = run.get("time")
    implicit = getattr(integrator, "implicit", False)
    rosenbrock = "rosenbrock" in getattr(integrator, "method_id", "")
    linear = run.get("linear", SparseDirect())
    simulation = Simulation(
        problem=problem,
        mesh={heat: meshes[run["cells"]], current: meshes[run["cells"]]},
        numerics={
            heat: (
                Numerics(spatial=run["spatial"], linear_solver=linear)
                if run.get("stationary")
                else Numerics(
                    spatial=run["spatial"],
                    time_integrator=integrator,
                    linear_solver=(
                        linear
                        if rosenbrock
                        or (not implicit and isinstance(run["spatial"], FiniteElement))
                        else None
                    ),
                    nonlinear_solver=(
                        run.get("nonlinear", newton) if implicit and not rosenbrock else None
                    ),
                    step_limits=TimeStepLimits(stability_safety_factor=run.get("safety", 0.5)),
                )
            ),
            current: Numerics(spatial=run["spatial"], linear_solver=linear),
        },
        coupling_solver=run["coupling"],
    )
    reference = steady if run.get("stationary") else transient
    started = time.perf_counter()
    check = simulation.validate()
    run.update(status="rejected", note="", error=np.nan, peak=np.nan, steps=0)
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = (
                simulation.solve_stationary()
                if run.get("stationary")
                else simulation.run(start_time=0.0, end_time=end_time, time_step=run["step"])
            )
            run["profile"] = result.field(heat.temperature).evaluate(centres[run["cells"]]).ravel()
            run["error"] = np.abs(run["profile"] - reference[run["cells"]]).mean()
            run["peak"] = np.abs(run["profile"] - reference[run["cells"]]).max()
            run["rise"] = (
                result.field(heat.temperature).evaluate(np.array([[length / 2, 0.0, 0.0]])).item()
                - ambient
            )
            run["steps"] = 0 if run.get("stationary") else len(result.times) - 1
            inside = np.isfinite(run["profile"]).all() and (
                ambient - 1.0 <= run["profile"].min()
                and run["profile"].max() <= steady["fine"].max() + 1.0
            )
            run["status"] = "ran" if inside else "failed"
            run["note"] = "" if inside else "temperature leaves [T0, T_peak] (unstable)"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "failed"
                if "unstable" in message
                or "finite" in message.replace("definite", "")
                or "converge" in message
                or "stagnat" in message
                else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    if run["status"] == "failed" and run.get("variant") is None and run["matrix"] != "mesh check":
        if run["row"] in stabilized_time:
            runs.append(
                {
                    **run,
                    "variant": "stabilized",
                    "time": stabilized_time[run["row"]][1],
                    "safety": stabilized_time[run["row"]][2],
                    "note_stabilized": stabilized_time[run["row"]][0],
                }
            )
        elif run["row"] in stable_step or run["column"] in stable_step:
            runs.append(
                {
                    **run,
                    "variant": "stabilized",
                    "step": stable_step.get(run["row"], stable_step.get(run["column"])),
                    "note_stabilized": f"Δt = {stable_step.get(run['row'], stable_step.get(run['column'])):.0e} s",
                }
            )
    print(
        f"{run['matrix'][:12]:12s} {run['row'][:34]:34s} {run['column'][:24]:24s} "
        f"{run.get('variant') or '':10s} {run['status']:8s} {run['seconds']:7.1f} s  "
        + (
            f"mean |T-T_exact| {run['error']:.2e} K  steps {run['steps']}"
            if run["status"] == "ran"
            else run["note"][:90]
        )
    )

# ## Space × coupling strategy
#
# The mean error of the steady temperature for every spatial scheme and every stationary coupling
# strategy. All four strategies converge to the same discrete solution of a scheme, so the error
# measures the spatial scheme alone. The steady profile is a cosine arc, no longer the parabola of
# the uncoupled wire, so higher-order schemes converge fast but do not reproduce it to rounding.

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

space_key = "space x coupling"
rows, columns = list(spatial), list(stationary_couplings)
fig = plots.figure(1, 1, size=(12, 10))
plots.matrix(
    [
        [max(cell[(space_key, row, c)]["error"], 1e-12) for c in columns]
        for row in rows
    ],
    rows,
    columns,
    scale="log",
    label="mean |T - T_s| [K]",
    text=[
        [
            (
                f"{cell[(space_key, row, c)]['error']:.2e}"
                if cell[(space_key, row, c)]["status"] == "ran"
                else cell[(space_key, row, c)]["status"]
            )
            for c in columns
        ]
        for row in rows
    ],
    hatched=[[(space_key, row, c) in hatch for c in columns] for row in rows],
    title=f"Steady state: space x coupling strategy, {cells} cells",
    ax=fig[0],
)
plots.show()

# All measures for the combinations that ran, sorted by the mean error.

matrix = [r for r in runs if r["matrix"] == space_key]
print(f"{'scheme':40s} {'coupling':30s} {'mean err':>9s} {'max err':>9s} {'time':>8s}")
for r in sorted((r for r in matrix if r["status"] == "ran"), key=lambda r: r["error"]):
    print(
        f"{r['row']:40s} {r['column']:30s} {r['error']:9.2e} {r['peak']:9.2e} {r['seconds']:7.1f}s"
    )

# ## Steady profiles of representative combinations
#
# Temperature along the wire against the exact cosine arc, and a close-up of the clamp at $x=0$.

shown = [
    r
    for r in matrix
    if r["row"] in representative and r["column"] == "monolithic Newton" and r["status"] == "ran"
]
fig = plots.figure(1, 2, size=(15, 5))
for panel, window, title in (
    (fig[0], None, "Steady temperature"),
    (fig[1], (0.0, 2.0), "Close-up of the clamp"),
):
    plots.line(
        y={r["row"]: (1e3 * centres[cells][:, 0], r["profile"]) for r in shown},
        reference={"exact steady state": (1e3 * fine, steady["fine"])},
        xlabel="position x [mm]",
        ylabel="temperature [K]",
        title=title,
        xlimits=window,
        ylimits=None if window is None else (ambient, ambient + 12.0),
        legend="lower center" if window is None else False,
        ax=panel,
    )
plots.show()

# ## Time integrator × space
#
# The transient at $t=5\,$s with the potential as quasi-stationary partner. Explicit integrators at
# their stable step, implicit integrators at $\Delta t=0.05\,$s, on 50 cells.

time_key = "time integrator x space"
time_rows, time_columns = list(integrators), list(time_spatial)
time_hatched = [[(time_key, row, c) in hatch for c in time_columns] for row in time_rows]
time_notes = (
    [
        "Hatched: the declared integrator is unstable at this step (not A-stable, or undamped); "
        "colour and value show its stabilized counterpart:"
    ]
    + [
        f"  {row}: same method at Δt = {stable_step[row]:.0e} s instead of {implicit_step} s."
        for row in stable_step
        if any((time_key, row, c) in hatch for c in time_columns)
    ]
    + [
        f"  {row}: {description}."
        for row, (description, _, _) in stabilized_time.items()
        if any((time_key, row, c) in hatch for c in time_columns)
    ]
)
fig = plots.figure(1, 2, size=(17, 12))
for panel, measure, label, title in (
    (fig[0], "error", "mean |T - T_exact| [K]", f"Mean temperature error at t = {end_time} s"),
    (fig[1], "seconds", "wall time [s]", "Wall time and accepted steps"),
):
    plots.matrix(
        [
            [
                cell[(time_key, row, c)][measure]
                if cell[(time_key, row, c)]["status"] == "ran"
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

# Transient profiles of three integrators on central differences against the series.

profiles = [
    r
    for r in runs
    if r["matrix"] == time_key
    and r["column"] == "FDM central, 2nd order"
    and r["row"] in ("forward Euler", "backward Euler", "BDF2 (Gear)")
    and r["status"] == "ran"
]
fig = plots.figure(1, 1, size=(9, 5))
plots.line(
    y={r["row"]: (1e3 * centres[time_cells][:, 0], r["profile"]) for r in profiles},
    reference={
        f"series, t = {end_time} s": (1e3 * fine, transient["fine"]),
        "steady state": (1e3 * fine, steady["fine"]),
    },
    xlabel="position x [mm]",
    ylabel="temperature [K]",
    title=f"Heating up: temperature at t = {end_time} s",
    legend="lower center",
    ax=fig[0],
)
plots.show()

# ## Time coupling × integrator
#
# Quasi-stationary and partitioned coupling of the transient with four implicit integrators on
# CG P2. Both converge the potential and the conductivity at every synchronization; differences
# would show a coupling error.

coupling_key = "time coupling x integrator"
coupling_rows, coupling_columns = list(coupling_integrators), list(time_couplings)
fig = plots.figure(1, 1, size=(11, 5))
plots.matrix(
    [
        [
            cell[(coupling_key, row, c)]["error"]
            if cell[(coupling_key, row, c)]["status"] == "ran"
            else np.nan
            for c in coupling_columns
        ]
        for row in coupling_rows
    ],
    coupling_rows,
    coupling_columns,
    scale="log",
    label="mean |T - T_exact| [K]",
    text=[
        [
            (
                f"{cell[(coupling_key, row, c)]['error']:.2e}\n"
                f"{cell[(coupling_key, row, c)]['seconds']:.1f} s"
            )
            if cell[(coupling_key, row, c)]["status"] == "ran"
            else cell[(coupling_key, row, c)]["status"]
            for c in coupling_columns
        ]
        for row in coupling_rows
    ],
    hatched=[[(coupling_key, row, c) in hatch for c in coupling_columns] for row in coupling_rows],
    notes=(
        [
            "Hatched: Crank-Nicolson does not damp the stiff modes of the switch-on; colour and "
            f"value show it at Δt = {stable_step['Crank-Nicolson']:.0e} s."
        ]
        if any((coupling_key, "Crank-Nicolson", c) in hatch for c in coupling_columns)
        else None
    ),
    title=f"Time coupling x integrator, CG P2, {time_cells} cells, Δt = {implicit_step} s",
    ax=fig[0],
)
plots.show()

# ## Nonlinear × linear solver
#
# Backward Euler on two-point finite volumes with 50 cells. The heat step is linear for a given
# Joule density, so every converging Newton variant reproduces the same discrete solution; the
# wall time differs. The linear solver of a column solves both the heat step and the potential.

solver_key = "nonlinear x linear solver"
solving = [r for r in runs if r["matrix"] == solver_key and r.get("variant") is None]
solver_rows = list(dict.fromkeys(r["row"] for r in solving))
solver_columns = [*linear_solvers, "none"]
fig = plots.figure(1, 1, size=(15, 6))
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
                    f"{cell[(solver_key, row, c)]['error']:.2e}"
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
    notes=(
        [
            f"Hatched: the fixed-point (Picard) iteration cannot contract at Δt = {implicit_step} s; "
            "colour and value show the same iteration at the step it converges for:"
        ]
        + [
            f"  {row}: Δt = {stable_step[row]:.0e} s."
            for row in ("fixed point", "fixed point + Aitken")
            if any((solver_key, row, c) in hatch for c in solver_columns)
        ]
    )
    if any((solver_key, row, "none") in hatch for row in ("fixed point", "fixed point + Aitken"))
    else None,
    title="Backward Euler: wall time and mean temperature error per solver pair",
    ax=fig[0],
)
plots.show()

# ## Mesh check: 200 against 100 cells
#
# The steady state with the monolithic Newton method on half as many cells. The key value is the
# temperature rise at the middle of the wire (exact: $(1/\cos(mL/2)-1)/\alpha=26.980\,$K); its change
# between the two levels and the error ratio show the order of each scheme: about four for second
# order and sixteen for quadratic elements. The higher-order differences converge more slowly than
# their interior order: at the clamps their conductivity reads the temperature interpolated from
# the nearest point values.

coarse = {r["row"]: r for r in runs if r["matrix"] == "mesh check"}
grid = {r["row"]: r for r in matrix if r["column"] == "monolithic Newton"}
print(
    f"{'scheme':32s} {'rise 100':>10s} {'rise 200':>10s} {'change':>9s} "
    f"{'err 100':>10s} {'err 200':>10s} {'ratio':>6s}"
)
for name, r in coarse.items():
    if r["status"] == "ran" and grid[name]["status"] == "ran":
        print(
            f"{name:32s} {r['rise']:10.5f} {grid[name]['rise']:10.5f} "
            f"{grid[name]['rise'] - r['rise']:9.2e} {r['error']:10.3e} {grid[name]['error']:10.3e} "
            f"{r['error'] / max(grid[name]['error'], 1e-300):6.2f}"
        )

# ## What the library refused, and what failed
#
# A refused combination is caught by validation or binding before any step; a failed one started
# and left the physical range or did not converge.

for status in ("rejected", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(
            f"  {r['matrix'][:12]:12s} {r['row'][:34]:34s} {r['column'][:24]:24s} "
            f"{r.get('variant') or '':10s} {r['note'][:90]}"
        )
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.0f} min in total"
)

# ## Methods that do not apply to this problem
#
# Riemann solvers, reconstructions and slope limiters act on convective fluxes; neither the
# current nor the heat balance has one, so finite volumes take only a diffusive face flux. SUPG,
# GLS and graph viscosity stabilize convection and have nothing to act on. Staggered Verlet needs
# a second-order system and IMEX splits a stiff from a non-stiff part, which a single diffusion
# operator does not have. A monolithic time integrator (`MonolithicCoupling`) advances transient
# fields only; the potential has no time derivative, so in time it is coupled as a stationary
# partner. `SequentialCoupling` visits every model once per step in the order of its one-way
# dependencies; the two-way coupling makes the dependency graph cyclic, so the strategies in time
# are the quasi-stationary and the partitioned coupling, which converge the loop at every step.
#
# ## Going further
#
# Replace a `FixedTemperature` by a `ConvectiveHeatFlux` to cool the wire surface, change the
# voltage to check that the peak rise grows with $U^2$ and not with the length, or put a second
# material into the middle of the wire with `regionwise`; the matrix code stays as it is.
