"""OneSimulation example: paraxial diffraction of a Gaussian beam through every method combination.
Requires the installed, activated OneSimulation runtime.

One physics declaration, the paraxial envelope equation with a Gaussian input beam, is solved on
400 cells with every applicable combination of discretization family, field representation, time
integrator, nonlinear solver and linear solver. Each run is graded against the exact Gaussian beam:
its complex amplitude and its width w(z).
"""

# # A diffracting Gaussian beam: one model, every method
#
# A monochromatic beam of wavelength $\lambda=1\,\mu$m enters a homogeneous medium with a Gaussian
# transverse profile of waist $w_0=20\,\mu$m. In the frame that travels with the group velocity
# $v_g$, the slowly varying envelope $A(x,t)$ obeys the paraxial (Schrödinger-type) equation
#
# $$\partial_t A=\frac{i\,v_g}{2k}\,\partial_x^2 A,\qquad z=v_g t,$$
#
# with one transverse coordinate $x$ and the wavenumber $k=2\pi/\lambda$. The operator is purely
# dispersive: every Fourier mode keeps its modulus and only turns its phase. The equation is declared
# once; everything numerical is an argument of `Numerics`: the discretization family (finite
# volumes, finite differences, continuous and discontinuous finite elements), the representation of
# the envelope (one complex field, or a real and an imaginary field), the time integrator and, for
# implicit steps, the nonlinear and the linear solver.
#
# **Reference.** The exact solution for $A(x,0)=E_0\,e^{-x^2/w_0^2}$ (Kogelnik and Li, *Appl. Opt.*
# 5, 1550, 1966, reduced to one transverse dimension) is
#
# $$A(x,z)=\frac{E_0}{\sqrt{1+i z/z_R}}\exp\!\Big(-\frac{x^2}{w_0^2\,(1+i z/z_R)}\Big),\qquad
# z_R=\frac{k w_0^2}{2},\qquad w(z)=w_0\sqrt{1+(z/z_R)^2}.$$
#
# The beam is followed to $z=2z_R$, where its width has grown by $\sqrt5$ and its peak amplitude has
# fallen by $5^{-1/4}$. The transverse window $|x|\le 12\,w_0$ holds the beam with amplitude below
# $10^{-12}$ at its edges, so the zero envelope prescribed there does not disturb it.
#
# **Grading.** At $z=2z_R$ every run is evaluated at the 400 cell centres $x_i$: the relative
# $L_2$ error of the complex amplitude, $\|A_h-A\|_2/\|A\|_2$, and the error of the beam width from
# the second moment, $w_h^2=4\sum x_i^2|A_h|^2/\sum|A_h|^2$. The wall time covers validation,
# binding and time stepping.

import time

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import CentralLaplacian, FiniteDifference, PolynomialLaplacian
from onesim.numerics.fem import (
    GLS,
    DiscontinuousLagrange,
    FiniteElement,
    GalerkinDiffusion,
    Lagrange,
    Nitsche,
    SymmetricInteriorPenalty,
)
from onesim.numerics.fvm import FiniteVolume, TwoPointDiffusion
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
    TimeMethod,
)
from onesim.physics.optics import ComplexParaxialEnvelope, ParaxialEnvelope
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.simulation.problem_definition.boundaries import PrescribedValueBoundary
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# The same beam in both field representations. Nothing in this cell names a numerical method.

wavelength, waist, amplitude, group_speed = 1.0e-6, 20.0e-6, 1.0e3, 2.99792458e8
wavenumber = 2 * np.pi / wavelength
rayleigh = wavenumber * waist**2 / 2
distance = 2 * rayleigh
end_time, cells, coarse_cells = distance / group_speed, 400, 200
window = Line(start=(-12 * waist, 0.0, 0.0), end=(12 * waist, 0.0, 0.0))
medium = dict(
    domain=window.line,
    wavenumber=wavenumber,
    group_speed=group_speed,
    permittivity=8.8541878128e-12,
)
envelopes = {
    "complex envelope": ComplexParaxialEnvelope(**medium),
    "real + imaginary pair": ParaxialEnvelope(**medium),
}
beams = {
    "complex envelope": Problem(
        name="Gaussian beam, complex envelope",
        geometry=window,
        physics=(envelopes["complex envelope"],),
        boundary_conditions=(
            PrescribedValueBoundary(
                physics=envelopes["complex envelope"],
                boundary=window.boundaries.all,
                value={"envelope": 0j},
            ),
        ),
        initial_conditions=(
            InitialValue(
                envelopes["complex envelope"].envelope,
                lambda x: amplitude * np.exp(-((x / waist) ** 2)) + 0j,
            ),
        ),
    ),
    "real + imaginary pair": Problem(
        name="Gaussian beam, real and imaginary fields",
        geometry=window,
        physics=(envelopes["real + imaginary pair"],),
        boundary_conditions=(
            PrescribedValueBoundary(
                physics=envelopes["real + imaginary pair"],
                boundary=window.boundaries.all,
                value={"real": 0.0, "imaginary": 0.0},
            ),
        ),
        initial_conditions=(
            InitialValue(
                envelopes["real + imaginary pair"].real,
                lambda x: amplitude * np.exp(-((x / waist) ** 2)),
            ),
            InitialValue(envelopes["real + imaginary pair"].imaginary, 0.0),
        ),
    ),
}
meshes = {n: StructuredLineMesher(cells=n).generate(window) for n in (cells, coarse_cells)}
centres = {n: mesh.cell_centers[:, 0] for n, mesh in meshes.items()}
print(
    f"{meshes[cells].cell_count} cells of width {24 * waist / cells * 1e6:.2f} µm "
    f"(coarse check: {meshes[coarse_cells].cell_count} cells); Rayleigh length "
    f"{rayleigh * 1e3:.3f} mm, propagation to z = {distance * 1e3:.3f} mm (t = {end_time:.3e} s)"
)

# ## The exact beam
#
# The complex amplitude at $z=2z_R$ at the cell centres and on a fine grid for the plots, and the
# exact width.

fine = np.linspace(-12 * waist, 12 * waist, 2401)
exact = {
    key: amplitude
    / np.sqrt(1 + 1j * distance / rayleigh)
    * np.exp(-((points / waist) ** 2) / (1 + 1j * distance / rayleigh))
    for key, points in {**centres, "fine": fine}.items()
}
exact_width = waist * np.sqrt(1 + (distance / rayleigh) ** 2)
print(
    f"exact width at z = 2 z_R: {exact_width * 1e6:.4f} µm (sqrt(5) w0), peak |A| = "
    f"{abs(exact['fine']).max():.3f} V/m, edge |A| = {abs(exact['fine'][0]):.1e} V/m"
)

# ## The method catalogue
#
# Finite volumes with the two-point flux; finite differences with the central and the 4th- and
# 6th-order polynomial Laplacians; continuous finite elements of degree 1 to 3 with the Galerkin
# diffusion form and the prescribed envelope imposed weakly by Nitsche's method (symmetric,
# incomplete and nonsymmetric variants), and with a Galerkin least-squares term; discontinuous
# finite elements of degree 1 to 3 with the symmetric interior penalty.
#
# **Explicit step.** The paraxial operator has an imaginary spectrum, so an explicit step must lie
# inside the part of the time method's stability region that covers a segment of the imaginary axis.
# Every spatial scheme is given the same rule, `SpectralEnclosureStability(enclosure="imaginary_axis")`:
# it estimates the spectral radius of the executed spatial operator (Arnoldi iteration on its
# residual) and returns the largest step for which the selected method's amplification stays below
# one on that segment. Implicit methods are not limited by it; their step is $\Delta t=t_{end}/100$.

rule = SpectralEnclosureStability(enclosure="imaginary_axis", spectral_radius="arnoldi")
spatial = {
    "FVM two-point flux": FiniteVolume(diffusion=TwoPointDiffusion(), stability=rule),
    "FDM central, 2nd order": FiniteDifference(diffusion=CentralLaplacian(), stability=rule),
    **{
        f"FDM polynomial, {order}th order": FiniteDifference(
            diffusion=PolynomialLaplacian(accuracy_order=order), stability=rule
        )
        for order in (4, 6)
    },
    **{
        f"FEM CG P{degree} + Nitsche": FiniteElement(
            element=Lagrange(degree),
            diffusion=GalerkinDiffusion(),
            boundary_enforcement=Nitsche(),
            stability=rule,
        )
        for degree in (1, 2, 3)
    },
    **{
        f"FEM CG P2 + {name} Nitsche": FiniteElement(
            element=Lagrange(2),
            diffusion=GalerkinDiffusion(),
            boundary_enforcement=Nitsche(symmetry=symmetry),
            stability=rule,
        )
        for name, symmetry in (("incomplete", 0.0), ("nonsymmetric", -1.0))
    },
    "FEM CG P2 + Nitsche + GLS": FiniteElement(
        element=Lagrange(2),
        diffusion=GalerkinDiffusion(),
        boundary_enforcement=Nitsche(),
        stabilization=GLS(tau=0.05 * end_time / 100),
        stability=rule,
    ),
    **{
        f"FEM DG P{degree} + SIPG": FiniteElement(
            element=DiscontinuousLagrange(degree),
            diffusion=SymmetricInteriorPenalty(),
            stability=rule,
        )
        for degree in (1, 2, 3)
    },
}

# Time integrators: explicit one-step and multistep methods, the leapfrog scheme with and without
# the Robert-Asselin-Williams filter, implicit Runge-Kutta, Adams-Moulton and BDF methods,
# Rosenbrock methods and the adaptive integrators. Implicit stages take a Newton solver with a
# coloured finite-difference Jacobian and a sparse direct solve; the linearly implicit Rosenbrock
# stages take the sparse direct solve alone.

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
    "leapfrog + RAW filter": TimeMethod("leapfrog_raw"),
    "adaptive RK45": AdaptiveRungeKutta45(minimum_step=1e-9 * end_time),
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
    "adaptive BDF": AdaptiveBDF(
        relative_tolerance=1e-3, absolute_tolerance=1e-1, minimum_step=1e-9 * end_time
    ),
}
implicit_step = end_time / 100
newton = Newton(linear_solver=SparseDirect(), jacobian="colored_finite_difference")
# The fixed-step Gear BDF methods of order 3 to 6 and Adams-Moulton 3 and 4 build their first
# history values with explicit RK4 steps at the full step size. On the stiff transverse spectrum
# of the finite elements that start-up is unstable, so these methods must fail at Δt = t_end/100.
# The stabilized counterpart of a Gear BDF method is the same order with backward-Euler start-up
# (`BDF(order=k)`, one ingredient changed); Adams-Moulton 3 and 4 (whose stability regions are
# also bounded) and the fixed-point iteration (which contracts only while Δt times the spectral
# radius stays below one) are repeated at Δt = t_end/3200. Counterparts are shown behind the
# hatching. On the discontinuous elements, whose penalty modes make the spectral radius far larger,
# Adams-Moulton 3 and 4 stay unstable even at that step.
gear_startup = {f"BDF{k} (Gear)": BDF(order=k) for k in (3, 4, 5, 6)}
stable_step = {
    name: end_time / 3200
    for name in ("Adams-Moulton 3", "Adams-Moulton 4", "fixed point", "fixed point + Aitken")
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
# Krylov solvers without a preconditioner may stall on the stage system; their counterpart adds an
# ILU preconditioner (one ingredient).
preconditioned = {
    (name, linear_name): (
        JFNK(linear_solver=kind(preconditioner=ILU()))
        if name == "JFNK"
        else solver_kind(
            linear_solver=kind(preconditioner=ILU()), jacobian="colored_finite_difference"
        )
    )
    for name, solver_kind in (
        ("Newton", Newton),
        ("damped Newton", DampedNewton),
        ("trust-region Newton", TrustRegionNewton),
        ("quasi-Newton", QuasiNewton),
        ("JFNK", JFNK),
    )
    for linear_name, kind in (
        ("GMRES", GMRES),
        ("right GMRES", RightGMRES),
        ("flexible GMRES", FlexibleGMRES),
    )
}

# ## The test matrix
#
# Three cuts through the space of combinations:
#
# 1. **Space × representation and time** — every spatial scheme with the complex envelope and with
#    the real/imaginary pair, each advanced implicitly with Crank-Nicolson at
#    $\Delta t=t_{end}/100$ and explicitly with classical RK4 at its stable step.
# 2. **Time integrator × space** — every integrator on the 4th-order finite differences and on
#    continuous and discontinuous P1 elements, complex envelope.
# 3. **Nonlinear × linear solver** — Crank-Nicolson on the 4th-order finite differences, for the
#    complex envelope and for the real/imaginary pair.
#
# Explicit steps of finite elements of degree 2 and 3, and of the GLS-stabilized elements, are
# limited by the Nitsche and interior-penalty boundary modes, whose eigenvalues grow like
# $p^4/h^2$: at 400 cells they need tens of thousands of steps. The library runs them (the
# stability rule gives the step), but they are outside this cut and marked with a dash.
#
# A few representative combinations are repeated on 200 cells for the mesh check.

columns = {
    f"{representation}, {label}": (representation, method)
    for representation in envelopes
    for label, method in (
        ("RK4", ClassicalRungeKutta4()),
        ("Crank-Nicolson", TimeMethod("crank_nicolson")),
    )
}
time_spatial = {
    name: spatial[name]
    for name in ("FDM polynomial, 4th order", "FEM CG P1 + Nitsche", "FEM DG P1 + SIPG")
}
explicit_rows = {
    name
    for name in spatial
    if name.startswith(("FVM", "FDM")) or name in ("FEM CG P1 + Nitsche", "FEM DG P1 + SIPG")
}
runs = [
    dict(
        matrix="space x representation",
        row=row,
        column=column,
        representation=representation,
        cells=cells,
        spatial=scheme,
        time=method,
    )
    for row, scheme in spatial.items()
    for column, (representation, method) in columns.items()
    if getattr(method, "implicit", False) or row in explicit_rows
]
runs += [
    dict(
        matrix="time integrator x space",
        row=name,
        column=space,
        representation="complex envelope",
        cells=cells,
        spatial=scheme,
        time=integrator,
    )
    for name, integrator in integrators.items()
    for space, scheme in time_spatial.items()
    if getattr(integrator, "implicit", False) or space in explicit_rows
]
runs += [
    dict(
        matrix=f"nonlinear x linear solver, {representation}",
        row=row,
        column=column,
        representation=representation,
        cells=cells,
        spatial=spatial["FDM polynomial, 4th order"],
        time=TimeMethod("crank_nicolson"),
        nonlinear=solver,
    )
    for representation in envelopes
    for (row, column), solver in nonlinear_solvers.items()
]
runs += [
    {
        **run,
        "variant": "stabilized",
        "step": stable_step[run["row"]],
        "note_stabilized": f"{run['row']} at Δt = t_end/{round(end_time / stable_step[run['row']])}",
    }
    for run in list(runs)
    if run["matrix"] != "space x representation" and run["row"] in stable_step
]
runs += [
    {**run, "variant": "stabilized", "time": gear_startup[run["row"]]}
    for run in list(runs)
    if run["matrix"] == "time integrator x space"
    and run["row"] in gear_startup
    and run.get("variant") is None
]
runs += [
    {**run, "variant": "stabilized", "nonlinear": preconditioned[(run["row"], run["column"])]}
    for run in list(runs)
    if run["matrix"].startswith("nonlinear")
    and (run["row"], run["column"]) in preconditioned
    and run.get("variant") is None
]
representative = [
    ("FVM two-point flux", "complex envelope, Crank-Nicolson"),
    ("FDM central, 2nd order", "complex envelope, Crank-Nicolson"),
    ("FDM polynomial, 4th order", "complex envelope, RK4"),
    ("FEM CG P1 + Nitsche", "complex envelope, RK4"),
    ("FEM CG P2 + Nitsche", "complex envelope, Crank-Nicolson"),
    ("FEM DG P2 + SIPG", "complex envelope, Crank-Nicolson"),
]
runs += [
    {**run, "matrix": "mesh check", "cells": coarse_cells}
    for run in runs
    if run["matrix"] == "space x representation"
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
# A stabilized counterpart is run only where the declared combination did not run.
# One loop over the two problem objects. A combination the library refuses is recorded with its
# validation message. Two kinds of failure are expected: an explicit method whose stability region
# contains no segment of the imaginary axis has no stable step at all (the stability rule says so
# before the first step), and a method that is not A-stable amplifies the high transverse modes at a
# large implicit step (the amplitude grows instead of keeping its modulus).

for run in runs:
    if run.get("variant") == "stabilized" and any(
        other.get("variant") is None
        and other.get("status") == "ran"
        and (other["matrix"], other["row"], other["column"])
        == (run["matrix"], run["row"], run["column"])
        for other in runs
    ):
        run.update(status="not needed", note="", error=np.nan, width=np.nan, steps=0)
        run["seconds"] = 0.0
        continue
    model = envelopes[run["representation"]]
    implicit = getattr(run["time"], "implicit", False)
    rosenbrock = "rosenbrock" in getattr(run["time"], "method_id", "")
    linear = SparseDirect()
    simulation = Simulation(
        problem=beams[run["representation"]],
        mesh=meshes[run["cells"]],
        numerics=Numerics(
            spatial=run["spatial"],
            linear_solver=(
                linear if isinstance(run["spatial"], FiniteElement) or rosenbrock else None
            ),
            time_integrator=run["time"],
            nonlinear_solver=(
                run.get("nonlinear", newton) if implicit and not rosenbrock else None
            ),
        ),
    )
    started = time.perf_counter()
    check = simulation.validate()
    run.update(status="rejected", note="", error=np.nan, width=np.nan, steps=0)
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = simulation.run(
                start_time=0.0,
                end_time=end_time,
                time_step=run.get("step", implicit_step if implicit else end_time),
            )
            points = np.column_stack((centres[run["cells"]], np.zeros((run["cells"], 2))))
            run["profile"] = (
                result.field(model.envelope).evaluate(points).reshape(-1)
                if run["representation"] == "complex envelope"
                else result.field(model.real).evaluate(points).reshape(-1)
                + 1j * result.field(model.imaginary).evaluate(points).reshape(-1)
            )
            run["error"] = np.linalg.norm(run["profile"] - exact[run["cells"]]) / np.linalg.norm(
                exact[run["cells"]]
            )
            run["width"] = np.sqrt(
                4
                * np.sum(centres[run["cells"]] ** 2 * abs(run["profile"]) ** 2)
                / np.sum(abs(run["profile"]) ** 2)
            )
            run["steps"] = len(result.times) - 1
            run["status"] = "ran" if np.isfinite(run["error"]) and run["error"] < 1 else "failed"
            run["note"] = "" if run["status"] == "ran" else "amplitude grew (unstable modes)"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "no stable step"
                if "unstable for a nonzero imaginary" in message
                else (
                    "failed"
                    if "finite" in message
                    or "did not converge" in message
                    or "maximum iteration" in message
                    else "rejected"
                )
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    print(
        f"{run['matrix'][:12]:12s} {run['row'][:30]:30s} {run['column'][:36]:36s} "
        f"{run.get('variant') or '':10s} {run['status']:14s} {run['seconds']:7.1f} s  "
        + (
            f"L2 {run['error']:.2e}  width {(run['width'] - exact_width) / exact_width:+.1e}"
            if run["status"] == "ran"
            else run["note"][:80]
        )
    )

# ## Space × representation and time
#
# The relative amplitude error of every spatial scheme, for both field representations and for an
# explicit and an implicit time integrator.

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

space_key = "space x representation"
rows = list(spatial)
fig = plots.figure(1, 1, size=(14, 9))
plots.matrix(
    [
        [
            cell[(space_key, row, c)]["error"] if (space_key, row, c) in cell else np.nan
            for c in columns
        ]
        for row in rows
    ],
    rows,
    list(columns),
    scale="log",
    label="relative L2 amplitude error at z = 2 z_R",
    text=[
        [
            (
                f"{cell[(space_key, row, c)]['error']:.2e}\n{cell[(space_key, row, c)]['steps']} steps"
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
    title=f"Space x field representation x time, {cells} cells",
    ax=fig[0],
)
plots.show()

# All measures for the combinations that ran, sorted by the amplitude error.

matrix = [r for r in runs if r["matrix"] == space_key]
print(
    f"{'scheme':34s} {'representation, time':38s} {'L2 rel':>9s} {'width err':>10s} "
    f"{'steps':>6s} {'time':>8s}"
)
for r in sorted((r for r in matrix if r["status"] == "ran"), key=lambda r: r["error"]):
    print(
        f"{r['row']:34s} {r['column']:38s} {r['error']:9.2e} "
        f"{(r['width'] - exact_width) / exact_width:+10.1e} {r['steps']:6d} {r['seconds']:7.1f}s"
    )

# ## Profiles of representative combinations
#
# Modulus and real part of the envelope at $z=2z_R$ against the exact beam, and the input beam for
# comparison.

shown = [
    r
    for r in matrix
    if (r["row"], r["column"]) in representative and r["status"] == "ran"
]
fig = plots.figure(1, 2, size=(15, 5))
for panel, part, name in (
    (fig[0], abs, "|A| [V/m]"),
    (fig[1], np.real, "Re A [V/m]"),
):
    plots.line(
        y={
            f"{r['row']}, {r['column']}": (centres[cells] * 1e6, part(r["profile"]))
            for r in shown
        },
        reference={
            "exact beam at z = 2 z_R": (fine * 1e6, part(exact["fine"])),
            "input beam": (fine * 1e6, part(amplitude * np.exp(-((fine / waist) ** 2)) + 0j)),
        },
        xlabel="transverse position x [µm]",
        ylabel=name,
        title=f"{name.split(' ')[0]} at z = 2 z_R",
        legend="upper right" if part is abs else False,
        ax=panel,
    )
plots.show()

# ## Time integrator × space
#
# Explicit integrators at their stable step, implicit integrators at $\Delta t=t_{end}/100$.

time_key = "time integrator x space"
time_rows, time_columns = list(integrators), list(time_spatial)
fig = plots.figure(1, 2, size=(17, 12))
time_hatched = [[(time_key, row, c) in hatch for c in time_columns] for row in time_rows]
time_notes = (
    [
        "Hatched: the declared integrator fails at Δt = t_end/100 (explicit RK4 start-up on a stiff "
        "spectrum, or a bounded stability region); colour and value show its counterpart:"
    ]
    + [
        f"  {row}: BDF({row[3]}) with backward-Euler start-up."
        for row in gear_startup
        if any((time_key, row, c) in hatch for c in time_columns)
    ]
    + [
        f"  {row}: same method at Δt = t_end/{round(end_time / stable_step[row])}."
        for row in stable_step
        if any((time_key, row, c) in hatch for c in time_columns)
    ]
)
for panel, measure, label, title in (
    (fig[0], "error", "relative L2 amplitude error", "Amplitude error"),
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

# ## Nonlinear × linear solver
#
# Crank-Nicolson on the 4th-order finite differences, once with the native complex envelope and
# once with the real/imaginary pair. The problem is linear, so every converging solver reproduces
# the same discrete solution; the wall time differs.
#
# The implicit stage of the native complex field is solved on real coordinates (each complex
# value as a real/imaginary pair), so every solver pair of the real/imaginary representation also
# runs on the complex envelope: the automatic linear-solver selection picks the sparse direct
# solve of that real system, JFNK with a sparse direct inner solve probes the stage on the paired
# coordinates, and Aitken acceleration measures the complex fixed-point update in the same
# coordinates.

solver_rows = list(dict.fromkeys(row for row, _ in nonlinear_solvers))
solver_columns = [*linear_solvers, "none"]
fig = plots.figure(1, 2, size=(20, 7))
for panel, representation in zip((fig[0], fig[1]), envelopes):
    solver_key = f"nonlinear x linear solver, {representation}"
    plots.matrix(
        [
            [
                (
                    cell[(solver_key, row, c)]["seconds"]
                    if (solver_key, row, c) in cell
                    and cell[(solver_key, row, c)]["status"] == "ran"
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
            "Hatched: the declared solver fails at Δt = t_end/100; colour and value show the "
            "fixed-point iteration at Δt = t_end/3200, or the Krylov solver with an ILU "
            "preconditioner."
        ],
        title=f"Crank-Nicolson, {representation}: wall time and relative L2 error",
        ax=panel,
    )
plots.show()

# ## Mesh check: 400 against 200 cells
#
# The representative combinations on half as many cells. The amplitude error of a second-order
# scheme falls by about four, that of the 4th-order differences by about sixteen, as long as the
# time error stays below the spatial one.

coarse = {(r["row"], r["column"]): r for r in runs if r["matrix"] == "mesh check"}
grid = {(r["row"], r["column"]): r for r in matrix}
print(
    f"{'scheme':34s} {'representation, time':38s} {'L2 200':>9s} {'L2 400':>9s} {'ratio':>6s}"
)
for key, r in coarse.items():
    fine_run = grid[key]
    if r["status"] == "ran" and fine_run["status"] == "ran":
        print(
            f"{key[0]:34s} {key[1]:38s} {r['error']:9.2e} {fine_run['error']:9.2e} "
            f"{r['error'] / fine_run['error']:6.2f}"
        )

# ## What the library refused, and what failed
#
# A refused combination is caught by validation or binding before any time step. "No stable step"
# is the stability rule's verdict before the first step; a failed run started and amplified its
# high modes.

for status in ("rejected", "no stable step", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(
            f"  {r['matrix'][:12]:12s} {r['row'][:30]:30s} {r['column'][:36]:36s} "
            f"{r.get('variant') or '':10s} {r['note'][:90]}"
        )
print(
    "\nsummary: "
    + ", ".join(
        f"{sum(r['status'] == s for r in runs)} {s}"
        for s in ("ran", "no stable step", "rejected", "failed")
    )
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.0f} min in total"
)

# ## Methods that do not apply to paraxial diffraction
#
# Forward Euler, Heun, SSP-RK2, the explicit midpoint rule, the embedded Heun-Euler pair and
# Adams-Bashforth 2 have stability regions that touch the imaginary axis only at the origin: on a
# purely dispersive operator every mode is amplified at every step size, so no stable step exists
# and the stability rule refuses them before the first step ("no stable step" in the matrix). No
# single added ingredient changes that without changing the method or the physics (a damping term).
#
# Other methods in the library are not in the matrix because the paraxial equation lacks what they
# are built for: Riemann solvers, reconstructions and limiters are for convective fluxes, and the
# envelope has none; Runge-Kutta-Chebyshev and -Legendre super-steps are built for the negative real
# axis of parabolic operators; conjugate gradients and algebraic multigrid need a symmetric positive
# definite operator, while the complex paraxial stage matrix is symmetric but not Hermitian.
#
# ## Going further
#
# Add an absorbing medium (`absorption=`) or a refractive-index perturbation
# (`refractive_index_perturbation=`) to the two envelope declarations: the exact beam is multiplied
# by $e^{-\alpha z}$ or $e^{ik\Delta n z}$, and the matrix code stays as it is.
