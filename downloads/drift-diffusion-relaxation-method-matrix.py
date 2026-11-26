"""OneSimulation example: diffusive relaxation in a drift-diffusion plasma through every method combination.
Requires the installed, activated OneSimulation runtime.

One declaration of a drift-diffusion plasma (electrons, argon ions, mean electron energy) between two
insulating walls is solved with every applicable combination of finite-volume, finite-difference and
finite-element discretization, time integrator, nonlinear solver and linear solver. Each run is graded
against the exact Fourier-mode solution.
"""

# # Diffusive relaxation in a drift-diffusion plasma: one model, every method
#
# Electrons and $\mathrm{Ar}^+$ ions fill a gap of width $L=1$ cm between two insulating walls
# (no particle and no energy flux). Their densities obey the drift-diffusion balances
#
# $$\partial_t n_s+\partial_x\Gamma_s=0,\qquad \Gamma_s=\pm\mu_s n_s E-D_s\,\partial_x n_s,$$
#
# and the electron energy density $w_e=n_e\bar\varepsilon$ obeys its own balance with the energy flux
# $\Gamma_w=-\tfrac{5}{3}\,\cdot\,$(enthalpy factor)$\,\cdot\,(\mu_e E w_e+\partial_x(D_e w_e))$.
# The electrons relax momentum at a constant collision frequency $\nu=10^{12}\,\mathrm{s^{-1}}$
# (so $D_e=2\bar\varepsilon/(3m_e\nu)$), the ions have the mobility $0.2\,\mathrm{m^2/(V\,s)}$ in
# argon at 300 K (so $D_i=\mu_i k_BT/e$), and the electric field is prescribed as zero.
#
# **Exact solution.** With a uniform mean energy $\bar\varepsilon=1$ eV and the enthalpy factor 1 the
# energy flux is $\bar\varepsilon\,\Gamma_e$, so $\bar\varepsilon$ stays uniform and both
# diffusivities stay constant. Each density then relaxes mode by mode:
#
# $$n_s(x,t)=n_0\Big[1+\tfrac12\cos\tfrac{\pi x}{L}\,e^{-D_s\pi^2t/L^2}
# +\tfrac15\cos\tfrac{3\pi x}{L}\,e^{-9D_s\pi^2t/L^2}\Big].$$
#
# The electrons diffuse $D_e/D_i\approx23$ times faster than the ions, a stiff pair of time scales.
# This is a free-diffusion test of the transport, the walls and the energy coupling; it has no
# space charge (the field is prescribed), so it is not ambipolar diffusion.
#
# **Grading.** At $t_\mathrm{end}=0.2\,L^2/(\pi^2D_e)$ the electron and ion densities are evaluated
# at the 100 cell centres and compared with the exact solution (cell averages for finite volumes,
# point values otherwise): $L_1=\frac{1}{N}\sum_i|n_h-n|/n_0$.

import time

import numpy as np
from onesim.cad.d1 import Line
from onesim.materials import NeutralGas
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import (
    CentralDifference,
    CentralLaplacian,
    FiniteDifference,
    PolynomialDifference,
    PolynomialLaplacian,
    SummationByPartsDifference,
)
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.fvm import (
    BarthJespersenLimiter,
    CellGradientReconstruction,
    FiniteVolume,
    ImplicitLocalSources,
    LeastSquaresGradient,
    ScharfetterGummel,
    VenkatakrishnanLimiter,
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
)
from onesim.physics.chemistry import Chemistry, ReactionNetworkIR, SpeciesIR
from onesim.physics.plasma import (
    DriftDiffusionPlasma,
    ElectronCollisionModel,
    IonMobilityModel,
    MagnetizedElectronTransport,
    Maxwellian,
    MeanElectronEnergyEquation,
    NoParticleFlux,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.constants import electron_mass, elementary_charge

theme.set()

# ## The physics, declared once
#
# Nothing in this cell names a numerical method.

width, density, cells, coarse_cells = 0.01, 1e15, 100, 50
gap = Line(start=(0.0, 0.0, 0.0), end=(width, 0.0, 0.0))
argon_plasma = ReactionNetworkIR(
    "argon drift-diffusion relaxation",
    "1",
    (
        SpeciesIR("electron", -1, electron_mass, role="electron"),
        SpeciesIR("Ar+", 1, 6.63e-26, transport={"mobility": {"value": 0.2, "unit": "m^2/V/s"}}),
        SpeciesIR("Ar", 0, 6.63e-26),
    ),
    (),
    metadata={"electron_momentum_transfer": {"kind": "constant_frequency", "value": 1e12, "unit": "1/s"}},
)
chemistry = Chemistry(network=argon_plasma)
gas = NeutralGas(species="Ar", pressure=1000.0, temperature=300.0)
plasma = DriftDiffusionPlasma(
    domain=gap.line,
    gas=gas,
    chemistry=chemistry,
    electron_energy=MeanElectronEnergyEquation(enthalpy_factor=1.0),
    electron_transport=MagnetizedElectronTransport(
        collisions=ElectronCollisionModel(gas=gas, chemistry=chemistry, energy_distribution=Maxwellian())
    ),
    ion_transport=IonMobilityModel(gas=gas, chemistry=chemistry),
    prescribed_inputs={"electrostatic_field": (0.0, 0.0, 0.0)},
)
relaxation = Problem(
    name="Drift-diffusion relaxation between insulating walls",
    geometry=gap,
    physics=(plasma,),
    boundary_conditions=(NoParticleFlux(boundary=gap.boundaries.all, physics=plasma),),
    initial_conditions=(
        InitialValue(plasma.electron_density, lambda x: density * (1 + 0.5 * np.cos(np.pi * x / width) + 0.2 * np.cos(3 * np.pi * x / width))),
        InitialValue(plasma.density("Ar+"), lambda x: density * (1 + 0.5 * np.cos(np.pi * x / width) + 0.2 * np.cos(3 * np.pi * x / width))),
        InitialValue(plasma.mean_electron_energy, 1.0, unit="eV"),
    ),
)
meshes = {n: StructuredLineMesher(cells=n).generate(gap) for n in (cells, coarse_cells)}
centres = {n: mesh.cell_centers[:, 0] for n, mesh in meshes.items()}
print(f"{meshes[cells].cell_count} cells of width {width / cells * 1e3:.3f} mm (coarse check: {coarse_cells} cells)")

# ## The exact solution
#
# The diffusivities come from the declared transport laws themselves; the solution is the sum of the
# two decaying modes (their cell averages carry the factor $\mathrm{sinc}$ of half a cell).

probe = np.array([[width / 2, 0.0, 0.0]])
diffusivity = {
    "electron": plasma.electron_transport.coefficients(
        mean_electron_energy=np.array([elementary_charge]), magnetic_field=np.zeros((1, 3)), points=probe
    ).diffusivity[0, 0, 0],
    "Ar+": plasma.ion_transport.coefficients(
        species=next(s for s in chemistry.species if s.species_id == "Ar+"), points=probe
    ).diffusivity[0, 0, 0],
}
end_time = 0.2 * width**2 / (np.pi**2 * diffusivity["electron"])
exact = {
    (n, species, average): density
    * (
        1
        + 0.5 * np.cos(np.pi * centres[n] / width) * np.exp(-d * (np.pi / width) ** 2 * end_time)
        * (np.sinc(0.5 / n) if average else 1.0)
        + 0.2 * np.cos(3 * np.pi * centres[n] / width) * np.exp(-d * (3 * np.pi / width) ** 2 * end_time)
        * (np.sinc(1.5 / n) if average else 1.0)
    )
    for n in (cells, coarse_cells)
    for species, d in diffusivity.items()
    for average in (True, False)
}
fine = np.linspace(0.0, width, 401)
print(
    f"D_e = {diffusivity['electron']:.4f} m^2/s, D_i = {diffusivity['Ar+']:.5f} m^2/s, "
    f"ratio {diffusivity['electron'] / diffusivity['Ar+']:.1f}; t_end = {end_time:.3e} s"
)

# ## The method catalogue
#
# Finite volumes with Scharfetter-Gummel fluxes for particles and energy (exact for a constant drift
# and diffusivity across each face), with the two-point or the least-squares gradient, with linear
# cell reconstruction, alone or limited by Barth-Jespersen or Venkatakrishnan. Finite differences:
# central, fourth- and sixth-order polynomial and summation-by-parts stencils. Continuous finite
# elements of degree 1 to 3. Every public time integrator, and every nonlinear/linear solver pair
# for implicit steps.

volume = dict(particle_flux=ScharfetterGummel(), energy_flux=ScharfetterGummel(), reaction_sources=ImplicitLocalSources())
spatial = {
    "FVM Scharfetter-Gummel": FiniteVolume(**volume),
    "FVM S-G, least-squares gradient": FiniteVolume(gradient=LeastSquaresGradient(), **volume),
    "FVM S-G, linear reconstruction": FiniteVolume(
        gradient=LeastSquaresGradient(), reconstruction=CellGradientReconstruction(), **volume
    ),
    "FVM S-G, linear + Barth-Jespersen": FiniteVolume(
        gradient=LeastSquaresGradient(), reconstruction=CellGradientReconstruction(),
        limiter=BarthJespersenLimiter(), **volume
    ),
    "FVM S-G, linear + Venkatakrishnan": FiniteVolume(
        gradient=LeastSquaresGradient(), reconstruction=CellGradientReconstruction(),
        limiter=VenkatakrishnanLimiter(), **volume
    ),
    "FDM central, 2nd order": FiniteDifference(
        gradient=CentralDifference(), diffusion=CentralLaplacian(), reaction_sources=ImplicitLocalSources()
    ),
    **{
        f"FDM polynomial, order {order}": FiniteDifference(
            gradient=PolynomialDifference(accuracy_order=order),
            diffusion=PolynomialLaplacian(accuracy_order=order),
            reaction_sources=ImplicitLocalSources(),
        )
        for order in (4, 6)
    },
    "FDM summation by parts, order 4": FiniteDifference(
        gradient=SummationByPartsDifference(accuracy_order=4),
        diffusion=PolynomialLaplacian(accuracy_order=4),
        reaction_sources=ImplicitLocalSources(),
    ),
    **{
        f"FEM CG P{degree}": FiniteElement(
            element=Lagrange(degree), quadrature_order=degree + 2, reaction_sources=ImplicitLocalSources()
        )
        for degree in (1, 2, 3)
    },
}
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
    **{f"BDF{order} (Gear)": TimeMethod(f"bdf{order}_gear") for order in (2, 3, 4, 5, 6)},
    "Rosenbrock-Euler": TimeMethod("rosenbrock_euler"),
    "Rosenbrock-Wanner Euler": TimeMethod("rosenbrock_wanner_euler"),
    "adaptive BDF": AdaptiveBDF(relative_tolerance=1e-6, absolute_tolerance=1e-3),
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
        (name, linear_name): kind(linear_solver=linear)
        for name, kind in (("Newton", Newton), ("damped Newton", DampedNewton),
                           ("trust-region Newton", TrustRegionNewton), ("quasi-Newton", QuasiNewton))
        for linear_name, linear in linear_solvers.items()
    },
    **{("JFNK", linear_name): JFNK(linear_solver=linear) for linear_name, linear in linear_solvers.items()},
    ("fixed point", "none"): FixedPoint(maximum_iterations=200),
}

# ## Time steps
#
# The plasma path takes the step it is given; explicit stability is the user's choice of step. The
# explicit integrators run at $\Delta t=c\,\Delta x^2/D_e$ with $c=0.2$ for volumes and differences
# and $c=0.03$ for linear elements (the consistent mass matrix has a three times larger spectral
# radius, and the filtered leapfrog needs the margin); every explicit method in the catalogue is
# stable there except leapfrog. The implicit
# integrators run at $\Delta t=t_\mathrm{end}/40$, about 25 times the explicit limit.
#
# Leapfrog has no stable region on the negative real axis, so it is unstable for diffusion at every
# step; its stabilized counterpart is the Robert-Asselin-Williams filtered leapfrog. BDF4 to BDF6 and
# Adams-Moulton 3 and 4 are not A-stable, and a fixed-point (Picard) iteration contracts only below
# the inverse spectral radius of the stage; where they fail, the same method runs at a stable step.

dx = width / cells
explicit_factor = {"FVM": 0.2, "FDM": 0.2, "FEM": 0.03}
implicit_step, space_step = end_time / 40, end_time / 160
stable_step = 0.25 * dx**2 / diffusivity["electron"]
stabilized_time = {"leapfrog": ("leapfrog + Robert-Asselin-Williams filter", TimeMethod("leapfrog_raw"))}

# ## The test matrix
#
# 1. **Space** — every spatial scheme with BDF2 (variable step), Newton and a sparse LU at
#    $\Delta t=t_\mathrm{end}/160$, small enough that the time error stays below the spatial error of
#    the fourth-order schemes.
# 2. **Time integrator × space** — every integrator on the Scharfetter-Gummel volumes, the central
#    differences and the linear elements.
# 3. **Nonlinear × linear solver** — backward Euler on the Scharfetter-Gummel volumes.
# 4. **Mesh check** — the space matrix on 50 cells.

newton = Newton(linear_solver=SparseDirect())
time_space = {name: spatial[name] for name in ("FVM Scharfetter-Gummel", "FDM central, 2nd order", "FEM CG P1")}
runs = [
    dict(matrix="space", row=name, column="BDF2", cells=cells, spatial=scheme, time=BDF(order=2),
         nonlinear=newton, step=space_step)
    for name, scheme in spatial.items()
]
runs += [
    dict(
        matrix="time x space", row=name, column=space, cells=cells, spatial=scheme, time=integrator,
        nonlinear=newton if getattr(integrator, "implicit", False) and "rosenbrock" not in integrator.method_id else None,
        linear=SparseDirect() if "rosenbrock" in integrator.method_id else None,
        step=implicit_step if getattr(integrator, "implicit", False) else explicit_factor[space[:3]] * dx**2 / diffusivity["electron"],
    )
    for name, integrator in integrators.items()
    for space, scheme in time_space.items()
]
runs += [
    dict(matrix="nonlinear x linear solver", row=row, column=column, cells=cells,
         spatial=spatial["FVM Scharfetter-Gummel"], time=BackwardEuler(), nonlinear=solver, step=implicit_step)
    for (row, column), solver in nonlinear_solvers.items()
]
runs += [
    {**run, "variant": "stabilized", "time": stabilized_time[run["row"]][1],
     "note_stabilized": stabilized_time[run["row"]][0]}
    for run in list(runs)
    if run["row"] in stabilized_time
]
runs += [
    {**run, "variant": "stabilized", "step": stable_step,
     "note_stabilized": f"same method at dt = {stable_step:.2e} s"}
    for run in list(runs)
    if run.get("variant") is None and run["step"] == implicit_step and run["matrix"] != "space"
]
runs += [{**run, "matrix": "mesh check", "cells": coarse_cells} for run in list(runs) if run["matrix"] == "space"]
print(
    f"{len(runs)} runs: "
    + ", ".join(f"{sum(r['matrix'] == m for r in runs)} {m}" for m in dict.fromkeys(r["matrix"] for r in runs))
)

# ## Run everything
#
# A combination the library refuses is recorded with its validation message; a run that stops because
# the mean energy left its positive domain, a step was rejected or a stage did not converge is
# recorded as failed. A stabilized variant runs only where its declared run failed.

declared = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") is None}
for run in runs:
    started = time.perf_counter()
    run.update(status="rejected", note="", errors=(np.nan, np.nan), steps=0)
    if run.get("variant") and declared[(run["matrix"], run["row"], run["column"])]["status"] == "ran":
        run.update(status="not needed", seconds=0.0)
        continue
    simulation = Simulation(
        problem=relaxation,
        mesh=meshes[run["cells"]],
        numerics=Numerics(
            spatial=run["spatial"],
            time_integrator=run["time"],
            nonlinear_solver=run["nonlinear"],
            linear_solver=SparseDirect() if isinstance(run["spatial"], FiniteElement) else run.get("linear"),
        ),
    )
    check = simulation.validate()
    if not check.valid:
        run["note"] = check.diagnostics[0].split(": ", 1)[-1].splitlines()[0]
    else:
        try:
            result = simulation.run(start_time=0.0, end_time=end_time, time_step=run["step"])
            points = np.column_stack((centres[run["cells"]], np.zeros((run["cells"], 2))))
            run["profiles"] = [
                result.field(field).evaluate(points).reshape(-1)
                for field in (plasma.electron_density, plasma.density("Ar+"))
            ]
            run["errors"] = tuple(
                np.mean(np.abs(profile - exact[(run["cells"], species, isinstance(run["spatial"], FiniteVolume))])) / density
                for profile, species in zip(run["profiles"], ("electron", "Ar+"))
            )
            run["steps"] = len(result.times) - 1
            run["status"] = "ran" if np.all(np.isfinite(run["errors"])) else "failed"
        except Exception as error:
            message = str(error).splitlines()[0]
            run["status"] = (
                "failed"
                if any(word in message for word in ("positive", "rejected", "converge", "finite", "smaller step"))
                else "rejected"
            )
            run["note"] = message.split(": ", 1)[-1] if ": " in message else message
    run["seconds"] = time.perf_counter() - started
    print(
        f"{run['matrix'][:14]:14s} {run['row'][:34]:34s} {run['column'][:22]:22s} {run.get('variant') or '':10s} "
        f"{run['status']:8s} {run['seconds']:7.1f} s  "
        + (f"L1 n_e {run['errors'][0]:.2e}  n_i {run['errors'][1]:.2e}" if run["status"] == "ran" else run["note"][:80])
    )

# ## Space
#
# The electron and ion density errors of every spatial scheme. All runs use the same step
# $t_\mathrm{end}/160$; the BDF2 time error at this step is about $3\cdot10^{-7}$, a common floor
# below which the high-order schemes cannot be told apart.
# With the field prescribed as zero the face reconstruction and its limiter act on a vanishing
# drift, so the five finite-volume variants give the same Scharfetter-Gummel diffusion flux and the
# same error; they differ once a field drives the particles.

main = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") is None}
substitute = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") == "stabilized"}
hatch = {key for key, r in main.items() if r["status"] != "ran" and substitute.get(key, {}).get("status") == "ran"}
cell = {key: substitute[key] if key in hatch else r for key, r in main.items()}
rows = list(spatial)
fig = plots.figure(1, 1, size=(10, 8))
plots.matrix(
    [[cell[("space", row, "BDF2")]["errors"][i] for i in (0, 1)] for row in rows],
    rows,
    ["L1 electron density", "L1 ion density"],
    scale="log",
    label="relative L1 error",
    text=[
        [
            f"{cell[('space', row, 'BDF2')]['errors'][i]:.2e}" if cell[("space", row, "BDF2")]["status"] == "ran"
            else cell[("space", row, "BDF2")]["status"]
            for i in (0, 1)
        ]
        for row in rows
    ],
    title=f"Space, {cells} cells, BDF2 at dt = t_end/160",
    ax=fig[0],
)
plots.show()

# ## Profiles
#
# Electron and ion densities at $t_\mathrm{end}$ for representative schemes against the exact solution.

shown = [main[("space", name, "BDF2")] for name in ("FVM Scharfetter-Gummel", "FDM polynomial, order 4", "FEM CG P2")
         if main[("space", name, "BDF2")]["status"] == "ran"]
fig = plots.figure(1, 2, size=(15, 5))
for panel, index, species in ((fig[0], 0, "electron"), (fig[1], 1, "Ar+")):
    plots.line(
        y={r["row"]: (centres[cells] * 1e3, r["profiles"][index] / density) for r in shown},
        reference={
            "exact": (
                fine * 1e3,
                1 + 0.5 * np.cos(np.pi * fine / width) * np.exp(-diffusivity[species] * (np.pi / width) ** 2 * end_time)
                + 0.2 * np.cos(3 * np.pi * fine / width) * np.exp(-diffusivity[species] * (3 * np.pi / width) ** 2 * end_time),
            )
        },
        xlabel="position x [mm]",
        ylabel="n / n0",
        title=f"{species} density at t_end",
        ax=panel,
    )
plots.show()

# ## Time integrator × space
#
# Explicit integrators at their stable step, implicit integrators at $t_\mathrm{end}/40$.

time_rows, time_columns = list(integrators), list(time_space)
fig = plots.figure(1, 2, size=(17, 12))
for panel, measure, label in ((fig[0], "errors", "L1 electron density error"), (fig[1], "seconds", "wall time [s]")):
    plots.matrix(
        [
            [
                (cell[("time x space", row, c)]["errors"][0] if measure == "errors" else cell[("time x space", row, c)]["seconds"])
                if cell[("time x space", row, c)]["status"] == "ran" else np.nan
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
                    f"{cell[('time x space', row, c)]['errors'][0]:.2e}" if measure == "errors"
                    else f"{cell[('time x space', row, c)]['seconds']:.0f} s, {cell[('time x space', row, c)]['steps']} steps"
                )
                if cell[("time x space", row, c)]["status"] == "ran" else cell[("time x space", row, c)]["status"]
                for c in time_columns
            ]
            for row in time_rows
        ],
        hatched=[[("time x space", row, c) in hatch for c in time_columns] for row in time_rows],
        notes=(
            ["Hatched: the declared integrator is unstable at this step; colour and value show:"]
            + list(dict.fromkeys(
                f"  {row}: {substitute[('time x space', row, c)]['note_stabilized']}."
                for row in time_rows for c in time_columns if ("time x space", row, c) in hatch
            ))
            if measure == "errors" and any(("time x space", row, c) in hatch for row in time_rows for c in time_columns)
            else None
        ),
        title=label,
        ax=panel,
    )
plots.show()

# ## Nonlinear × linear solver
#
# Backward Euler on the Scharfetter-Gummel volumes. The discrete solution does not depend on the
# converged solver; the wall time does.

solver_rows = list(dict.fromkeys(row for row, _ in nonlinear_solvers))
solver_columns = [*linear_solvers, "none"]
key = "nonlinear x linear solver"
fig = plots.figure(1, 1, size=(13, 6))
plots.matrix(
    [
        [cell[(key, row, c)]["seconds"] if (key, row, c) in cell and cell[(key, row, c)]["status"] == "ran" else np.nan
         for c in solver_columns]
        for row in solver_rows
    ],
    solver_rows,
    solver_columns,
    scale="log",
    label="wall time [s]",
    text=[
        [
            (f"{cell[(key, row, c)]['seconds']:.0f} s\n{cell[(key, row, c)]['errors'][0]:.3e}"
             if cell[(key, row, c)]["status"] == "ran" else cell[(key, row, c)]["status"])
            if (key, row, c) in cell else ""
            for c in solver_columns
        ]
        for row in solver_rows
    ],
    hatched=[[(key, row, c) in hatch for c in solver_columns] for row in solver_rows],
    notes=(
        ["Hatched: the fixed-point iteration cannot contract at dt = t_end/40; colour and value show it at "
         f"dt = {stable_step:.2e} s."]
        if any((key, row, c) in hatch for row in solver_rows for c in solver_columns)
        else None
    ),
    title="Backward Euler: wall time and L1 electron density error per solver pair",
    ax=fig[0],
)
plots.show()

# ## Mesh check: 100 against 50 cells

print(f"{'scheme':36s} {'L1 n_e 50':>11s} {'L1 n_e 100':>11s} {'ratio':>6s}")
for r in (r for r in runs if r["matrix"] == "mesh check"):
    fine_run = main[("space", r["row"], r["column"])]
    if r["status"] == "ran" and fine_run["status"] == "ran":
        print(f"{r['row']:36s} {r['errors'][0]:11.3e} {fine_run['errors'][0]:11.3e} {r['errors'][0] / fine_run['errors'][0]:6.2f}")

# ## What the library refused, and what failed

for status in ("rejected", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(f"  {r['matrix'][:14]:14s} {r['row'][:34]:34s} {r['column'][:22]:22s} {r.get('variant') or '':10s} {r['note'][:90]}")
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.0f} min in total"
)

# ## Methods that do not apply
#
# Discontinuous elements are not offered for the plasma balances: the plasma finite-element
# formulation is a continuous volume formulation and refuses them with that reason. Riemann solvers,
# Lax-Friedrichs splitting and shock-capturing limiters target hyperbolic systems; the drift-diffusion
# flux is parabolic and its upwinding is built into the Scharfetter-Gummel flux. IMEX splittings and
# Runge-Kutta-Chebyshev super-steps are not public time declarations of this path.
