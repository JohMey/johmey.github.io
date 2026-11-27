"""OneSimulation example: linear Landau damping and the two-stream instability through every kinetic method.
Requires the installed, activated OneSimulation runtime.

One collisionless electron plasma is solved twice: as a distribution function on a phase-space grid
(Vlasov-Poisson) and as macro-particles (particle-in-cell). Every phase-space reconstruction and face
flux, every particle shape function, every public time integrator and the matrix-free stage solvers
are combined; each run is graded against the root of the kinetic dispersion relation.
"""

# # Landau damping: one plasma, every kinetic method
#
# Electrons with a Maxwellian velocity distribution $f_0(v)=e^{-v^2/2}/\sqrt{2\pi}$ move through a
# neutralizing ion background in a periodic box of length $L=2\pi/k$, $k=0.5$ (units of the Debye
# length, the plasma frequency and the thermal speed). A small density perturbation
#
# $$f(x,v,0)=\big(1+\alpha\cos kx\big)\,f_0(v)$$
#
# launches a Langmuir wave. The Vlasov-Poisson system
#
# $$\partial_t f+v\,\partial_x f-E\,\partial_v f=0,\qquad \partial_x E=1-\int f\,dv$$
#
# is declared once. Everything numerical is an argument: the phase-space reconstruction and face
# flux, the particle shape function and the time integrator with its stage solvers.
#
# **Reference.** For small $\alpha$ the field oscillates at the complex frequency $\omega=\omega_r-i\gamma$
# that solves $1+\big(1+\zeta Z(\zeta)\big)/k^2=0$, $\zeta=\omega/(\sqrt2 k)$, with the plasma
# dispersion function $Z(\zeta)=i\sqrt\pi\,w(\zeta)$ (Faddeeva function $w$; Fried and Conte, 1961).
# For $k=0.5$: $\omega_r\approx1.4156$, $\gamma\approx0.1533$.
#
# **Grading.** The electric field energy $W_E(t)$ oscillates at $2\omega_r$ inside the envelope
# $e^{-2\gamma t}$. Its maxima for $1<t<14$ give the measured damping rate (slope of $\ln W_E$) and
# frequency (spacing of the maxima); the error is $|\gamma_h-\gamma|/\gamma$.

import time
from functools import partial

import numpy as np
from onesim.numerics.fvm import (
    MinmodLimiter,
    MonotonizedCentralLimiter,
    SuperbeeLimiter,
    VanLeerLimiter,
)
from onesim.numerics.fvm.spatial.fluxes.directional_transport import (
    constant_face_traces,
    minmod_face_traces,
    nonnegative_face_traces,
    scalar_upwind_flux,
    upwind_phase_flux,
)
from onesim.numerics.discretization.spatial.fields.particles.periodic_particle_transfer import (
    CardinalBSplineWeights,
    cloud_in_cell_weights,
)
from onesim.numerics.discretization.spatial.equations.periodic_spectral_fields import (
    PeriodicSpectralFields,
)
from onesim.numerics.linear import GMRES, FlexibleGMRES, RightGMRES
from onesim.numerics.time.selection.declared_time_provider import declared_time_descriptor
from onesim.physics.plasma.kinetic.numerics.backends.periodic_vlasov import CourantStepLimit
from onesim.physics.plasma.particles.numerics.particle_pushers import boris_velocity
from onesim.numerics.nonlinear import JFNK, FixedPoint
from onesim.numerics.time import (
    BDF,
    AdaptiveBDF,
    AdaptiveRungeKutta45,
    BackwardEuler,
    ClassicalRungeKutta4,
    ForwardEuler,
    TimeMethod,
)
from onesim.physics.plasma import (
    DistributionFunctionState,
    LimitedPhaseTraces,
    PeriodicParticleInCell,
    PeriodicVlasov,
    PICSpecies,
    VlasovDiscretization,
    VlasovPoissonEquationContract,
    VlasovPoissonRequest,
    central_phase_flux,
)
from onesim.physics.plasma.fluid.models.declarations.species.species import (
    SpeciesDefinition,
    SpeciesSet,
)
from scipy.optimize import brentq, root
from scipy.signal import find_peaks
from scipy.special import wofz
from onesim.meshing import TensorCollocationMesh
from onesim.meshing.meshes import CartesianMesh
from onesim.visualization import plots, theme

theme.set()

# ## The physics, declared once
#
# The phase-space grid has $64\times128$ cells on $[0,L)\times[-6,6]$; the particle model uses 64
# cells and $1024\times128$ macro-particles placed on a regular phase-space lattice and weighted with
# $f(x,v,0)$ (a quiet start, so the particle noise stays far below the wave). `scale` reduces all
# counts for a quick check.

scale = 1.0
k, alpha, end_time = 0.5, 0.01, 15.0
length = 2 * np.pi / k
nx, nv = int(64 * scale), int(128 * scale)
x = length * (np.arange(nx) + 0.5) / nx
v = -6 + (np.arange(nv) + 0.5) * 12 / nv
maxwellian = np.exp(-(v**2) / 2)
maxwellian /= maxwellian.sum() * (12 / nv)  # discrete normalization: exact neutrality
electrons = SpeciesSet((SpeciesDefinition("e", "electron", charge_number=-1),))
landau = VlasovPoissonRequest(
    DistributionFunctionState(
        electrons, {"e": (1 + alpha * np.cos(k * x[:, None])) * maxwellian}, (nx,), (v,)
    ),
    VlasovPoissonEquationContract(1, 1),
)
coarse_landau = VlasovPoissonRequest(
    DistributionFunctionState(
        electrons,
        {
            "e": (1 + alpha * np.cos(k * (length * (np.arange(nx // 2) + 0.5) / (nx // 2)))[:, None])
            * np.exp(-((-6 + (np.arange(nv // 2) + 0.5) * 24 / nv) ** 2) / 2)
            / (np.exp(-((-6 + (np.arange(nv // 2) + 0.5) * 24 / nv) ** 2) / 2).sum() * 24 / nv)
        },
        (nx // 2,),
        (-6 + (np.arange(nv // 2) + 0.5) * 24 / nv,),
    ),
    VlasovPoissonEquationContract(1, 1),
)
vlasov, coarse_vlasov = (
    PeriodicVlasov(
        mesh=TensorCollocationMesh(
            axes=(length * (np.arange(n) + 0.5) / n,), coordinate_system="cartesian", periodic=(True,)
        ),
        background_charge_density=1.0,
    )
    for n in (nx, nx // 2)
)
cells, lattice_x, lattice_v, pic_alpha = int(64 * scale), int(1024 * scale), int(128 * scale), 0.05
x0 = length * (np.arange(lattice_x) + 0.5) / lattice_x
v0 = -6 + (np.arange(lattice_v) + 0.5) * 12 / lattice_v
particle_velocities = np.zeros((lattice_x * lattice_v, 3))
particle_velocities[:, 0] = np.tile(v0, lattice_x)
particles = PICSpecies(
    positions=np.repeat(x0, lattice_v)[:, None],
    velocities=particle_velocities,
    charge=-1.0,
    mass=1.0,
    weights=np.outer(1 + pic_alpha * np.cos(k * x0), np.exp(-(v0**2) / 2)).ravel()
    * length
    / np.outer(1 + pic_alpha * np.cos(k * x0), np.exp(-(v0**2) / 2)).sum(),
)
print(
    f"Vlasov: {nx} x {nv} = {nx * nv} phase-space cells; PIC: {cells} cells, "
    f"{particles.positions.shape[0]} macro-particles"
)

# ## The exact damping rate and the two-stream growth rate
#
# The Landau root of $1+(1+\zeta Z(\zeta))/k^2=0$, and the growth rate of two warm counter-streaming
# beams ($\pm u$, thermal spread $\sigma$), the real root of
# $1+\tfrac12\sum_\pm\big(1+\zeta_\pm Z(\zeta_\pm)\big)/(k\sigma)^2=0$ with
# $\zeta_\pm=(i\Gamma/k\mp u)/(\sqrt2\sigma)$.

landau_root = root(
    lambda w: [
        part(
            1
            + (
                1
                + (w[0] - 1j * w[1]) / (np.sqrt(2) * k)
                * 1j * np.sqrt(np.pi) * wofz((w[0] - 1j * w[1]) / (np.sqrt(2) * k))
            )
            / k**2
        )
        for part in (np.real, np.imag)
    ],
    [1.4, 0.15],
    tol=1e-14,
).x
omega_exact, gamma_exact = landau_root
beam, spread = 1.0, 0.14
growth_exact = brentq(
    lambda g: 1
    + np.mean(
        (
            1
            + (1j * g / k - np.array([-beam, beam])) / (np.sqrt(2) * spread)
            * 1j * np.sqrt(np.pi)
            * wofz((1j * g / k - np.array([-beam, beam])) / (np.sqrt(2) * spread))
        ).real
    )
    / (k * spread) ** 2,
    1e-6,
    1.0,
)
print(
    f"Landau: omega_r = {omega_exact:.5f}, gamma = {gamma_exact:.5f}; "
    f"two-stream growth rate = {growth_exact:.5f}"
)

# ## The method catalogue
#
# **Phase space.** The transport in $x$ and $v$ is a conservative finite-volume flux. The face
# traces are constant (first order) or MUSCL-linear with no limiter (Fromm), with one of four slope
# limiters, or minmod plus a positivity-preserving trace limiter; the face flux is upwind or the
# non-dissipative central average. The field comes from a spectral Poisson solve.
#
# **Particles.** Deposition and interpolation share one shape function: nearest grid point (B-spline
# of degree 0), cloud-in-cell (degree 1), triangular-shaped cloud (degree 2) and the cubic B-spline.
#
# **Time.** Every public integrator. Implicit stages are solved matrix-free: Jacobian-free
# Newton-Krylov or a fixed-point iteration, passed explicitly; Rosenbrock stages by GMRES with
# finite-difference Jacobian products.

reconstructions = {
    "constant (first order)": constant_face_traces,
    "MUSCL unlimited (Fromm)": LimitedPhaseTraces(None),
    "MUSCL + minmod": LimitedPhaseTraces(MinmodLimiter()),
    "MUSCL + van Leer": LimitedPhaseTraces(VanLeerLimiter()),
    "MUSCL + MC": LimitedPhaseTraces(MonotonizedCentralLimiter()),
    "MUSCL + superbee": LimitedPhaseTraces(SuperbeeLimiter()),
    "MUSCL + minmod + positivity": "positivity",
}
face_fluxes = {"upwind": scalar_upwind_flux, "central": central_phase_flux}
phase_fluxes = {
    (row, column): partial(
        upwind_phase_flux,
        reconstruction=minmod_face_traces if traces == "positivity" else traces,
        numerical_flux=flux,
        trace_limiter=nonnegative_face_traces if traces == "positivity" else None,
    )
    for row, traces in reconstructions.items()
    for column, flux in face_fluxes.items()
}
phase_space = {
    key: VlasovDiscretization(
        field_factory=PeriodicSpectralFields,
        phase_flux=flux,
        velocity_flux=flux,
        transport_step_limit=CourantStepLimit(0.45),
    )
    for key, flux in phase_fluxes.items()
}
# A linear reconstruction without limiter, and the central flux on any traces, are not monotone:
# by Godunov's theorem they create undershoots where f is small (the velocity tails), and the
# library stops when f turns negative instead of clipping it. Their stabilized counterpart adds the
# positivity-preserving trace limiter to the same reconstruction and flux.
stabilized_space = {
    key: (
        f"{key[0]}, {key[1]} flux + positivity-preserving trace limiter",
        VlasovDiscretization(
            field_factory=PeriodicSpectralFields,
            phase_flux=(
                stabilized_flux := partial(
                    upwind_phase_flux,
                    reconstruction=minmod_face_traces if traces == "positivity" else traces,
                    numerical_flux=face_fluxes[key[1]],
                    trace_limiter=nonnegative_face_traces,
                )
            ),
            velocity_flux=stabilized_flux,
            transport_step_limit=CourantStepLimit(0.45),
        ),
    )
    for key, traces in (
        (key, reconstructions[key[0]]) for key in phase_space
    )
    if traces is not constant_face_traces or key[1] == "central"
}
shapes = {
    "nearest grid point": CardinalBSplineWeights(0),
    "cloud in cell": cloud_in_cell_weights,
    "triangular cloud (TSC)": CardinalBSplineWeights(2),
    "cubic B-spline": CardinalBSplineWeights(3),
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
    "adaptive BDF": AdaptiveBDF(relative_tolerance=1e-4, absolute_tolerance=1e-8),
}
stage_solvers = {
    ("JFNK", "GMRES"): JFNK(linear_solver=GMRES()),
    ("JFNK", "right GMRES"): JFNK(linear_solver=RightGMRES()),
    ("JFNK", "flexible GMRES"): JFNK(linear_solver=FlexibleGMRES()),
    ("fixed point", "none"): FixedPoint(maximum_iterations=200),
}
# Leapfrog has no damping; on the dissipative upwind transport its computational mode grows, and it
# does not preserve positivity, so f turns negative within the first steps. Its stabilized
# counterpart is the same leapfrog on the non-dissipative central face flux with the same
# reconstruction, the classical leapfrog pairing. BDF4 to BDF6 and Adams-Moulton 3 and 4 are not
# A-stable, and a fixed-point iteration contracts only for steps below the inverse spectral radius;
# at the implicit step $\Delta t=0.1$ they may fail, and their counterpart is the same method at a
# step it is stable for.
implicit_step, stable_step = 0.1, 0.02

# ## The test matrix
#
# 1. **Phase-space reconstruction × face flux** (Vlasov), SSP-RK3 at the transport stability step.
# 2. **Time integrator × phase space** (Vlasov): every integrator on first-order upwind, MUSCL-minmod
#    and MUSCL-van Leer fluxes; explicit integrators at the transport stability step
#    ($\Delta t\approx0.015$), implicit ones at $\Delta t=0.1$, beyond any explicit limit.
# 3. **Stage solver** (Vlasov): Crank-Nicolson and SDIRK2 with every nonlinear/linear pair,
#    Rosenbrock-Euler with every linear solver.
# 4. **Time integrator × shape function** (PIC): every integrator with cloud-in-cell, every shape
#    function with the library's own kick-drift particle-field step and with classical RK4,
#    $\Delta t=0.1$ (explicit) and $0.2$ (implicit).
# 5. **Two-stream instability**: Vlasov and PIC with four integrators.
# 6. **Mesh check**: representative Vlasov combinations on a $32\times64$ grid.

time_space = {
    name: phase_space[(name, "upwind")]
    for name in ("constant (first order)", "MUSCL + minmod", "MUSCL + van Leer")
}
runs = [
    dict(model="vlasov", matrix="phase space", row=row, column=column, spatial=spatial,
         time=TimeMethod("ssprk3"), step=0.05)
    for (row, column), spatial in phase_space.items()
]
runs += [
    {**dict(model="vlasov", matrix="phase space", row=row, column=column, spatial=spatial,
            time=TimeMethod("ssprk3"), step=0.05),
     "variant": "stabilized", "note_stabilized": description}
    for (row, column), (description, spatial) in stabilized_space.items()
]
runs += [
    dict(model="vlasov", matrix="time x space", row=name, column=space, spatial=spatial,
         time=integrator, step=implicit_step if getattr(integrator, "implicit", False) else 0.05)
    for name, integrator in integrators.items()
    for space, spatial in time_space.items()
]
runs += [
    dict(model="vlasov", matrix="stage solver", row=f"{method_name}: {solver_name}",
         column=linear_name, spatial=time_space["MUSCL + minmod"], time=method,
         nonlinear=solver, step=implicit_step)
    for method_name, method in (("Crank-Nicolson", TimeMethod("crank_nicolson")),
                                ("SDIRK2", TimeMethod("dirk_sdirk2")))
    for (solver_name, linear_name), solver in stage_solvers.items()
]
runs += [
    dict(model="vlasov", matrix="stage solver", row="Rosenbrock-Euler", column=linear_name,
         spatial=time_space["MUSCL + minmod"], time=TimeMethod("rosenbrock_euler"),
         linear=linear, step=implicit_step)
    for linear_name, linear in (("GMRES", GMRES()), ("right GMRES", RightGMRES()),
                                ("flexible GMRES", FlexibleGMRES()))
]
runs += [
    dict(model="pic", matrix="PIC time x shape", row=name, column="cloud in cell",
         shape=cloud_in_cell_weights, time=integrator,
         step=0.2 if getattr(integrator, "implicit", False) else 0.1)
    for name, integrator in integrators.items()
]
runs += [
    dict(model="pic", matrix="PIC shape", row=shape_name, column=column,
         shape=shape, time=integrator, step=0.1)
    for shape_name, shape in shapes.items()
    for column, integrator in (("kick-drift particle-field step", None),
                               ("classical RK4", ClassicalRungeKutta4()))
]
runs += [
    {**run, "variant": "stabilized", "spatial": phase_space[(run["column"], "central")],
     "note_stabilized": "leapfrog on the central face flux with the same reconstruction"}
    for run in list(runs)
    if run["row"] == "leapfrog" and run["model"] == "vlasov" and run.get("variant") is None
]
runs += [
    {**run, "variant": "stabilized", "step": stable_step,
     "note_stabilized": f"same method at dt = {stable_step}"}
    for run in list(runs)
    if run.get("variant") is None
    and (getattr(run["time"], "implicit", False) or "fixed point" in run["row"])
    and run["matrix"] != "phase space"
]
runs += [
    dict(model=model, matrix="two-stream", row=name, column=model_name, spatial=VlasovDiscretization(
                 field_factory=PeriodicSpectralFields, phase_flux=upwind_phase_flux,
                 velocity_flux=upwind_phase_flux, transport_step_limit=CourantStepLimit(0.45)),
         shape=cloud_in_cell_weights, time=integrator, step=0.2 if getattr(integrator, "implicit", False) else 0.05)
    for name, integrator in (("SSP-RK3", TimeMethod("ssprk3")), ("classical RK4", ClassicalRungeKutta4()),
                             ("Crank-Nicolson", TimeMethod("crank_nicolson")), ("BDF2", BDF(order=2)))
    for model, model_name in (("vlasov", "Vlasov, MUSCL-minmod"), ("pic", "PIC, cloud in cell"))
]
runs += [
    {**run, "matrix": "mesh check", "request": coarse_landau}
    for run in list(runs)
    if run["matrix"] == "phase space" and run.get("variant") is None
    and run["column"] == "upwind"
    and run["row"] in ("constant (first order)", "MUSCL + minmod", "MUSCL + van Leer", "MUSCL + superbee")
]
print(
    f"{len(runs)} runs: "
    + ", ".join(f"{sum(r['matrix'] == m for r in runs)} {m}" for m in dict.fromkeys(r["matrix"] for r in runs))
)

# ## Run everything
#
# A refused combination is recorded with its message; a run that stops because the distribution
# turned negative, a stage solve did not converge or the state left the finite range is recorded as
# failed. The two-stream runs use beams at $\pm1$ with spread $0.14$ in the same box, on
# $v\in[-2.3,2.3]$. They are seeded with the linear eigenfunction of the growing mode,
# $\delta f\propto\mathrm{Re}\big[e^{ikx}\,(-ik f_0'(v))/(\Gamma+ikv)\big]$, so the field energy grows
# as $e^{2\Gamma t}$ from the start; the growth rate is the slope of $\tfrac12\ln W_E$ for $1<t<5$.

two_stream_v = -2.3 + (np.arange(nv) + 0.5) * 4.6 / nv
two_stream_lattice_v = -2.3 + (np.arange(lattice_v) + 0.5) * 4.6 / lattice_v
two_stream_vlasov, two_stream_lattice = [
    (
        0.5 * (np.exp(-((speeds - beam) ** 2) / (2 * spread**2)) + np.exp(-((speeds + beam) ** 2) / (2 * spread**2)))
        / (np.sqrt(2 * np.pi) * spread)
        + amplitude
        * (
            np.exp(1j * k * points)[:, None]
            * (
                1j * k * 0.5
                * ((speeds - beam) * np.exp(-((speeds - beam) ** 2) / (2 * spread**2))
                   + (speeds + beam) * np.exp(-((speeds + beam) ** 2) / (2 * spread**2)))
                / (np.sqrt(2 * np.pi) * spread**3)
            )
            / (growth_exact + 1j * k * speeds)
        ).real
    )
    for points, speeds, amplitude in ((x, two_stream_v, 2e-4), (x0, two_stream_lattice_v, 1e-3))
]
two_stream = VlasovPoissonRequest(
    DistributionFunctionState(
        electrons,
        {"e": two_stream_vlasov * length / (two_stream_vlasov.sum() * (length / nx) * (4.6 / nv))},
        (nx,),
        (two_stream_v,),
    ),
    VlasovPoissonEquationContract(1, 1),
)
two_stream_particles = PICSpecies(
    positions=np.repeat(x0, lattice_v)[:, None],
    velocities=np.column_stack(
        (np.tile(two_stream_lattice_v, lattice_x), np.zeros((lattice_x * lattice_v, 2)))
    ),
    charge=-1.0,
    mass=1.0,
    weights=two_stream_lattice.ravel() * length / two_stream_lattice.sum(),
)
pic_model = PeriodicParticleInCell(
    mesh=CartesianMesh(shape=(cells,), extents=(length,), periodic=True),
    background_charge_density=1.0,
    permittivity=1.0, light_speed=1.0,
)
declared = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") is None}
for run in runs:
    started = time.perf_counter()
    run.update(status="rejected", note="", rate=np.nan, frequency=np.nan, error=np.nan, steps=0)
    # Stage solvers are explicit objects: the matrix' own choice, or Newton-Krylov with GMRES.
    needs = declared_time_descriptor(run["time"]) if run["time"] is not None else None
    nonlinear = run.get("nonlinear") or (
        stage_solvers[("JFNK", "GMRES")] if needs and needs.requires_nonlinear_solver else None
    )
    linear = run.get("linear") or (GMRES() if needs and needs.requires_linearized_solver else None)
    if run.get("variant") and declared[(run["matrix"], run["row"], run["column"])]["status"] == "ran":
        run.update(status="not needed", seconds=0.0)
        continue
    stream = run["matrix"] == "two-stream"
    try:
        result = (
            (coarse_vlasov if run.get("request") is coarse_landau else vlasov).solve(
                two_stream if stream else run.get("request", landau),
                spatial=run["spatial"],
                time_integrator=run["time"],
                nonlinear_solver=nonlinear,
                linear_solver=linear,
                time_step=run["step"],
                final_time=6.0 if stream else end_time,
            )
            if run["model"] == "vlasov"
            else pic_model.solve(
                species=(two_stream_particles if stream else particles,),
                time_step=run["step"],
                steps=int(round((6.0 if stream else end_time) / run["step"])),
                spatial=pic_model.discretization(
                    shape_function=run["shape"],
                    field_factory=PeriodicSpectralFields,
                    particle_pusher=boris_velocity,
                ),
                time_integrator=(
                    pic_model.midpoint_time_integrator() if run["time"] is None else run["time"]
                ),
                nonlinear_solver=nonlinear,
            )
        )
        energy = result.electric_energy if run["model"] == "vlasov" else result.field_energy
        run["times"], run["energy"], run["steps"] = result.time, energy, len(result.time) - 1
        if stream:
            window = (result.time > 1.0) & (result.time < 5.0)
            run["rate"] = 0.5 * np.polyfit(result.time[window], np.log(energy[window]), 1)[0]
            run["error"] = abs(run["rate"] - growth_exact) / growth_exact
        else:
            peaks = find_peaks(energy)[0]
            peaks = peaks[(result.time[peaks] > 1.0) & (result.time[peaks] < 14.0)]
            run["rate"] = -0.5 * np.polyfit(result.time[peaks], np.log(energy[peaks]), 1)[0]
            run["frequency"] = np.pi / np.polyfit(np.arange(peaks.size), result.time[peaks], 1)[0]
            run["error"] = abs(run["rate"] - gamma_exact) / gamma_exact
        run["status"] = "ran" if np.isfinite(run["error"]) else "failed"
    except Exception as error:
        message = str(error).splitlines()[0]
        run["status"] = (
            "failed"
            if any(word in message for word in ("nonnegative", "converge", "finite", "overflow"))
            else "rejected"
        )
        run["note"] = message
    run["seconds"] = time.perf_counter() - started
    print(
        f"{run['matrix'][:16]:16s} {run['row'][:30]:30s} {run['column'][:24]:24s} "
        f"{run.get('variant') or '':10s} {run['status']:8s} {run['seconds']:6.1f} s  "
        + (f"rate {run['rate']:.4f} (err {run['error']:.1e})" if run["status"] == "ran" else run["note"][:80])
    )

# ## Phase-space reconstruction × face flux
#
# The relative error of the damping rate. Hatched cells failed as declared (the distribution turned
# negative); colour and value are those of the stabilized counterpart named below the matrix.

main = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") is None}
substitute = {(r["matrix"], r["row"], r["column"]): r for r in runs if r.get("variant") == "stabilized"}
hatch = {key for key, r in main.items() if r["status"] != "ran" and substitute.get(key, {}).get("status") == "ran"}
cell = {key: substitute[key] if key in hatch else r for key, r in main.items()}
for matrix_name, rows_, columns_, title, size in (
    ("phase space", list(reconstructions), list(face_fluxes), "Vlasov: reconstruction x face flux, SSP-RK3", (9, 7)),
    ("time x space", list(integrators), list(time_space), "Vlasov: time integrator x phase-space flux", (12, 14)),
    ("stage solver", list(dict.fromkeys(r["row"] for r in runs if r["matrix"] == "stage solver")),
     ["GMRES", "right GMRES", "flexible GMRES", "none"], "Vlasov: implicit stage solvers", (11, 6)),
    ("PIC time x shape", list(integrators), ["cloud in cell"], "PIC: time integrator, cloud in cell", (7, 14)),
    ("PIC shape", list(shapes), ["kick-drift particle-field step", "classical RK4"], "PIC: shape function", (9, 4)),
    ("two-stream", ["SSP-RK3", "classical RK4", "Crank-Nicolson", "BDF2"],
     ["Vlasov, MUSCL-minmod", "PIC, cloud in cell"], "Two-stream instability: growth-rate error", (9, 4)),
):
    fig = plots.figure(1, 1, size=size)
    plots.matrix(
        [[cell[(matrix_name, row, c)]["error"] if (matrix_name, row, c) in cell else np.nan for c in columns_] for row in rows_],
        rows_,
        columns_,
        scale="log",
        label="relative rate error",
        text=[
            [
                (f"{cell[(matrix_name, row, c)]['error']:.1e}\n{cell[(matrix_name, row, c)]['seconds']:.0f} s"
                 if cell[(matrix_name, row, c)]["status"] == "ran" else cell[(matrix_name, row, c)]["status"])
                if (matrix_name, row, c) in cell else "–"
                for c in columns_
            ]
            for row in rows_
        ],
        hatched=[[(matrix_name, row, c) in hatch for c in columns_] for row in rows_],
        notes=(
            ["Hatched: the declared scheme failed; colour and value show its stabilized counterpart:"]
            + list(dict.fromkeys(
                f"  {row}: {substitute[(matrix_name, row, c)]['note_stabilized']}."
                for row in rows_ for c in columns_ if (matrix_name, row, c) in hatch
            ))
            if any((matrix_name, row, c) in hatch for row in rows_ for c in columns_)
            else None
        ),
        title=title,
        ax=fig[0],
    )
    plots.show()

# ## Field energy of representative runs
#
# $W_E(t)$ against the exact envelope $e^{-2\gamma t}$ (scaled to the first maximum).

shown = [
    main[key] for key in (
        ("phase space", "constant (first order)", "upwind"),
        ("phase space", "MUSCL + minmod", "upwind"),
        ("phase space", "MUSCL + superbee", "upwind"),
        ("time x space", "Crank-Nicolson", "MUSCL + minmod"),
        ("PIC time x shape", "classical RK4", "cloud in cell"),
    ) if main[key]["status"] == "ran"
]
fig = plots.figure(1, 1, size=(12, 6))
plots.line(
    y={f"{r['matrix']}: {r['row']}, {r['column']}": (r["times"], r["energy"] / r["energy"][0]) for r in shown},
    reference={"exact damping envelope": (np.linspace(0, end_time, 200), np.exp(-2 * gamma_exact * np.linspace(0, end_time, 200)))},
    xlabel="time t (1/plasma frequency)",
    ylabel="W_E(t) / W_E(0)",
    scale="logy",
    title="Electric field energy, linear Landau damping",
    ax=fig[0],
)
plots.show()

# ## Mesh check: 64 x 128 against 32 x 64 cells

print(f"{'reconstruction':28s} {'error 32x64':>12s} {'error 64x128':>13s}")
for r in (r for r in runs if r["matrix"] == "mesh check"):
    fine_run = main[("phase space", r["row"], r["column"])]
    print(f"{r['row']:28s} {r['error']:12.2e} {fine_run['error']:13.2e}")

# ## What the library refused, and what failed

for status in ("rejected", "failed"):
    selected = [r for r in runs if r["status"] == status]
    print(f"\n{len(selected)} {status}:")
    for r in selected:
        print(f"  {r['matrix'][:16]:16s} {r['row'][:30]:30s} {r['column'][:24]:24s} {r.get('variant') or '':10s} {r['note'][:90]}")
print(
    "\nsummary: "
    + ", ".join(f"{sum(r['status'] == s for r in runs)} {s}" for s in ("ran", "rejected", "failed"))
    + f" of {len(runs)} runs, {sum(r['seconds'] for r in runs) / 60:.0f} min in total"
)

# ## Methods that do not apply
#
# Newton with an assembled Jacobian and sparse direct or preconditioned linear solvers need a stage
# matrix; the phase-space stage couples every cell through the Poisson solve, so the kinetic stages
# are solved matrix-free (JFNK, fixed point, GMRES without preconditioner) and the assembled solvers
# are refused with that reason. Without a preconditioner, right and flexible GMRES generate the same
# Krylov iterates as GMRES. The electromagnetic particle model advances its fields with a discrete
# Maxwell step that has no rate form, so generic time methods drive the electrostatic model only.
