(() => {
  "use strict";
  const pi = Math.PI;
  const c = 299792458;
  const eps = 8.8541878128e-12;

  function line(name, start, stop, evaluate, intervals = 180) {
    const points = Array.from({length: intervals + 1}, (_, index) => {
      const x = start + (stop - start) * index / intervals;
      return [x, evaluate(x)];
    });
    return {name, points};
  }

  function number(value, digits = 3) {
    return Number(value.toPrecision(digits)).toString();
  }

  Object.assign(window.fundamentalStudies ||= {}, {
    "em-distribution": {
      controls: [["Core radius a (mm)", 0.25, 3, 0.05, 1], ["Collector radius R (mm)", 0.25, 6, 0.05, 2]],
      xLabel: "Collector radius (mm)", yLabel: "Fraction of total charge enclosed",
      calculate([a, probe]) {
        const enclosed = r => (r / Math.hypot(r, a)) ** 3;
        return {curves: [line("Smooth charge core", 0, 8, enclosed), line("Ideal point source, R > 0", 0.01, 8, () => 1)],
          result: "At " + probe + " mm the collector encloses " + number(100 * enclosed(probe)) + "% of the charge. Doubling its radius would enclose " + number(100 * enclosed(2 * probe)) + "%."};
      }
    },
    "em-green": {
      controls: [["Sheet position s/L", 0.05, 0.95, 0.01, 0.30], ["Sheet charge density (nC/m²)", -8, 8, 0.1, 3]],
      xLabel: "Position x/L", yLabel: "Potential between grounded plates (V)",
      calculate([s, sigma]) {
        const scale = sigma * 1e-9 * 0.01 / eps;
        const potential = x => scale * Math.min(x, s) * (1 - Math.max(x, s));
        return {curves: [{name: "Grounded 10 mm gap", points: [[0, 0], [s, potential(s)], [1, 0]]}],
          result: "Induced densities are " + number(-sigma * (1 - s)) + " nC/m² on the left and " + number(-sigma * s) + " nC/m² on the right. Potential at the sheet is " + number(potential(s)) + " V."};
      }
    },
    "em-strip": {
      controls: [["Height y/a", 0, 1, 0.01, 0.18], ["Third harmonic at the boundary (V)", -5, 5, 0.1, 3]],
      xLabel: "Position x/a", yLabel: "Potential (V)",
      calculate([height, third]) {
        const first = x => 3 * Math.sin(pi * x);
        const high = x => third * Math.sin(3 * pi * x);
        const interior = x => first(x) * Math.exp(-pi * height) + high(x) * Math.exp(-3 * pi * height);
        return {curves: [line("Driven boundary", 0, 1, x => first(x) + high(x)), line("Interior slice", 0, 1, interior)],
          result: "At the middle of the strip the potential is " + number(interior(0.5)) + " V. The third harmonic retains " + number(100 * Math.exp(-3 * pi * height)) + "% of its boundary amplitude."};
      }
    },
    "em-sphere": {
      controls: [["Observation radius r/a", 0.1, 3, 0.05, 0.5], ["Quadrupole on the sphere (V)", -4, 4, 0.1, 2]],
      xLabel: "Polar angle (degrees)", yLabel: "Potential (V)",
      calculate([r, quadrupole]) {
        function potential(degrees, radius) {
          const u = Math.cos(degrees * pi / 180);
          const radial = order => radius <= 1 ? radius ** order : radius ** (-order - 1);
          return radial(0) + u * radial(1) + quadrupole * (3 * u * u - 1) * radial(2) / 2;
        }
        return {curves: [line("Boundary at r = a", 0, 180, x => potential(x, 1)), line("Selected radius", 0, 180, x => potential(x, r))],
          result: "The north pole and equator read " + number(potential(0, r)) + " V and " + number(potential(90, r)) + " V at this radius. Both branches coincide on the boundary."};
      }
    },
    "em-coils": {
      controls: [["Coil separation / radius", 0.2, 2, 0.02, 1], ["Axial test position / radius", 0.05, 1, 0.01, 0.25]],
      xLabel: "Axial position z/a", yLabel: "Axial magnetic field B(z)/B(0)",
      calculate([separation, probe]) {
        const field = (z, d) => (1 + (z - d / 2) ** 2) ** -1.5 + (1 + (z + d / 2) ** 2) ** -1.5;
        const relative = (z, d) => field(z, d) / field(0, d);
        return {curves: [line("Selected pair", -1.5, 1.5, x => relative(x, separation)), line("Helmholtz spacing d/a = 1", -1.5, 1.5, x => relative(x, 1))],
          result: "At z/a = " + probe + " the selected field differs from the central field by " + number(100 * (relative(probe, separation) - 1)) + "%. The sign distinguishes a central peak from a central dip close to the origin."};
      }
    },
    "em-layer": {
      controls: [["Second relative permittivity", 1, 12, 0.25, 8], ["First layer thickness (mm)", 0.2, 1.8, 0.05, 1]],
      xLabel: "Depth into the two layers (mm)", yLabel: "Accumulated voltage drop (V)",
      calculate([er, thickness]) {
        const series = thickness / 2 + (2 - thickness) / er;
        const firstDrop = 10 * thickness / (2 * series);
        const field1 = 10 / (2 * series);
        const field2 = 10 / (er * series);
        return {curves: [{name: "Matched displacement", points: [[0, 0], [thickness, firstDrop], [2, 10]]}, line("Homogeneous reference", 0, 2, x => 5 * x)],
          result: "Layer 1 drops " + number(firstDrop) + " V. The joint bound interface density is " + number(eps * (field2 - field1) * 1e12) + " nC/m²; the free interface density is zero."};
      }
    },
    "em-magnet": {
      controls: [["Demagnetising factor N", 0, 1, 0.01, 0.33], ["Intrinsic susceptibility", 0, 30, 0.1, 9]],
      xLabel: "Intrinsic susceptibility", yLabel: "Apparent susceptibility M/H applied",
      calculate([factor, chi]) {
        const apparent = (x, n) => x / (1 + n * x);
        return {curves: [line("Selected ellipsoid", 0, 30, x => apparent(x, factor)), line("Sphere N = 1/3", 0, 30, x => apparent(x, 1 / 3))],
          result: "At the selected susceptibility the apparent value is " + number(apparent(chi, factor)) + "; the internal H field is " + number(100 / (1 + factor * chi)) + "% of the applied H field."};
      }
    },
    "em-field-momentum": {
      controls: [["Outer radius b (mm)", 12, 50, 1, 30], ["Initial field B (T)", 0.05, 0.5, 0.01, 0.25]],
      xLabel: "Fraction of initial axial flux removed", yLabel: "Angular momentum per length (10⁻¹² kg m/s)",
      calculate([b, field]) {
        const scale = 0.5 * 20e-9 * field * 1e12;
        const inner = scale * 0.01 ** 2;
        const outer = -scale * (b / 1000) ** 2;
        const total = inner + outer;
        return {curves: [line("Field", 0, 1, x => total * (1 - x)), line("Inner shell", 0, 1, x => inner * x), line("Outer shell", 0, 1, x => outer * x), line("Field plus both shells", 0, 1, () => total)],
          result: "After switch-off the inner shell carries " + number(inner) + " and the outer shell " + number(outer) + " in plotted units. Their sum equals the initial field value " + number(total) + "."};
      }
    },
    "em-fresnel": {
      controls: [["Incident refractive index", 1, 2, 0.05, 1], ["Transmitted refractive index", 1, 2, 0.05, 1.5]],
      xLabel: "Angle of incidence (degrees)", yLabel: "Reflected power fraction",
      calculate([n1, n2]) {
        function reflect(angle, sPolarised) {
          const theta = angle * pi / 180;
          const transmittedSine = n1 / n2 * Math.sin(theta);
          if (transmittedSine >= 1) return 1;
          const ci = Math.cos(theta);
          const ct = Math.sqrt(1 - transmittedSine ** 2);
          const a = sPolarised ? n1 * ci : n1 * ct;
          const b = sPolarised ? n2 * ct : n2 * ci;
          return ((a - b) / (a + b)) ** 2;
        }
        const angle = Math.atan(n2 / n1) * 180 / pi;
        const critical = n1 > n2 ? " The critical angle is " + number(Math.asin(n2 / n1) * 180 / pi) + "°." : "";
        return {curves: [line("s polarisation", 0, 89.9, x => reflect(x, true)), line("p polarisation", 0, 89.9, x => reflect(x, false))],
          result: n1 === n2 ? "Index matching gives zero reflection at every incidence angle." : "The Brewster minimum occurs at " + number(angle) + "°. At 45°, reflected power is " + number(100 * reflect(45, true)) + "% (s) and " + number(100 * reflect(45, false)) + "% (p)." + critical};
      }
    },
    "em-dispersion": {
      controls: [["Damping γ/ω₀", 0.05, 0.8, 0.01, 0.20], ["Oscillator strength ωₚ²/ω₀²", 0.1, 1, 0.05, 0.5]],
      xLabel: "Drive frequency ω/ω₀", yLabel: "Susceptibility",
      calculate([damping, strength]) {
        const denom = x => (1 - x * x) ** 2 + (damping * x) ** 2;
        return {curves: [line("Re χ", 0, 2, x => strength * (1 - x * x) / denom(x), 320), line("Im χ", 0, 2, x => strength * damping * x / denom(x), 320)],
          result: "At ω = ω₀ the susceptibility is purely imaginary with Im χ = " + number(strength / damping) + ". At zero frequency the real susceptibility is " + number(strength) + "."};
      }
    },
    "em-retarded": {
      controls: [["Second detector radius (m)", 0.3, 3, 0.05, 1.8], ["Pulse duration (ns)", 2, 8, 0.25, 4]],
      xLabel: "Time since emission started (ns)", yLabel: "Field relative to first detector peak",
      calculate([r, width]) {
        function pulse(t, radius) {
          const phase = (t - radius / c * 1e9) / width;
          return phase <= 0 || phase >= 1 ? 0 : 0.3 / radius * Math.sin(pi * phase) ** 2;
        }
        return {curves: [line("At 0.30 m", 0, 20, x => pulse(x, 0.3), 320), line("Second detector", 0, 20, x => pulse(x, r), 320)],
          result: "The second trace starts " + number((r - 0.30) / c * 1e9) + " ns after the first. Its peak is " + number(100 * 0.30 / r) + "% of the first peak."};
      }
    },
    "em-moving": {
      controls: [["Source speed β", 0, 0.9, 0.01, 0.6]],
      xLabel: "Angle from velocity and acceleration (degrees)", yLabel: "Emission power per solid angle / peak",
      calculate([beta]) {
        const pattern = (theta, speed) => Math.sin(theta) ** 2 / (1 - speed * Math.cos(theta)) ** 5;
        const maximumAngle = Math.acos(5 * beta / (1 + Math.sqrt(1 + 15 * beta ** 2)));
        const maximum = pattern(maximumAngle, beta);
        return {curves: [line("Moving source", 0, 180, x => pattern(x * pi / 180, beta) / maximum), line("Slow-source limit", 0, 180, x => Math.sin(x * pi / 180) ** 2)],
          result: "Peak emission is at " + number(maximumAngle * 180 / pi) + "°. Its unnormalised height is " + number(maximum) + " times the slow-source maximum at the same instantaneous acceleration."};
      }
    },
    "em-boost": {
      controls: [["Electric field E (kV/m)", 20, 200, 5, 90], ["Magnetic field cB (kV/m)", 20, 200, 5, 150], ["Observer speed β", -0.9, 0.9, 0.01, 0.6]],
      xLabel: "Observer speed v/c", yLabel: "Transformed field (kV/m)",
      calculate([e, b, probe]) {
        const electric = x => (e - x * b) / Math.sqrt(1 - x * x);
        const magnetic = x => (b - x * e) / Math.sqrt(1 - x * x);
        return {curves: [line("Transformed E", -0.95, 0.95, electric), line("Transformed cB", -0.95, 0.95, magnetic)],
          result: "At the selected speed E′ = " + number(electric(probe)) + " kV/m and cB′ = " + number(magnetic(probe)) + " kV/m. E′² − c²B′² = " + number(e * e - b * b, 5) + " (kV/m)²."};
      }
    },
    "em-action": {
      controls: [["Trial distortion η", -0.45, 0.45, 0.01, 0.1]],
      xLabel: "Position x/L", yLabel: "Potential relative to the electrode voltage",
      calculate([eta]) {
        return {curves: [line("Trial function", 0, 1, x => x + eta * Math.sin(pi * x)), line("Stationary potential", 0, 1, x => x)],
          result: "The trial field energy exceeds the stationary value by " + number(50 * pi * pi * eta * eta) + "%. Reversing the distortion changes its direction while preserving this energy difference."};
      }
    },
    "em-cavity": {
      controls: [["Loaded quality factor Q", 1000, 15000, 100, 6000], ["Cavity length d (mm)", 40, 120, 2, 80]],
      xLabel: "Time after switch-off (µs)", yLabel: "Fraction of initial value",
      calculate([q, d]) {
        const frequency = c / 2 * Math.hypot(1 / 0.06, 1000 / d);
        const decayTime = q / (2 * pi * frequency) * 1e6;
        return {curves: [line("Stored energy", 0, 3, x => Math.exp(-x / decayTime)), line("Field envelope", 0, 3, x => Math.exp(-x / (2 * decayTime)))],
          result: "The mode frequency is " + number(frequency / 1e9, 4) + " GHz with full power linewidth " + number(frequency / q / 1e3) + " kHz. The energy e-folding time is " + number(decayTime) + " µs."};
      }
    },
    "em-scatter": {
      controls: [["Sphere radius (nm)", 5, 25, 0.5, 20], ["Relative permittivity", 1, 4, 0.05, 2.25]],
      xLabel: "Vacuum wavelength (nm)", yLabel: "Scattering cross section (nm²)",
      calculate([radius, er]) {
        const contrast = (er - 1) / (er + 2);
        const crossSection = wavelength => 8 * pi / 3 * (2 * pi / wavelength) ** 4 * radius ** 6 * contrast ** 2;
        return {curves: [line("Rayleigh sphere", 400, 800, crossSection)],
          result: "The leading cross section at 600 nm is " + number(crossSection(600)) + " nm². The shortest plotted wavelength gives ka = " + number(2 * pi * radius / 400) + "; finite-size corrections grow as this ratio grows."};
      }
    },
    "em-circuit": {
      controls: [["Open line length (cm)", 5, 30, 1, 15], ["Test frequency / first pole", 0.05, 0.9, 0.01, 0.6]],
      xLabel: "Frequency (MHz)", yLabel: "Magnitude of input admittance (mS)",
      calculate([length, fraction]) {
        const poleMHz = 2e8 / (4 * length / 100) / 1e6;
        const phase = frequency => pi * frequency / (2 * poleMHz);
        const exact = f => 20 * Math.tan(phase(f));
        const lumped = f => 20 * phase(f);
        const test = fraction * poleMHz;
        return {curves: [line("Distributed line", 0, 0.9 * poleMHz, exact), line("Lumped capacitor", 0, 0.9 * poleMHz, lumped)],
          result: "At " + number(test) + " MHz distributed admittance exceeds lumped admittance by " + number(100 * (exact(test) / lumped(test) - 1)) + "%. The first pole is at " + number(poleMHz) + " MHz."};
      }
    },
    "em-eikonal": {
      controls: [["Index gradient g (1/m)", 8, 35, 0.5, 20], ["Common input slope (mrad)", -15, 15, 0.5, 0]],
      xLabel: "Axial distance z (mm)", yLabel: "Ray height y (mm)",
      calculate([g, slope]) {
        const curves = [-1, -0.5, 0, 0.5, 1].map(height => line("Launch at " + height + " mm", 0, 220, z => height * Math.cos(g * z / 1000) + slope / g * Math.sin(g * z / 1000)));
        return {curves, result: "The first crossing occurs at z = " + number(1000 * pi / (2 * g)) + " mm and y = " + number(slope / g) + " mm. Finite-wavelength diffraction limits the intensity at that crossing."};
      }
    },
    "em-self-force": {
      controls: [["Mean velocity / oscillating speed", -1, 1, 0.05, 0], ["Emission phase (degrees)", 0, 360, 5, 0]],
      xLabel: "Oscillation phase (degrees)", yLabel: "Power relative to mτω²vₐ²",
      calculate([drift, probe]) {
        const radiated = t => Math.cos(t * pi / 180) ** 2;
        const self = t => {
          const v = Math.sin(t * pi / 180);
          return -(drift + v) * v;
        };
        const stored = t => -radiated(t) - self(t);
        return {curves: [line("Power radiated", 0, 360, radiated), line("Self-force work on charge", 0, 360, self), line("Bound-field energy rate", 0, 360, stored)],
          result: "At " + probe + "°, radiation, self-force work and bound-field rate are " + [radiated(probe), self(probe), stored(probe)].map(x => number(Math.abs(x) < 1e-12 ? 0 : x)).join(", ") + " in the plotted units. Their sum is zero."};
      }
    },
    "em-diffraction": {
      controls: [["Fresnel number (full slit width)", 0, 6, 0.05, 0.5], ["Midpoint quadrature cells", 4, 128, 4, 16]],
      xLabel: "Reduced detector coordinate ax/(λz)", yLabel: "Intensity relative to far-field axial reference",
      calculate([fresnel, cells]) {
        function intensity(u, count) {
          let real = 0;
          let imaginary = 0;
          for (let j = 0; j < count; j++) {
            const s = (j + 0.5) / count - 0.5;
            const phase = pi * fresnel * s * s - 2 * pi * u * s;
            real += Math.cos(phase);
            imaginary += Math.sin(phase);
          }
          return (real * real + imaginary * imaginary) / (count * count);
        }
        const far = x => x === 0 ? 1 : (Math.sin(pi * x) / (pi * x)) ** 2;
        return {curves: [line("Selected quadrature", -3, 3, u => intensity(u, cells)), line("Refined, twice as many cells", -3, 3, u => intensity(u, 2 * cells)), line("Fraunhofer limit", -3, 3, far)],
          result: "At u = 1 the two quadratures predict intensities " + number(intensity(1, cells), 4) + " and " + number(intensity(1, 2 * cells), 4) + ". A persistent nonzero value for finite Fresnel number measures near-field interference."};
      }
    },
    "em-worldline": {
      controls: [["Electric field magnitude (kV/m)", 20, 150, 5, 100], ["Lab time for a reading (ns)", 2, 50, 1, 17]],
      xLabel: "Elapsed lab time (ns)", yLabel: "Speed magnitude relative to c",
      calculate([strength, probe]) {
        const force = 1.602176634e-19 * strength * 1000;
        const characteristicTime = 9.1093837e-31 * c / force * 1e9;
        const momentum = t => t / characteristicTime;
        const speed = t => momentum(t) / Math.hypot(1, momentum(t));
        const properTime = characteristicTime * Math.asinh(momentum(probe));
        const kineticKeV = 510.999 * momentum(probe) ** 2 / (Math.hypot(1, momentum(probe)) + 1);
        return {curves: [line("Relativistic speed", 0, 50, speed), line("Newtonian extrapolation", 0, 50, momentum), line("Light speed", 0, 50, () => 1)],
          result: "At " + probe + " ns laboratory time the electron speed is " + number(speed(probe)) + " c. Its clock has advanced " + number(properTime) + " ns and it has gained " + number(kineticKeV) + " keV."};
      }
    }
  });
})();
