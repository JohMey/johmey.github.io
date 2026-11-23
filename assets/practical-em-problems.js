(() => {
  "use strict";
  const pi = Math.PI;
  const format = value => Number(value.toPrecision(4)).toString();

  function line(name, lo, hi, evaluate, count = 144) {
    const points = Array.from({length: count + 1}, (_, j) => {
      const x = lo + (hi - lo) * j / count;
      return [x, evaluate(x)];
    });
    return {name, points};
  }

  // Integral representations remain well conditioned for all arguments used here.
  function bessel(order, argument) {
    const points = 48;
    let result = 0;
    for (let j = 0; j < points; j++) {
      const angle = pi * (j + 0.5) / points;
      result += Math.cos(order * angle - argument * Math.sin(angle));
    }
    return result / points;
  }

  const zeros = [2.4048255577, 5.5200781103, 8.6537279129, 11.7915344391,
    14.9309177085, 18.0710639679, 21.2116366299, 24.3524715307];
  const coefficients = zeros.map(root => 8 / (root ** 3 * bessel(1, root)));

  Object.assign(window.fundamentalStudies ||= {}, {
    "em-annulus": {
      controls: [["Hole / outer radius", 0.1, 0.9, 0.01, 0.5], ["Radial integration cells", 2, 64, 2, 8]],
      xLabel: "Axial height z/b", yLabel: "Axial field 2ε₀E/σ",
      calculate([hole, cells]) {
        const exact = z => z * (1 / Math.hypot(z, hole) - 1 / Math.hypot(z, 1));
        function midpoint(z) {
          let sum = 0;
          const step = (1 - hole) / cells;
          for (let i = 0; i < cells; i++) {
            const r = hole + (i + 0.5) * step;
            sum += r * step * z / Math.hypot(z, r) ** 3;
          }
          return sum;
        }
        const h = hole ** (4 / 3);
        const peak = Math.sqrt((h - hole * hole) / (1 - h));
        return {curves: [line("Analytic integral", 0, 2, exact), line("Ring quadrature", 0, 2, midpoint)],
          result: "The analytic peak is at z/b = " + format(peak) + " with field " + format(exact(peak)) + ". At that location the quadrature differs by " + format(100 * (midpoint(peak) / exact(peak) - 1)) + "%."};
      }
    },
    "em-sphere-images": {
      controls: [["Source distance d/a", 1.2, 5, 0.05, 2], ["Isolated conductor charge Qc/q", -1, 1, 0.05, 0]],
      xLabel: "Polar angle from source (degrees)", yLabel: "Surface density in units q/(4πa²)",
      calculate([d, charge]) {
        const grounded = theta => -(d * d - 1) / (d * d + 1 - 2 * d * Math.cos(theta * pi / 180)) ** 1.5;
        const isolated = theta => grounded(theta) + charge + 1 / d;
        const forceG = -d / (d * d - 1) ** 2;
        const forceC = forceG + (charge + 1 / d) / (d * d);
        return {curves: [line("Grounded conductor", 0, 180, grounded), line("Isolated conductor", 0, 180, isolated)],
          result: "Grounding draws net charge " + format(-1 / d) + " q. Forces in units q²/(4πε₀a²) are " + format(forceG) + " grounded and " + format(forceC) + " isolated; positive means away from the sphere."};
      }
    },
    "em-bessel": {
      controls: [["Height z/a", 0, 1.5, 0.01, 0.25], ["Retained modes", 1, 8, 1, 3]],
      xLabel: "Radius r/a", yLabel: "Potential V/V₀",
      calculate([height, modes]) {
        function sum(r, count) {
          let potential = 0;
          for (let j = 0; j < count; j++) potential += coefficients[j] * bessel(0, zeros[j] * r) * Math.exp(-zeros[j] * height);
          return potential;
        }
        return {curves: [line("Selected modal sum", 0, 1, x => sum(x, modes)), line("Eight modes", 0, 1, x => sum(x, 8)), line("Imposed end voltage", 0, 1, x => 1 - x * x)],
          result: "The on-axis potential is " + format(sum(0, modes)) + " V₀ with the selected modes, versus " + format(sum(0, 8)) + " V₀ using eight. The slowest mode has e-folding length " + format(1 / zeros[0]) + " a."};
      }
    },
    "em-coenergy": {
      controls: [["First current at preparation (A)", -3, 3, 0.1, 2], ["Second current at preparation (A)", -2, 2, 0.1, 1]],
      xLabel: "Coil angle (degrees)", yLabel: "Electromagnetic torque (mN m)",
      calculate([i1, i2]) {
        const flux1 = 4 * i1 + 1.5 * i2;
        const flux2 = 1.5 * i1 + 9 * i2;
        const currentTorque = theta => -3 * i1 * i2 * Math.sin(theta * pi / 180);
        function fluxTorque(theta) {
          const angle = theta * pi / 180;
          const mutual = 3 * Math.cos(angle);
          const determinant = 36 - mutual * mutual;
          const first = (9 * flux1 - mutual * flux2) / determinant;
          const second = (4 * flux2 - mutual * flux1) / determinant;
          return -3 * first * second * Math.sin(angle);
        }
        return {curves: [line("Currents held fixed", 0, 180, currentTorque), line("Fluxes held fixed after 60°", 0, 180, fluxTorque)],
          result: "At 60° both preparations give " + format(currentTorque(60)) + " mN m. At 90° the current-controlled and flux-controlled torques are " + format(currentTorque(90)) + " and " + format(fluxTorque(90)) + " mN m."};
      }
    },
    "em-charging": {
      controls: [["Plate radius (mm)", 10, 40, 1, 20], ["Charging current (mA)", -8, 8, 0.25, 4], ["Measurement radius / plate radius", 0.05, 0.95, 0.05, 0.5]],
      xLabel: "Distance from the centre (mm)", yLabel: "Current (mA)",
      calculate([radius, current, fraction]) {
        const gap = x => current * (x / radius) ** 2;
        const plate = x => current - gap(x);
        const probe = fraction * radius;
        const magnetic = 200 * current * probe / (radius * radius);
        return {curves: [line("Displacement through gap disk", 0, radius, gap), line("Conduction across plate ring", 0, radius, plate), line("Full lead current", 0, radius, () => current)],
          result: "At r = " + format(probe) + " mm the gap encloses " + format(gap(probe)) + " mA and the radial plate current is " + format(plate(probe)) + " mA. The local azimuthal magnetic field is " + format(magnetic) + " nT."};
      }
    },
    "em-half-space": {
      controls: [["Loss ratio σ/(ωε)", 0.05, 8, 0.05, 2], ["Lossless impedance / incident impedance", 0.25, 2, 0.05, 1]],
      xLabel: "Depth k_d z", yLabel: "Fraction of incident power",
      calculate([loss, mismatch]) {
        const phase = Math.sqrt((Math.hypot(1, loss) + 1) / 2);
        const attenuation = loss / (2 * phase);
        const magnitudeSquared = phase * phase + attenuation * attenuation;
        const real = mismatch * phase / magnitudeSquared;
        const imag = -mismatch * attenuation / magnitudeSquared;
        const reflection = ((real - 1) ** 2 + imag * imag) / ((real + 1) ** 2 + imag * imag);
        const forward = depth => (1 - reflection) * Math.exp(-2 * attenuation * depth);
        const heat = depth => 1 - reflection - forward(depth);
        return {curves: [line("Still travelling inside", 0, 4, forward), line("Heat deposited to this depth", 0, 4, heat), line("Reflected at entry", 0, 4, () => reflection)],
          result: "The face reflects " + format(100 * reflection) + "% of incident power. The field attenuation α/k_d is " + format(attenuation) + " and impedance relative to the incident medium is " + format(real) + " − " + format(-imag) + "i."};
      }
    },
    "em-crystal": {
      controls: [["Ordinary principal index", 1.3, 1.9, 0.01, 1.5], ["Extraordinary principal index", 1.3, 1.9, 0.01, 1.6], ["Probe wave-normal angle (degrees)", 0, 90, 1, 45]],
      xLabel: "Wave-normal angle to optic axis (degrees)", yLabel: "Energy angle relative to wave normal (degrees)",
      calculate([ordinary, extraordinary, probe]) {
        function walkoff(theta) {
          const t = theta * pi / 180;
          return Math.atan2(ordinary ** 2 * Math.sin(t), extraordinary ** 2 * Math.cos(t)) * 180 / pi - theta;
        }
        const t = probe * pi / 180;
        const index = 1 / Math.hypot(Math.sin(t) / extraordinary, Math.cos(t) / ordinary);
        const shift = 10 * Math.tan(walkoff(probe) * pi / 180);
        return {curves: [line("Extraordinary energy walk-off", 0, 90, walkoff), line("Ordinary energy walk-off", 0, 90, () => 0)],
          result: "At the chosen angle the extraordinary phase index is " + format(index) + ". A 10 mm plate normal to the wavevector displaces its energy by " + format(shift) + " mm relative to the ordinary ray."};
      }
    },
    "em-quadrupole": {
      controls: [["Separation d/λ", 0.01, 0.4, 0.005, 0.035], ["Opposing amplitude deficit", 0, 0.3, 0.01, 0]],
      xLabel: "Angle to dipole axis (degrees)", yLabel: "Intensity relative to one dipole peak",
      calculate([separation, deficit]) {
        const kd = 2 * pi * separation;
        function intensity(angle, smallPhase) {
          const theta = angle * pi / 180;
          const phase = kd * Math.cos(theta) / 2;
          const interference = smallPhase ? phase : Math.sin(phase);
          return Math.sin(theta) ** 2 * (deficit ** 2 + 4 * (1 - deficit) * interference ** 2);
        }
        let integral = 0;
        const cells = 256;
        for (let i = 0; i < cells; i++) {
          const angle = (i + 0.5) * pi / cells;
          integral += intensity(angle * 180 / pi, false) * Math.sin(angle) * pi / cells;
        }
        return {curves: [line("Full phase difference", 0, 180, x => intensity(x, false)), line("Long-wavelength expansion", 0, 180, x => intensity(x, true))],
          result: "Integrating over solid angle gives " + format(100 * integral * 3 / 4) + "% of the total power of one dipole. At 90° the intensity relative to its equatorial peak is " + format(deficit ** 2) + "."};
      }
    }
  });
})();
