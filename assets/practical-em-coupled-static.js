(() => {
  const epsilon0 = 8.8541878128e-12;
  const boltzmann = 1.380649e-23;

  function sample(left, right, intervals, value) {
    return Array.from({ length: intervals + 1 }, (_, index) => {
      const x = left + (right - left) * index / intervals;
      return [x, value(x)];
    });
  }

  function bisect(left, right, value) {
    const positiveOnLeft = value(left) > 0;
    for (let iteration = 0; iteration < 56; iteration++) {
      const mid = (left + right) / 2;
      if ((value(mid) > 0) === positiveOnLeft) left = mid;
      else right = mid;
    }
    return (left + right) / 2;
  }

  function contactGradient(x, z, halfSpacing) {
    const factor = -1 / (2 * Math.PI * 0.2);
    const source = x + halfSpacing;
    const sink = x - halfSpacing;
    const sourceR3 = Math.hypot(source, z) ** 3;
    const sinkR3 = Math.hypot(sink, z) ** 3;
    return [factor * (source / sourceR3 - sink / sinkR3), factor * z * (1 / sourceR3 - 1 / sinkR3)];
  }

  Object.assign(window.fundamentalStudies ||= {}, {
    'em-coupled-static-pull-in': {
      controls: [['Bias as fraction of static pull-in', 0.35, 1.1, 0.005, 0.93], ['Spring stiffness (N/m)', 1, 4, 0.1, 2]],
      xLabel: 'Displacement / initial gap',
      yLabel: 'Force (µN)',
      calculate([fraction, stiffness]) {
        const gap = 2e-6;
        const area = 1e-8;
        const pullIn = Math.sqrt(8 * stiffness * gap ** 3 / (27 * epsilon0 * area));
        const voltage = fraction * pullIn;
        const scale = stiffness * gap * 1e6;
        const loading = 4 / 27 * fraction ** 2;
        const mismatch = s => s * (1 - s) ** 2 - loading;
        let equilibrium = 'There is no open-gap static equilibrium.';
        if (fraction < 1) {
          const stable = bisect(0, 1 / 3, mismatch);
          const unstable = bisect(1 / 3, 1, mismatch);
          const hz = Math.sqrt(stiffness * (1 - 3 * stable) / (1 - stable) / 5e-10) / (2 * Math.PI);
          equilibrium = 'Stable x/g = ' + stable.toFixed(3) + ', unstable x/g = ' + unstable.toFixed(3) +
            ', small-signal frequency ' + (hz / 1000).toFixed(2) + ' kHz.';
        } else if (Math.abs(fraction - 1) < 1e-12) {
          equilibrium = 'Both branches meet at x/g = 1/3 with zero tangent stiffness.';
        }
        return {
          curves: [
            { name: 'Spring reaction', points: sample(0, 0.86, 180, s => scale * s) },
            { name: 'Constant-voltage attraction', points: sample(0, 0.86, 180, s => scale * loading / (1 - s) ** 2) },
            { name: 'Charge fixed at initial open-gap value', points: sample(0, 0.86, 180, () => scale * loading) }
          ],
          result: 'Bias ' + voltage.toFixed(3) + ' V; static pull-in ' + pullIn.toFixed(3) + ' V. ' + equilibrium
        };
      }
    },
    'em-coupled-static-electrode-kernel': {
      controls: [['Inclusion depth (mm)', 3, 20, 0.25, 8], ['Sensing half-spacing (mm)', 2, 12, 0.25, 5]],
      xLabel: 'Lateral inclusion centre (mm)',
      yLabel: 'First-order transfer change (mΩ)',
      calculate([depthMM, senseMM]) {
        const drive = 0.015;
        const sense = senseMM / 1000;
        const volume = 4 * Math.PI * 0.001 ** 3 / 3;
        const change = (xMM, zMM) => {
          const driveField = contactGradient(xMM / 1000, zMM / 1000, drive);
          const senseField = contactGradient(xMM / 1000, zMM / 1000, sense);
          return -1000 * 0.02 * volume * (driveField[0] * senseField[0] + driveField[1] * senseField[1]);
        };
        const base = (1 / (drive - sense) - 1 / (drive + sense)) / (Math.PI * 0.2);
        return {
          curves: [
            { name: 'Selected depth', points: sample(-28, 28, 224, x => change(x, depthMM)) },
            { name: 'Twice the selected depth', points: sample(-28, 28, 224, x => change(x, 2 * depthMM)) }
          ],
          result: 'Centre perturbation ' + change(0, depthMM).toFixed(3) + ' mΩ; unperturbed transfer ' + base.toFixed(2) + ' Ω. Sphere radius 1 mm, conductivity increase 0.020 S/m.'
        };
      }
    },
    'em-coupled-static-sliding-dielectric': {
      controls: [['Bias at preparation (V)', 10, 100, 1, 80], ['Edge transition length (mm)', 0.2, 2, 0.05, 0.5]],
      xLabel: 'Position of the leading edge (mm)',
      yLabel: 'Lateral force (µN)',
      calculate([volts, edgeMM]) {
        const length = 0.03;
        const ell = edgeMM / 1000;
        const slope = epsilon0 * 0.020 * (1 / 0.00020 - 1 / 0.00050);
        const empty = epsilon0 * 0.020 * length / 0.00050;
        const capacitance = x => empty + slope / 2 * (length + ell * (Math.log(Math.cosh(x / ell)) - Math.log(Math.cosh((x - length) / ell))));
        const gradient = x => slope / 2 * (Math.tanh(x / ell) - Math.tanh((x - length) / ell));
        const charge = capacitance(length / 2) * volts;
        return {
          curves: [
            { name: 'Connected voltage source', points: sample(-5, 35, 200, mm => 0.5e6 * volts ** 2 * gradient(mm / 1000)) },
            { name: 'Charge fixed at half insertion', points: sample(-5, 35, 200, mm => 0.5e6 * (charge / capacitance(mm / 1000)) ** 2 * gradient(mm / 1000)) }
          ],
          result: 'Interior fixed-voltage pull ' + (0.5e6 * volts ** 2 * slope).toFixed(3) + ' µN; entrance force gradient ' +
          (1000 * volts ** 2 * slope / (4 * ell)).toFixed(3) + ' mN/m. The two preparations coincide at 15 mm.'
        };
      }
    },
    'em-coupled-static-bridge-noise': {
      controls: [['Input and stray capacitance (pF)', 0, 12, 0.25, 4], ['Carrier frequency (kHz)', 10, 250, 2.5, 100]],
      xLabel: 'Carrier voltage on each drive (V rms)',
      yLabel: 'Displacement density (pm/√Hz)',
      calculate([strayPF, frequencyKHz]) {
        const gap = 100e-6;
        const c0 = 2e-12;
        const total = 2 * c0 + strayPF * 1e-12;
        const response = 2 * c0 / total;
        const omega = 2 * Math.PI * 1000 * frequencyKHz;
        const h = 2 * c0 * (1 - response) / gap ** 2;
        const baseSpring = 0.020;
        const thermalForce = Math.sqrt(4 * boltzmann * 300 * 1e-6);
        const white = Math.hypot(1e-8, 1e-14 / (omega * total)) * gap / response;
        const ceiling = Math.min(2.5, Math.sqrt(1e-7 / (2 * omega * c0 * 0.010)), h ? Math.sqrt(0.1 * baseSpring / h) : Infinity);
        const readout = volts => 1e12 * white / volts;
        const brownian = volts => 1e12 * thermalForce / (baseSpring - h * volts ** 2);
        const combined = volts => Math.hypot(readout(volts), brownian(volts));
        const derivative = volts => -(white ** 2) / volts ** 4 + 2 * h * thermalForce ** 2 / (baseSpring - h * volts ** 2) ** 3;
        const best = h > 0 && derivative(ceiling) > 0 ? bisect(0.01, ceiling, derivative) : ceiling;
        return {
          curves: [
            { name: 'Electrical readout', points: sample(0.2, ceiling, 160, readout) },
            { name: 'Mechanical Brownian motion', points: sample(0.2, ceiling, 160, brownian) },
            { name: 'Combined free-probe noise', points: sample(0.2, ceiling, 160, combined) }
          ],
          result: 'Maximum allowed drive ' + ceiling.toFixed(3) + ' V rms. Minimum combined noise ' +
          combined(best).toFixed(2) + ' pm/√Hz at ' + best.toFixed(3) + ' V rms.'
        };
      }
    }
  });
})();
