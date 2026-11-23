(() => {
  function sample(a, b, n, value) {
    return Array.from({ length: n + 1 }, (_, i) => {
      const x = a + (b - a) * i / n;
      return [x, value(x)];
    });
  }

  function logError(value, reference) {
    return Math.log10(Math.max(1e-17, Math.abs(value - reference)));
  }

  function relaxationFit(rate, observations) {
    const samples = [0, 1, 2, 3].map(t => Math.exp(-rate * t));
    const meanY = observations.reduce((sum, y) => sum + y, 0) / observations.length;
    const meanS = samples.reduce((sum, s) => sum + s, 0) / samples.length;
    const numerator = samples.reduce((sum, s, i) => sum + (s - meanS) * (observations[i] - meanY), 0);
    const denominator = samples.reduce((sum, s) => sum + (s - meanS) ** 2, 0);
    const amplitude = numerator / denominator;
    const baseline = meanY - amplitude * meanS;
    const squaredResidual = (a, b) => samples.reduce((sum, s, i) => sum + (observations[i] - a * s - b) ** 2, 0);
    return { amplitude, baseline, profiled: squaredResidual(amplitude, baseline), fixed: squaredResidual(4, 1) };
  }

  function locateEvent(h, level, cubic) {
    const truth = Math.acos(level);
    const left = h * Math.floor(truth / h), right = left + h;
    const y0 = Math.cos(left), y1 = Math.cos(right);
    const d0 = -Math.sin(left), d1 = -Math.sin(right);
    const interpolate = s => cubic ?
      (2 * s ** 3 - 3 * s ** 2 + 1) * y0 + (s ** 3 - 2 * s ** 2 + s) * h * d0 +
      (-2 * s ** 3 + 3 * s ** 2) * y1 + (s ** 3 - s ** 2) * h * d1 :
      (1 - s) * y0 + s * y1;
    let low = 0, high = 1;
    for (let j = 0; j < 48; j++) {
      const middle = (low + high) / 2;
      if (interpolate(middle) > level) low = middle;
      else high = middle;
    }
    return left + h * (low + high) / 2;
  }

  function burgersState(left, right, speed) {
    if (left >= right) return speed < (left + right) / 2 ? left : right;
    return Math.max(left, Math.min(right, speed));
  }

  Object.assign(window.fundamentalStudies ||= {}, {
    'numerical-methods-nonnormal': {
      controls: [['Coupling K', 0, 20, 0.25, 12], ['Initial angle (degrees)', 0, 180, 1, 90]],
      xLabel: 'Scaled time', yLabel: 'State norm or maximum gain',
      calculate([coupling, angle]) {
        const theta = angle * Math.PI / 180;
        const value = (t, optimal) => {
          const a = Math.exp(-t), d = a * a, b = coupling * (a - d);
          if (!optimal) return Math.hypot(a * Math.cos(theta) + b * Math.sin(theta), d * Math.sin(theta));
          const trace = a * a + d * d + b * b;
          return Math.sqrt((trace + Math.sqrt(Math.max(0, trace * trace - 4 * a * a * d * d))) / 2);
        };
        const chosen = sample(0, 4, 160, t => value(t, false));
        const optimal = sample(0, 4, 160, t => value(t, true));
        return { curves: [{ name: 'Selected unit input', points: chosen }, { name: 'Optimal input at each time', points: optimal }],
          result: 'At t = ln 2, selected norm ' + value(Math.log(2), false).toFixed(3) + ', optimal gain ' +
          value(Math.log(2), true).toFixed(3) + '. Both eigenvalues stay −1 and −2.' };
      }
    },
    'numerical-methods-derivative-step': {
      controls: [['Evaluation point x', -2, 2, 0.1, 0], ['Largest plotted log₁₀ step', -4, 0, 0.25, -1]],
      xLabel: 'log₁₀ of step h', yLabel: 'log₁₀ absolute derivative error',
      calculate([x, end]) {
        const truth = Math.exp(x);
        const rules = [
          ['Forward difference', h => (Math.exp(x + h) - truth) / h],
          ['Centred difference', h => (Math.exp(x + h) - Math.exp(x - h)) / (2 * h)],
          ['Complex-step imaginary part', h => truth * Math.sin(h) / h]
        ];
        const curves = rules.map(([name, fn]) => ({ name, points: sample(-16, end, 135, logH => logError(fn(10 ** logH), truth)) }));
        return { curves, result: 'Exact derivative exp(x) = ' + truth.toPrecision(6) +
          '. Errors below 10⁻¹⁷ are drawn at the plotting floor. The complex exponential is evaluated analytically in real and imaginary parts.' };
      }
    },
    'numerical-methods-rational': {
      controls: [['Curvature β', 0.25, 1.5, 0.05, 1.2], ['Highest even power / 2', 1, 7, 1, 4], ['Curvature bias (%)', -20, 20, 1, 5]],
      xLabel: 'Real argument x', yLabel: 'Response',
      calculate([beta, order, bias]) {
        const fitted = beta * (1 + bias / 100);
        const polynomial = x => {
          let value = 1;
          for (let j = 0; j < order; j++) value = 1 - beta * x * x * value;
          return value;
        };
        return { curves: [{ name: 'Exact response', points: sample(-1, 1, 180, x => 1 / (1 + beta * x * x)) },
          { name: 'Even Taylor truncation', points: sample(-1, 1, 180, polynomial) },
          { name: 'Curvature-fitted rational', points: sample(-1, 1, 180, x => 1 / (1 + fitted * x * x)) }],
          result: 'True pole distance ' + (1 / Math.sqrt(beta)).toFixed(3) + ', fitted pole distance ' +
            (1 / Math.sqrt(fitted)).toFixed(3) + '. The Taylor convergence boundary is |x| = ' + (1 / Math.sqrt(beta)).toFixed(3) + '.' };
      }
    },
    'numerical-methods-variable-projection': {
      controls: [['Change in last observation', -0.6, 0.6, 0.01, 0], ['Rate under inspection (1/s)', 0.1, 1.6, 0.01, 0.69]],
      xLabel: 'Trial rate (1/s)', yLabel: 'Sum of squared residuals',
      calculate([change, inspect]) {
        const y = [5, 3, 2, 1.5 + change];
        const fit = relaxationFit(inspect, y);
        let low = 0.10, high = 1.6;
        for (let i = 0; i < 45; i++) {
          const left = low + (high - low) / 3, right = high - (high - low) / 3;
          if (relaxationFit(left, y).profiled < relaxationFit(right, y).profiled) high = right;
          else low = left;
        }
        return { curves: [{ name: 'Amplitude and background refitted', points: sample(0.1, 1.6, 160, k => relaxationFit(k, y).profiled) },
          { name: 'Amplitude and background fixed', points: sample(0.1, 1.6, 160, k => relaxationFit(k, y).fixed) }],
          result: 'At the inspected rate: amplitude ' + fit.amplitude.toFixed(3) + ', baseline ' + fit.baseline.toFixed(3) +
            ', profiled residual sum ' + fit.profiled.toPrecision(3) + '. Profile minimum on the plot near ' + ((low + high) / 2).toFixed(3) + ' 1/s.' };
      }
    },
    'numerical-methods-event-location': {
      controls: [['Crossing level', 0.05, 0.9, 0.01, 0.5], ['Inspection step (s)', 0.05, 0.75, 0.01, 0.4]],
      xLabel: 'Sample spacing (s)', yLabel: 'log₁₀ absolute crossing-time error (s)',
      calculate([level, step]) {
        const exact = Math.acos(level);
        return { curves: [{ name: 'Linear interpolation', points: sample(0.05, 0.75, 150, h => logError(locateEvent(h, level, false), exact)) },
          { name: 'Cubic Hermite interpolation', points: sample(0.05, 0.75, 150, h => logError(locateEvent(h, level, true), exact)) }],
          result: 'Exact crossing ' + exact.toFixed(6) + ' s. At the selected step: linear ' + locateEvent(step, level, false).toFixed(6) +
            ' s, Hermite ' + locateEvent(step, level, true).toFixed(6) + ' s. Exact endpoints isolate interpolation error.' };
      }
    },
    'numerical-methods-burgers-flux': {
      controls: [['Left state', -1, 1, 0.05, 1], ['Right state', -1, 1, 0.05, -0.5], ['Cells', 40, 200, 10, 100]],
      xLabel: 'Position at t = 0.25', yLabel: 'Conserved state',
      calculate([left, right, count]) {
        const dx = 2 / count, end = 0.25;
        const maxSpeed = Math.max(Math.abs(left), Math.abs(right));
        const steps = Math.max(1, Math.ceil(end * maxSpeed / (0.8 * dx)));
        const dt = end / steps;
        let values = Array.from({ length: count }, (_, i) => i < count / 2 ? left : right);
        for (let n = 0; n < steps; n++) {
          const flux = Array.from({ length: count + 1 }, (_, i) => {
            const atFace = burgersState(i === 0 ? left : values[i - 1], i === count ? right : values[i], 0);
            return atFace * atFace / 2;
          });
          values = values.map((v, i) => v - dt / dx * (flux[i + 1] - flux[i]));
        }
        const points = values.map((v, i) => [-1 + (i + 0.5) * dx, v]);
        const jump = left > right ? 'Shock speed ' + ((left + right) / 2).toFixed(3) : left < right ?
          'Rarefaction edges ' + left.toFixed(2) + ' and ' + right.toFixed(2) : 'Constant data';
        return { curves: [{ name: 'Godunov cell averages', points },
          { name: 'Exact entropy solution', points: sample(-1, 1, 400, x => burgersState(left, right, x / end)) }],
          result: jump + '. Time steps ' + steps + ', maximum CFL ' + (maxSpeed * dt / dx).toFixed(3) +
            ', integral ' + (dx * values.reduce((sum, v) => sum + v, 0)).toFixed(5) + '.' };
      }
    }
  });
})();

(() => {
  function series(name, count, value) {
    return { name, points: Array.from({ length: count }, (_, i) => value(i)) };
  }
  function twoModeIteration(slow, fast, mixing) {
    let state = [0, 0], previous = null;
    const points = [];
    for (let k = 0; k <= 24; k++) {
      const model = [slow * state[0] + 1 - slow, fast * state[1] + 1 - fast];
      const residual = model.map((v, j) => v - state[j]);
      points.push([k, Math.log10(Math.max(1e-14, Math.hypot(...residual)))]);
      let next = model;
      if (mixing && previous) {
        const change = residual.map((v, j) => previous.residual[j] - v);
        const denominator = change[0] ** 2 + change[1] ** 2;
        if (denominator > 1e-28) {
          const beta = -(residual[0] * change[0] + residual[1] * change[1]) / denominator;
          next = model.map((v, j) => v + beta * (previous.model[j] - v));
        }
      }
      previous = { residual, model };
      state = next;
    }
    return points;
  }
  function rootsFromStart(start, damping, deflated) {
    let x = start;
    const points = [[0, x]];
    for (let n = 1; n <= 18; n++) {
      const f = deflated ? (x - 1) * (x + 2) : x * (x - 1) * (x + 2);
      const derivative = deflated ? 2 * x + 1 : 3 * x ** 2 + 2 * x - 2;
      if (Math.abs(derivative) < 1e-14) break;
      x -= damping * f / derivative;
      points.push([n, x]);
    }
    return points;
  }
  function contourCount(radius, angle, count) {
    let sum1 = 0, sum2 = 0;
    for (let j = 0; j < count; j++) {
      const phase = angle - 2 * Math.PI * j / count;
      const x = 1 - radius * Math.cos(phase), y = -radius * Math.sin(phase);
      const magnitude2 = x * x + y * y;
      sum1 += x / magnitude2;
      sum2 -= y / magnitude2;
    }
    return [sum1 / count, sum2 / count];
  }
  function exactDelay(t, strength) {
    let sum = 1, coefficient = 1;
    for (let j = 1; j <= Math.floor(t) + 1; j++) {
      coefficient *= -strength / j;
      sum += coefficient * (t - j + 1) ** j;
    }
    return sum;
  }
  function collocatedValue(x, k, count) {
    const h = 1 / count, z = k * h;
    const r = (1 - z / 2 + z * z / 12) / (1 + z / 2 + z * z / 12);
    const denom = 1 - r ** (2 * count);
    const state = j => [(r ** j - r ** (2 * count - j)) / denom, -k * (r ** j + r ** (2 * count - j)) / denom];
    const i = Math.min(count - 1, Math.floor(x / h)), s = x / h - i;
    const [v0, d0] = state(i), [v1, d1] = state(i + 1);
    return (2 * s ** 3 - 3 * s ** 2 + 1) * v0 + (s ** 3 - 2 * s ** 2 + s) * h * d0 +
      (-2 * s ** 3 + 3 * s ** 2) * v1 + (s ** 3 - s ** 2) * h * d1;
  }

  Object.assign(window.fundamentalStudies ||= {}, {
    'numerical-methods-anderson': {
      controls: [['Slow multiplier', 0.6, 0.99, 0.01, 0.9], ['Second multiplier', -0.7, 0.7, 0.05, 0.2]],
      xLabel: 'Residual evaluation number', yLabel: 'log₁₀ residual norm',
      calculate([slow, fast]) {
        const plain = twoModeIteration(slow, fast, false), mixed = twoModeIteration(slow, fast, true);
        return { curves: [{ name: 'Ordinary fixed point', points: plain }, { name: 'Two-history Anderson', points: mixed }],
          result: 'After six updates: ordinary residual ' + (10 ** plain[6][1]).toPrecision(3) +
            ', mixed residual ' + (10 ** mixed[6][1]).toPrecision(3) + '. Plot floor 10⁻¹⁴.' };
      }
    },
    'numerical-methods-deflated-roots': {
      controls: [['Initial x', 0.05, 0.45, 0.01, 0.2], ['Newton damping', 0.3, 1, 0.05, 1]],
      xLabel: 'Newton iteration', yLabel: 'Candidate x',
      calculate([start, damping]) {
        const plain = rootsFromStart(start, damping, false), deflated = rootsFromStart(start, damping, true);
        const p = plain[plain.length - 1][1], d = deflated[deflated.length - 1][1];
        return { curves: [{ name: 'Original cubic', points: plain }, { name: 'Known zero divided out', points: deflated }],
          result: 'Final candidates ' + p.toPrecision(4) + ' and ' + d.toPrecision(4) + '. Original cubic residual on the deflated branch ' +
            Math.abs(d * (d - 1) * (d + 2)).toExponential(2) + '.' };
      }
    },
    'numerical-methods-contour-count': {
      controls: [['Root radius', 0.2, 0.99, 0.01, 0.9], ['Root angle (degrees)', 0, 90, 1, 0], ['Inspected samples', 4, 96, 1, 8]],
      xLabel: 'Uniform samples on the unit circle', yLabel: 'Estimated complex count',
      calculate([radius, deg, probe]) {
        const angle = deg * Math.PI / 180;
        const counts = Array.from({length: 93}, (_, i) => contourCount(radius, angle, i + 4));
        const read = counts[probe - 4];
        return { curves: [series('Real part', 93, i => [i + 4, counts[i][0]]),
          series('Imaginary part', 93, i => [i + 4, counts[i][1]]), series('Exact count', 93, i => [i + 4, 1])],
          result: 'At ' + probe + ' samples: ' + read[0].toFixed(5) + ' + (' + read[1].toFixed(5) + ')i. True count 1.' };
      }
    },
    'numerical-methods-delay-history': {
      controls: [['Dimensionless feedback aτ', 0.3, 2.3, 0.05, 1], ['Steps per delay', 4, 40, 1, 10]],
      xLabel: 'Time / delay', yLabel: 'State',
      calculate([strength, steps]) {
        const values = [1];
        for (let i = 0; i < 6 * steps; i++) values.push(values[i] - strength / steps * (i - steps < 0 ? 1 : values[i - steps]));
        const maxError = Math.max(...values.map((v, i) => Math.abs(v - exactDelay(i / steps, strength))));
        return { curves: [series('Method-of-steps exact', 241, i => [i / 40, exactDelay(i / 40, strength)]),
          { name: 'Euler with aligned history grid', points: values.map((v, i) => [i / steps, v]) }],
          result: 'Largest grid error ' + maxError.toPrecision(3) + '. First continuum instability at aτ = π/2 ≈ 1.571; current aτ = ' +
            strength.toFixed(2) + '.' };
      }
    },
    'numerical-methods-lobatto-layer': {
      controls: [['Layer parameter κ', 1, 25, 0.5, 10], ['Uniform panels', 2, 32, 1, 6]],
      xLabel: 'Distance from left boundary', yLabel: 'Boundary solution',
      calculate([k, count]) {
        const exact = x => Math.sinh(k * (1 - x)) / Math.sinh(k);
        const curve = series('Cubic Lobatto solution', 241, i => [i / 240, collocatedValue(i / 240, k, count)]);
        const error = Math.max(...curve.points.map(([x, y]) => Math.abs(y - exact(x))));
        return { curves: [series('Exact boundary layer', 241, i => [i / 240, exact(i / 240)]), curve],
          result: 'Sampled maximum field error ' + error.toPrecision(3) + '; shooting endpoint amplification sinh(κ)/κ = ' +
            (Math.sinh(k) / k).toPrecision(4) + '.' };
      }
    },
    'numerical-methods-stiff-forcing': {
      controls: [['log₁₀ of |hλ|', -1, 2.5, 0.05, 2], ['Initial mismatch', 0, 0.02, 0.001, 0.01]],
      xLabel: 'Time (s), h = 0.1 s', yLabel: 'State minus target t²',
      calculate([power, initial]) {
        const z = -(10 ** power), h = 0.1, r = (1 + z / 2) / (1 - z / 2);
        let be = initial, trap = initial, mid = initial;
        const curves = ['Exact decaying mismatch', 'Backward Euler', 'Trapezoid', 'Implicit midpoint'].map(name => ({ name, points: [] }));
        for (let i = 0; i <= 18; i++) {
          [initial * Math.exp(z * i), be, trap, mid].forEach((v, j) => curves[j].points.push([i * h, v]));
          be = (be + h * h) / (1 - z);
          trap *= r;
          mid = r * mid + z * h * h / (4 * (1 - z / 2));
        }
        return { curves, result: 'hλ = ' + z.toPrecision(3) + ', homogeneous midpoint/trapezoid factor ' + r.toFixed(4) +
          ', backward Euler factor ' + (1 / (1 - z)).toFixed(4) + '. Initial mismatch is shown relative to the target.' };
      }
    },
    'numerical-methods-flux-certificate': {
      controls: [['Uniform elements', 2, 30, 1, 2], ['Flux offset from optimum', -0.6, 0.6, 0.02, 0.2]],
      xLabel: 'Position', yLabel: 'Flux',
      calculate([count, offset]) {
        const raw = [];
        for (let j = 0; j < count; j++) {
          const value = 2 * (j + 0.5) / count - 1;
          raw.push([j / count, value], [(j + 1) / count, value]);
        }
        const exact = 1 / (Math.sqrt(3) * count);
        return { curves: [series('Exact flux q', 101, i => [i / 100, 2 * i / 100 - 1]), { name: 'Raw finite-element flux', points: raw },
          series('Equilibrated reconstruction', 101, i => [i / 100, 2 * i / 100 - 1 + offset])],
          result: 'True energy error ' + exact.toFixed(5) + ', guaranteed reconstructed bound ' + Math.hypot(exact, offset).toFixed(5) +
            '. The offset does not change the exact divergence.' };
      }
    },
    'numerical-methods-poisson-alias': {
      controls: [['Base grid samples N', 12, 48, 2, 12], ['Requested input mode m', 1, 12, 1, 5]],
      xLabel: 'Periodic coordinate x', yLabel: 'Potential after mean subtraction',
      calculate([count, requested]) {
        const m = Math.min(requested, count / 2 - 1);
        const trueMode = 2 * m;
        const alias = Math.min(trueMode % count, count - trueMode % count);
        const signal = (k, x) => k ? 0.5 * Math.cos(k * x) / (k * k) : 0;
        const projected = trueMode <= count / 2 ? trueMode : 0;
        const curve = (name, k) => series(name, 301, i => { const x = 2 * Math.PI * i / 300; return [x, signal(k, x)]; });
        return { curves: [curve('Continuum solution', trueMode), curve('Unpadded grid product', alias), curve('Resolved-band projection', projected)],
          result: 'Representable input m = ' + m + ', product mode ' + trueMode + ', coarse alias ' + alias +
            '. Continuum amplitude ' + (0.5 / trueMode ** 2).toFixed(5) + ', coarse amplitude ' + (0.5 / alias ** 2).toFixed(5) + '.' };
      }
    },
    'numerical-methods-wave-boundary': {
      controls: [['Boundary to physical speed ratio', 0.5, 1.5, 0.025, 1], ['Inspected angle (degrees)', 0, 85, 1, 60]],
      xLabel: 'Incidence angle from the normal (degrees)', yLabel: 'Reflected energy fraction',
      calculate([ratio, angle]) {
        const amplitude = (speed, degrees) => {
          const factor = speed * Math.cos(degrees * Math.PI / 180);
          return (factor - 1) / (factor + 1);
        };
        const reference = a => amplitude(1, a) ** 2;
        return { curves: [series('Selected first-order boundary', 171, i => [i / 2, amplitude(ratio, i / 2) ** 2]),
          series('Speed-matched first-order boundary', 171, i => [i / 2, reference(i / 2)])],
          result: 'At ' + angle.toFixed(0) + '° the selected signed amplitude is ' + amplitude(ratio, angle).toFixed(4) +
            ', reflected energy ' + (100 * amplitude(ratio, angle) ** 2).toFixed(2) + '%.' };
      }
    }
  });
})();
