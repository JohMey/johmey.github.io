(() => {
  function profile(a, sample, blank, ratio) {
    return -sample * a - (blank + sample) * Math.log1p(ratio * Math.exp(-a));
  }

  function derivative(x, y, mu) {
    const radial = mu - x * x - y * y;
    return [radial * x - 2 * Math.PI * y, radial * y + 2 * Math.PI * x];
  }

  function oneTurn(radius, mu, steps) {
    let x = radius;
    let y = 0;
    const h = 1 / steps;
    for (let step = 0; step < steps; step++) {
      const k1 = derivative(x, y, mu);
      const k2 = derivative(x + h * k1[0] / 2, y + h * k1[1] / 2, mu);
      const k3 = derivative(x + h * k2[0] / 2, y + h * k2[1] / 2, mu);
      const k4 = derivative(x + h * k3[0], y + h * k3[1], mu);
      x += h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6;
      y += h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6;
    }
    return { radius: Math.hypot(x, y), phase: Math.atan2(y, x) };
  }

  Object.assign(window.fundamentalStudies ||= {}, {
    'semester-algebra-string-spectrum': {
      controls: [['Mesh intervals', 4, 64, 1, 8], ['Consistent-mass fraction', 0, 1, 0.05, 0.5]],
      xLabel: 'Mode number',
      yLabel: 'Frequency error (%)',
      calculate([intervals, alpha]) {
        const ratio = (mode, blend) => {
          const theta = mode * Math.PI / intervals;
          return 2 * Math.sin(theta / 2) / theta / Math.sqrt(1 - blend * (1 - Math.cos(theta)) / 3);
        };
        const curve = (name, blend) => ({
          name, points: Array.from({ length: intervals - 1 }, (_, i) => [i + 1, 100 * (ratio(i + 1, blend) - 1)])
        });
        const mode = Math.min(4, intervals - 1);
        return {
          curves: [curve('Lumped mass', 0), curve('Consistent mass', 1), curve('Selected blend', alpha)],
          result: 'Mode ' + mode + ': exact ' + (50 * mode).toFixed(2) + ' Hz, selected blend ' +
            (50 * mode * ratio(mode, alpha)).toFixed(2) + ' Hz. The two endpoints are fixed.'
        };
      }
    },
    'semester-algebra-photon-profile': {
      controls: [['Blank count', 1, 100, 1, 40], ['Sample count', 0, 100, 1, 10], ['Sample time / blank time', 0.25, 4, 0.25, 1]],
      xLabel: 'Trial optical thickness',
      yLabel: 'Twice profile log-likelihood loss',
      calculate([blank, sample, ratio]) {
        const estimate = sample ? Math.max(0, Math.log(blank * ratio / sample)) : Infinity;
        const best = sample ? profile(estimate, sample, blank, ratio) : 0;
        const right = sample ? Math.max(4, estimate + 2.5) : 8;
        const points = Array.from({ length: 161 }, (_, i) => {
          const a = right * i / 160;
          return [a, Math.max(0, 2 * (best - profile(a, sample, blank, ratio)))];
        });
        const reference = sample > 0 && estimate > 0 ?
          'Interior curvature standard error ' + Math.sqrt(1 / sample + 1 / blank).toFixed(3) + '.' :
          'A local Gaussian error bar is unsuitable at the boundary.';
        const lower = sample ? '' : ' Exact 95% one-sided conditional lower bound ' +
          Math.max(0, Math.log(ratio * 0.05 ** (1 / blank) / (1 - 0.05 ** (1 / blank)))).toFixed(3) + '.';
        return {
          curves: [{ name: 'Conditional Poisson profile', points }],
          result: 'Best optical thickness ' + (sample ? estimate.toFixed(3) : 'tends to infinity') + '. ' + reference + lower
        };
      }
    },
    'semester-algebra-orbit-shooting': {
      controls: [['Amplitude parameter µ', 0.025, 0.4, 0.025, 0.1], ['RK4 steps per revolution', 8, 80, 1, 16]],
      xLabel: 'Initial radius',
      yLabel: 'Return radius minus initial radius',
      calculate([mu, steps]) {
        const multiplier = Math.exp(-2 * mu);
        const rStar = Math.sqrt(mu);
        const exact = [], numerical = [];
        for (let i = 0; i <= 140; i++) {
          const r = (0.35 + 1.25 * i / 140) * rStar;
          const analytic = 1 / Math.sqrt(-Math.expm1(-2 * mu) / mu + multiplier / (r * r));
          exact.push([r, analytic - r]);
          numerical.push([r, oneTurn(r, mu, steps).radius - r]);
        }
        let lo = 0.35 * rStar, hi = 1.6 * rStar;
        for (let n = 0; n < 40; n++) {
          const mid = (lo + hi) / 2;
          if (oneTurn(mid, mu, steps).radius > mid) lo = mid;
          else hi = mid;
        }
        const found = (lo + hi) / 2;
        const phase = oneTurn(found, mu, steps).phase * 180 / Math.PI;
        return {
          curves: [{ name: 'Exact radial return', points: exact }, { name: 'Cartesian RK4 return', points: numerical }],
          result: 'Exact amplitude ' + rStar.toFixed(5) + ', RK4 radial fixed point ' + found.toFixed(5) +
            ', phase defect ' + phase.toExponential(2) + '°. Transverse multiplier ' + multiplier.toFixed(5) + '.'
        };
      }
    }
  });
})();
