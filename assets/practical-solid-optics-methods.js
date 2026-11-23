(() => {
    function grid(low, high, intervals) {
        return Array.from({length: intervals + 1}, (_, i) => low + (high - low) * i / intervals);
    }
    function sinc(x) {
        return Math.abs(x) < 1e-7 ? 1 - x * x / 6 : Math.sin(x) / x;
    }
    function thermalSlope(energy, thermalEnergy) {
        const exp = Math.exp(-Math.abs(energy / thermalEnergy));
        return exp / (thermalEnergy * (1 + exp) ** 2);
    }
    function conductance(bias, gap, thermalEnergy) {
        const limit = Math.max(gap, Math.abs(bias)) + 32 * thermalEnergy;
        const intervals = 900, step = limit / intervals;
        let integral = 0;
        for (let i = 0; i <= intervals; i++) {
            const xi = i * step, energy = Math.hypot(xi, gap);
            const weight = (i === 0 || i === intervals) ? 0.5 : 1;
            integral += weight * (thermalSlope(energy - bias, thermalEnergy) + thermalSlope(-energy - bias, thermalEnergy));
        }
        return step * integral;
    }
    Object.assign(window.fundamentalStudies ||= {}, {
        'solid-optics-methods-bragg-order': {
            controls: [['Chemical order S', 0, 1, 0.05, 0.6], ['Vacuum wavelength (nm)', 0.07, 0.28, 0.001, 0.154]],
            xLabel: 'Scattering angle 2θ (°)', yLabel: 'Corrected intensity relative to even peak',
            calculate([order, wavelength]) {
                const spacings = [1, 2, 3, 4, 5, 6];
                const peaks = spacings.map(n => {
                    const sine = wavelength * Math.sqrt(n) / (2 * 0.4);
                    return {n, angle: sine <= 1 ? 360 / Math.PI * Math.asin(sine) : Infinity};
                });
                const angles = grid(0, 135, 600);
                const line = (parity, strength) => angles.map(x => [x, peaks.filter(p => p.n % 2 === parity && Number.isFinite(p.angle))
                    .reduce((s, p) => s + strength * Math.exp(-0.5 * ((x - p.angle) / 0.8) ** 2), 0)]);
                return {curves: [{name: 'Even sum (selected families)', points: line(0, 1)}, {name: 'Odd sum (selected families)', points: line(1, order ** 2 / 9)}],
                    result: 'The corrected odd/even line ratio is ' + (order ** 2 / 9).toFixed(4) + '. The (100) and (110) geometric positions are ' + peaks[0].angle.toFixed(2) + '° and ' + peaks[1].angle.toFixed(2) + '°. Multiplicity and instrument response are excluded.'};
            }
        },
        'solid-optics-methods-periodic-barriers': {
            controls: [['Barrier strength P', 0, 5, 0.05, 1], ['Selected ka', 0.05, 12.5, 0.05, 3.4]],
            xLabel: 'Energy parameter ka', yLabel: 'Cell half-trace',
            calculate([barrier, selected]) {
                const halfTrace = x => Math.cos(x) + barrier * sinc(x);
                const F = halfTrace(selected), xs = grid(0, 12.6, 400);
                const allowed = Math.abs(F) <= 1 + 1e-12;
                const conclusion = allowed ? 'Allowed Bloch phase qa = ' + Math.acos(Math.max(-1, Math.min(1, F))).toFixed(3) + ' rad.'
                    : 'The decaying Bloch amplitude falls by 1/e in ' + (1 / Math.acosh(Math.abs(F))).toFixed(2) + ' cells.';
                return {curves: [{name: 'Half-trace', points: xs.map(x => [x, halfTrace(x)])}, {name: 'Upper edge', points: [[0, 1], [12.6, 1]]},
                    {name: 'Lower edge', points: [[0, -1], [12.6, -1]]}], result: 'Half-trace at the selected energy is ' + F.toFixed(4) + '. ' + conclusion};
            }
        },
        'solid-optics-methods-london-slab': {
            controls: [['Film thickness (nm)', 20, 1200, 10, 400], ['London length (nm)', 40, 300, 5, 100]],
            xLabel: 'Distance from centre (nm)', yLabel: 'Induction relative to the applied field',
            calculate([depth, london]) {
                const half = depth / (2 * london), xs = grid(-depth / 2, depth / 2, 240);
                return {curves: [{name: 'Local induction', points: xs.map(x => [x, Math.cosh(x / london) / Math.cosh(half)])}, {name: 'Cross-section mean', points: [[-depth / 2, Math.tanh(half) / half], [depth / 2, Math.tanh(half) / half]]}],
                    result: 'At the centre B/Ba = ' + (1 / Math.cosh(half)).toFixed(4) + ', whereas the cross-section mean is ' + (Math.tanh(half) / half).toFixed(4) + '. These apply to a parallel field in a vortex-free slab.'};
            }
        },
        'solid-optics-methods-gap-spectrum': {
            controls: [['Gap Δ (meV)', 0, 2, 0.05, 1.5], ['Electron temperature (K)', 0.3, 5, 0.1, 2]],
            xLabel: 'Bias (mV)', yLabel: 'Normalised conductance',
            calculate([gap, temperature]) {
                const kT = 0.0861733 * temperature, xs = grid(-4, 4, 200);
                const positive = xs.slice(100).map(x => conductance(x, gap, kT));
                const points = xs.map((x, i) => [x, positive[Math.abs(i - 100)]]);
                return {curves: [{name: 'Thermally broadened tunnelling', points}, {name: 'Normal-state reference', points: [[-4, 1], [4, 1]]}],
                    result: 'Thermal kernel FWHM ' + (3.5255 * kT).toFixed(3) + ' meV. Zero-bias conductance is ' + positive[0].toFixed(4) + ' of the normal value. Gap and probe temperature vary independently.'};
            }
        },
        'solid-optics-methods-abbe-transfer': {
            controls: [['Numerical aperture', 0.2, 1.3, 0.01, 0.8], ['Vacuum wavelength (nm)', 400, 700, 5, 550], ['Magnification', 10, 80, 1, 40]],
            xLabel: 'Specimen spatial frequency (cycles/µm)', yLabel: 'Normalised transfer',
            calculate([na, wavelengthNm, magnification]) {
                const coherent = na / (wavelengthNm * 0.001), incoherent = 2 * coherent, sampling = magnification / 13;
                const xs = grid(0, Math.max(incoherent, sampling) * 1.1, 240);
                const transfer = f => {
                    const u = f / incoherent;
                    return u >= 1 ? 0 : 2 / Math.PI * (Math.acos(u) - u * Math.sqrt(1 - u * u));
                };
                return {curves: [{name: 'Coherent field pupil', points: [[0, 1], [coherent, 1], [coherent, 0], [xs.at(-1), 0]]},
                    {name: 'Incoherent optical MTF', points: xs.map(x => [x, transfer(x)])},
                    {name: 'Incoherent with pixel integration', points: xs.filter(x => x <= sampling).map(x => [x, transfer(x) * sinc(Math.PI * x * 6.5 / magnification)])}],
                    result: 'Coherent field cutoff ' + coherent.toFixed(2) + ', incoherent optical cutoff ' + incoherent.toFixed(2) + ', camera Nyquist ' + sampling.toFixed(2) + ' cycles/µm. The sampled curve ends at Nyquist.'};
            }
        },
        'solid-optics-methods-pulse-tilt': {
            controls: [['Input spectral phase (fs²)', -800, 800, 20, 400], ['Added dispersion (fs²)', -800, 800, 20, -300], ['Arrival slope (fs/mm)', 0, 120, 2, 40]],
            xLabel: 'Delay (fs)', yLabel: 'Intensity / transform-limited peak',
            calculate([initial, added, slope]) {
                const factor = 2 * Math.sqrt(Math.log(2)), tau0 = 30 / factor;
                const local = tau0 * Math.hypot(1, (initial + added) / tau0 ** 2), full = Math.hypot(local, slope * 0.5);
                const times = grid(-3 * full, 3 * full, 220);
                const trace = tau => times.map(t => [t, tau0 / tau * Math.exp(-((t / tau) ** 2))]);
                return {curves: [{name: 'Locally measured pulse', points: trace(local)}, {name: 'Collected across beam', points: trace(full)}],
                    result: 'Local intensity FWHM ' + (factor * local).toFixed(2) + ' fs. Full-beam intensity FWHM ' + (factor * full).toFixed(2) + ' fs. Residual quadratic phase ' + (initial + added).toFixed(0) + ' fs².'};
            }
        },
        'solid-optics-methods-coincidences': {
            controls: [['Recovery time (ns)', 0.5, 8, 0.1, 2], ['Bin width (ns)', 0.2, 10, 0.2, 4], ['Signal fraction in each arm', 0.1, 1, 0.05, 0.8]],
            xLabel: 'Detection delay (ns)', yLabel: 'Normalised coincidences',
            calculate([tau, width, fraction]) {
                const times = grid(-5 * tau - width / 2, 5 * tau + width / 2, 240);
                const primitive = x => -Math.sign(x) * tau * Math.expm1(-Math.abs(x) / tau);
                const binned = x => 1 - fraction ** 2 * (primitive(x + width / 2) - primitive(x - width / 2)) / width;
                return {curves: [{name: 'Intrinsic two-state recovery', points: times.map(x => [x, 1 - Math.exp(-Math.abs(x) / tau)])}, {name: 'Background + binning', points: times.map(x => [x, binned(x)])}, {name: 'Independent baseline', points: [[times[0], 1], [times.at(-1), 1]]}],
                    result: 'The raw zero-bin correlation is ' + binned(0).toFixed(3) + '. At 40 and 30 kcounts/s for 100 s, the expected zero-bin count is ' + (120 * width * binned(0)).toFixed(1) + ' against an independent baseline of ' + (120 * width).toFixed(1) + '.'};
            }
        },
        'solid-optics-methods-time-lens': {
            controls: [['First dispersion (ps²)', 50, 200, 10, 100], ['Second dispersion (ps²)', -400, 400, 25, 400], ['Lens K (10⁻³ ps⁻²)', -20, 40, 0.5, 12.5]],
            xLabel: 'Input feature time (ps)', yLabel: 'Output arrival time (ps)',
            calculate([D1, D2, lensMilli]) {
                const K = lensMilli / 1000, A = 1 - K * D2, B = D1 + D2 - K * D1 * D2;
                const offsets = [-0.15, 0, 0.15], xs = grid(-4, 4, 60);
                const response = Math.abs(B) < 1e-8 ? 'The image is focused at M = ' + A.toFixed(2) + '.' : 'Two components separated by 0.30 rad/ps arrive ' + (0.30 * Math.abs(B)).toFixed(2) + ' ps apart for the same input time.';
                return {curves: offsets.map(offset => ({name: 'Offset ' + offset.toFixed(2) + ' rad/ps', points: xs.map(x => [x, A * x + B * offset])})),
                    result: 'Temporal map coefficients A = ' + A.toFixed(2) + ', B = ' + B.toFixed(2) + ' ps². ' + response};
            }
        }
    });
})();
