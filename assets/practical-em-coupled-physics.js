(() => {
    const mu0 = 4e-7 * Math.PI;

    function scaledSum(state, delta, factor) {
        return state.map((value, i) => value + factor * delta[i]);
    }

    function advanceCircuit(state, dt, rates) {
        const a = rates(state);
        const b = rates(scaledSum(state, a, dt / 2));
        const c = rates(scaledSum(state, b, dt / 2));
        const d = rates(scaledSum(state, c, dt));
        return state.map((value, i) => value + dt * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) / 6);
    }

    function dampedStep(t, naturalFrequency, ratio) {
        const s = t * naturalFrequency;
        if (Math.abs(ratio - 1) < 1e-7) return 1 - Math.exp(-s) * (1 + s);
        const root = Math.sqrt(Math.abs(1 - ratio * ratio));
        if (ratio < 1) {
            return 1 - Math.exp(-ratio * s) * (Math.cos(root * s) + ratio * Math.sin(root * s) / root);
        }
        return 1 - ((ratio + root) * Math.exp((-ratio + root) * s)
            - (ratio - root) * Math.exp((-ratio - root) * s)) / (2 * root);
    }

    Object.assign(window.fundamentalStudies ||= {}, {
        'em-coupled-physics-galvanometer': {
            controls: [['Total resistance (Ω)', 0.3, 2, 0.01, 1], ['Winding inductance (mH)', 1, 25, 0.5, 5]],
            xLabel: 'Time (s)', yLabel: 'Deflection (mrad)',
            calculate([resistance, inductanceMilli]) {
                const inductance = inductanceMilli / 1000;
                const motorConstant = 0.005, inertia = 2e-6, spring = 8e-4, mechanicalDrag = 1.6e-5;
                const targetCurrent = 0.005, voltage = targetCurrent * resistance;
                const ratio = (mechanicalDrag + motorConstant ** 2 / resistance) / (2 * Math.sqrt(inertia * spring));
                const finalAngle = 1000 * motorConstant * targetCurrent / spring;
                const rates = ([angle, speed, current]) => [speed, (motorConstant * current - spring * angle - mechanicalDrag * speed) / inertia,
                    (voltage - resistance * current - motorConstant * speed) / inductance];
                const full = [], reduced = [];
                let state = [0, 0, 0];
                for (let i = 0; i <= 320; i++) {
                    const t = i * 0.0025;
                    full.push([t, 1000 * state[0]]);
                    reduced.push([t, finalAngle * dampedStep(t, Math.sqrt(spring / inertia), ratio)]);
                    if (i < 320) for (let j = 0; j < 16; j++) state = advanceCircuit(state, 0.0025 / 16, rates);
                }
                const peak = Math.max(...full.map(point => point[1]));
                return {curves: [{name: 'Coupled winding and pointer', points: full}, {name: 'Short electrical time approximation', points: reduced}],
                    result: 'The reduced damping ratio is ' + ratio.toFixed(3) + '. The largest deflection in the first 0.80 s is ' + peak.toFixed(2) + ' mrad. The common static limit is ' + finalAngle.toFixed(2) + ' mrad.'};
            }
        },
        'em-coupled-physics-eddy-brake': {
            controls: [['Travel speed (m/s)', 0.25, 16, 0.25, 2], ['Loop inductance (mH)', 0.05, 0.4, 0.01, 0.15], ['Peak field (T)', 0.05, 0.4, 0.01, 0.2]],
            xLabel: 'Speed (m/s)', yLabel: 'Mean drag (mN)',
            calculate([speed, milliHenry, field]) {
                const resistance = 0.05, period = 0.10;
                const coupling = 2 * 0.05 * field * Math.sin(Math.PI * 0.025 / period);
                const crossover = resistance / (2 * Math.PI / period * milliHenry * 0.001);
                const lowDrag = u => coupling ** 2 * u / (2 * resistance);
                const drag = u => lowDrag(u) / (1 + (u / crossover) ** 2);
                const grid = Array.from({length: 201}, (_, i) => 16 * i / 200);
                const frequency = speed / period;
                const depth = 1000 * Math.sqrt(1 / (Math.PI * frequency * mu0 * 5.8e7));
                return {curves: [{name: 'Inductive loop', points: grid.map(u => [u, 1000 * drag(u)])}, {name: 'Resistive extrapolation', points: grid.map(u => [u, 1000 * lowDrag(u)])}],
                    result: 'At ' + speed.toFixed(2) + ' m/s the drag is ' + (1000 * drag(speed)).toFixed(2) + ' mN and the mean heat is ' + (1000 * speed * drag(speed)).toFixed(2) + ' mW. The copper skin depth at ' + frequency.toFixed(1) + ' Hz is ' + depth.toFixed(2) + ' mm. Peak drag occurs at ' + crossover.toFixed(2) + ' m/s.'};
            }
        },
        'em-coupled-physics-core-loss': {
            controls: [['Physical induction lag (°)', 0, 35, 0.5, 12], ['Frequency (Hz)', 50, 500, 10, 200], ['Induction channel delay (µs)', -50, 50, 1, 0]],
            xLabel: 'H (A/m)', yLabel: 'B (T)',
            calculate([lag, frequency, delay]) {
                const actual = lag * Math.PI / 180;
                const measured = actual + 2 * Math.PI * frequency * delay * 1e-6;
                const angles = Array.from({length: 241}, (_, i) => i * Math.PI / 120);
                const loop = phase => angles.map(t => [120 * Math.cos(t), 0.2 * Math.cos(t - phase)]);
                const area = phase => Math.PI * 120 * 0.2 * Math.sin(phase);
                return {curves: [{name: 'Recorded loop', points: loop(measured)}, {name: 'Reversible reference', points: loop(0)}],
                    result: 'True loop area ' + area(actual).toFixed(2) + ' J/m³ per cycle. The recorder indicates ' + area(measured).toFixed(2) + ' J/m³. Physical heating of the 40 cm³ specimen is ' + (frequency * 0.00004 * area(actual)).toFixed(3) + ' W.'};
            }
        },
        'em-coupled-physics-diamagnetic-trap': {
            controls: [['Axial B curvature (T/m²)', -10000, 10000, 100, 1000], ['Equilibrium field (T)', 0.5, 1.2, 0.01, 0.8]],
            xLabel: 'Small displacement (mm)', yLabel: 'Potential per mass (µJ/kg)',
            calculate([curvature, field]) {
                const density = 2200, magnitude = 4e-4;
                const slope = -mu0 * density * 9.81 / (magnitude * field);
                const factor = magnitude / (2 * mu0 * density);
                const vertical = factor * 2 * (slope ** 2 + field * curvature);
                const transverse = factor * (slope ** 2 / 2 - field * curvature);
                const grid = Array.from({length: 161}, (_, i) => (i - 80) / 160);
                const potential = omega2 => grid.map(x => [x, 0.5 * omega2 * x ** 2]);
                const describe = w2 => Math.abs(w2) < 1e-10 ? 'marginal quadratic response' : w2 > 0 ? (Math.sqrt(w2) / (2 * Math.PI)).toFixed(2) + ' Hz oscillation' : Math.sqrt(-w2).toFixed(2) + ' s⁻¹ growth rate';
                return {curves: [{name: 'Vertical', points: potential(vertical)}, {name: 'Transverse', points: potential(transverse)}],
                    result: 'Force balance fixes B′ at ' + slope.toFixed(2) + ' T/m. Vertical motion: ' + describe(vertical) + '. Transverse motion: ' + describe(transverse) + '.'};
            }
        }
    });
})();
