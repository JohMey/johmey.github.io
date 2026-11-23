(() => {
    function oscillatoryIntegral(panels, cycles) {
        const k = 2 * Math.PI * cycles, h = 1 / panels, d = k * h / 2;
        const amplitude = x => 1 + x * x;
        const moment0 = k === 0 ? h : 2 * Math.sin(d) / k;
        const sineDifference = Math.abs(d) < 0.01 ? d ** 3 * (1 / 3 - d * d / 30 + d ** 4 / 840) : Math.sin(d) - d * Math.cos(d);
        const moment1 = k === 0 ? 0 : 2 * sineDifference / (k * k);
        let trap = 0, filon = 0;
        for (let i = 0; i < panels; i++) {
            const x = i * h, y = (i + 1) * h, mid = (x + y) / 2;
            const a = amplitude(x), b = amplitude(y);
            trap += h * (a * Math.cos(k * x) + b * Math.cos(k * y)) / 2;
            filon += (a + b) * moment0 * Math.cos(k * mid) / 2 - (b - a) * moment1 * Math.sin(k * mid) / h;
        }
        const exact = Math.abs(k) < 0.001 ? 4 / 3 - 4 * k * k / 15 + k ** 4 / 70 : 2 * Math.sin(k) / k + 2 * (Math.cos(k) - Math.sin(k) / k) / (k * k);
        return [exact, trap, filon];
    }

    function firstBounceHeight(t, height, restitution) {
        const g = 9.81, tau = Math.sqrt(2 * height / g);
        if (t < tau) return height - g * t * t / 2;
        const elapsed = t - tau;
        return restitution * g * tau * elapsed - g * elapsed * elapsed / 2;
    }

    function nodalTransport(ratio, nodes, i) {
        if (!Number.isFinite(ratio)) return i === nodes ? 0 : 1;
        if (ratio === 1) return 1 - i / nodes;
        return (1 - ratio ** (i - nodes)) / (1 - ratio ** (-nodes));
    }

    Object.assign(window.fundamentalStudies ||= {}, {
        'semester-discretisation-filon': {
            controls: [['Envelope panels', 2, 48, 1, 8], ['Selected carrier cycles', 0, 16, 0.125, 8]],
            xLabel: 'Carrier cycles across the interval', yLabel: 'Signed integral',
            calculate([panels, selected]) {
                const curves = ['Exact integral', 'Trapezoid on the product', 'Linear Filon rule'].map(name => ({name, points: []}));
                for (let i = 0; i <= 384; i++) {
                    const cycles = i / 24;
                    oscillatoryIntegral(panels, cycles).forEach((value, j) => curves[j].points.push([cycles, value]));
                }
                const [exact, trap, filon] = oscillatoryIntegral(panels, selected);
                return {curves, result: 'At ' + selected.toFixed(3) + ' cycles: exact ' + exact.toFixed(7) + ', sampled-product trapezoid ' + trap.toFixed(7) + ', linear Filon ' + filon.toFixed(7) + '.'};
            }
        },
        'semester-discretisation-saltation': {
            controls: [['Release height (m)', 0.3, 2, 0.05, 1], ['Restitution', 0.4, 0.95, 0.05, 0.8], ['Height perturbation (%)', 0.1, 4, 0.1, 1]],
            xLabel: 'Fixed camera time (s)', yLabel: 'Height sensitivity (m/m)',
            calculate([height, restitution, percent]) {
                const tau = Math.sqrt(2 * height / 9.81), increment = height * percent / 100;
                const end = tau * (1 + 1.7 * restitution), exact = [], numerical = [];
                const tangent = t => t < tau ? 1 : -restitution + (1 + restitution) * (t - tau) / tau;
                for (let i = 0; i <= 300; i++) {
                    const t = end * i / 300;
                    exact.push([t, tangent(t)]);
                    numerical.push([t, (firstBounceHeight(t, height + increment, restitution) - firstBounceHeight(t, height - increment, restitution)) / (2 * increment)]);
                }
                return {curves: [{name: 'Event-aware one-sided tangent', points: exact}, {name: 'Centred trajectory difference', points: numerical}],
                    result: 'The nominal impact is at ' + tau.toFixed(4) + ' s. At the fixed camera time ' + (1.33 * tau).toFixed(4) + ' s the height sensitivity is ' + tangent(1.33 * tau).toFixed(4) + ' m/m. For these parameters both perturbed trajectories stay on or before the first rebound throughout the plot.'};
            }
        },
        'semester-discretisation-fitted-flux': {
            controls: [['Segments', 4, 40, 1, 8], ['Global Péclet number', 1, 64, 1, 24]],
            xLabel: 'Channel position', yLabel: 'Concentration',
            calculate([nodes, globalPe]) {
                const cellPe = globalPe / nodes;
                const exact = x => Math.expm1(-globalPe * (1 - x)) / Math.expm1(-globalPe);
                const centre = (1 + cellPe / 2) / (1 - cellPe / 2), upwind = 1 + cellPe;
                const grid = Array.from({length: nodes + 1}, (_, i) => i);
                const index = nodes - 1;
                return {curves: [{name: 'Analytic profile', points: Array.from({length: 241}, (_, i) => [i / 240, exact(i / 240)])},
                    {name: 'Fitted nodes joined linearly', points: grid.map(i => [i / nodes, exact(i / nodes)])},
                    {name: 'Centred flux', points: grid.map(i => [i / nodes, nodalTransport(centre, nodes, i)])},
                    {name: 'Upwind flux', points: grid.map(i => [i / nodes, nodalTransport(upwind, nodes, i)])}],
                    result: 'Cell Péclet number ' + cellPe.toFixed(2) + '. At the penultimate node, fitted ' + exact(index / nodes).toFixed(5) + ', centred ' + nodalTransport(centre, nodes, index).toFixed(5) + ', upwind ' + nodalTransport(upwind, nodes, index).toFixed(5) + '.'};
            }
        }
    });
})();
