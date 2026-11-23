"""Reproduce a floating resistor network and two sensor fits with OneSimulation.

Each subcommand writes CSV on standard output. Resistance is in ohms and current
in amperes. The calibration exercises use the dimensionless data in the chapters.
"""
import argparse
import csv
import sys

import numpy as np
from onesim.numerics.linear.matrices.dense_linear_algebra import least_squares_dense
from onesim.physics.systems import Circuit, CircuitCurrentSource, CircuitResistor


def network(conductance, current):
    parts = (
        CircuitResistor(name="12", positive="1", negative="2", resistance=1 / conductance),
        CircuitResistor(name="23", positive="2", negative="3", resistance=1),
        CircuitResistor(name="13", positive="1", negative="3", resistance=1),
        CircuitCurrentSource(name="supply", positive="3", negative="1", current=current),
    )
    reference = Circuit(components=parts, ground="3").solve(times=(0., 1.), max_step=1.)
    v1 = reference.voltages["1"][-1]
    v2 = reference.voltages["2"][-1]
    # Grounding another node fixes a different additive constant in the same circuit.
    other = Circuit(components=parts, ground="2").solve(times=(0., 1.), max_step=1.)
    if not np.isclose(other.voltages["1"][-1], v1 - v2, atol=1e-10):
        raise ValueError("The independent choice of voltage reference changed a voltage drop")
    for offset in np.linspace(-2, 2, 61):
        yield offset, v1 + offset, v2 + offset, offset, v1 - v2


def sensor_line(coordinate, offset):
    # Refit the full augmented data rather than applying the chapter's rank-one formula.
    original = np.column_stack((np.ones(3), [-1., 0., 1.]))
    added = np.vstack((original, [1., coordinate]))
    y = original[:, 1]
    base = least_squares_dense(original, y)
    update = least_squares_dense(added, np.append(y, coordinate + offset))
    for x in np.linspace(-1.5, 6.4, 91):
        yield x, base[0] + base[1]*x, update[0] + update[1]*x


def validation(sensitivity, offset):
    x = np.array([1., 1., 1., 1., sensitivity])
    y = np.array([1.10, .90, 1.05, .95, sensitivity+offset])

    def ridge(weights, readings, penalty):
        # Fit the penalised problem by including the zero target for sqrt(lambda) times the coefficient.
        a = np.append(weights, np.sqrt(penalty))[:, None]
        return least_squares_dense(a, np.append(readings, 0.))[0]

    for log_penalty in np.linspace(-2., 1.3, 101):
        penalty = 10.**log_penalty
        left_out = []
        for i in range(y.size):
            keep = np.arange(y.size) != i
            left_out.append(y[i] - x[i]*ridge(x[keep], y[keep], penalty))
        full_fit = ridge(x, y, penalty)
        effective_parameters = (x @ x) / (x @ x + penalty)
        gcv = np.mean((y - x*full_fit)**2) / (1 - effective_parameters/y.size)**2
        yield log_penalty, np.mean(np.square(left_out)), gcv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    circuit = modes.add_parser("network")
    circuit.add_argument("--conductance", type=float, default=1.)
    circuit.add_argument("--current", type=float, default=1.)
    fit = modes.add_parser("line")
    fit.add_argument("--coordinate", type=float, default=4.)
    fit.add_argument("--offset", type=float, default=1.)
    cv = modes.add_parser("validation")
    cv.add_argument("--sensitivity", type=float, default=3.)
    cv.add_argument("--offset", type=float, default=.8)
    args = parser.parse_args()
    if args.mode == "network":
        if not .1 <= args.conductance <= 4 or not .2 <= args.current <= 2:
            parser.error("the chapter uses 0.1–4 S and 0.2–2 A")
        columns = ("Reference shift (V)", "V1 (V)", "V2 (V)", "V3 (V)", "V1 minus V2 (V)")
        rows = network(args.conductance, args.current)
    elif args.mode == "line":
        if not 1 <= args.coordinate <= 6 or not -2 <= args.offset <= 2:
            parser.error("the chapter uses coordinates 1–6 and offsets from -2 to 2")
        columns = ("x", "Original fitted value", "Updated fitted value")
        rows = sensor_line(args.coordinate, args.offset)
    else:
        if not .25 <= args.sensitivity <= 5 or not -2 <= args.offset <= 2:
            parser.error("the chapter uses sensitivities 0.25–5 and offsets from -2 to 2")
        columns = ("log10 penalty", "Omitted-sensor prediction score", "Generalised cross-validation")
        rows = validation(args.sensitivity, args.offset)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)


if __name__ == "__main__":
    main()
