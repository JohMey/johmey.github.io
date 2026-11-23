"""Reproduce the oscillator pair, double well, freely rotating body and three-colour chain.

Each command writes numeric CSV to stdout. The eigenproblems use OneSimulation's
contour solver. The rotor uses its classical four-stage Runge–Kutta integrator.
"""
import argparse
import csv
import math
import sys

import numpy as np
from onesim.numerics.linear.kernels.algorithms.eigen.contour_eigen import BeynContourEigenSolver, elliptic_contour_quadrature
from onesim.numerics.time import ClassicalRungeKutta4TimeIntegrator


def real_spectrum(matrix):
    """Enclose the real modes of a symmetric matrix using a row-sum norm bound."""
    size = matrix.shape[0]
    scale = float(np.max(np.sum(np.abs(matrix), axis=1)))
    identity = np.eye(size)
    contour = elliptic_contour_quadrature(0.0, (1.6, 1.6), 48)
    solver = BeynContourEigenSolver(contour, probe=identity, residual_tolerance=1e-9)
    modes = solver(lambda z: matrix / scale - z * identity)
    if len(modes.values) != size or np.max(np.abs(modes.values.imag)) > 1e-8:
        raise ValueError("The chosen contour did not resolve the real spectrum")
    return np.sort(scale * modes.values.real)


def paired_oscillators(coupling, separation):
    weight = coupling / separation**3
    stiffness = np.array([[1.0, weight], [weight, 1.0]])
    slow, fast = np.sqrt(real_spectrum(stiffness))
    return separation, slow, fast, (slow + fast) / 2 - 1


def double_well(barrier, hopping):
    matrix = np.diag([0.0, barrier, barrier, 0.0])
    for i in range(3):
        matrix[i, i+1] = matrix[i+1, i] = -hopping
    levels = real_spectrum(matrix)
    return (*levels, levels[1] - levels[0])


def free_rotation(spin, seed):
    moments = (0.030, 0.040, 0.050)

    def torque_balance(velocity, time):
        x, y, z = velocity
        i, j, k = moments
        return (j-k)*y*z/i, (k-i)*z*x/j, (i-j)*x*y/k

    method = ClassicalRungeKutta4TimeIntegrator()
    velocity = (seed, spin, 0.0)
    h = 0.002
    for step in range(601):
        if step % 5 == 0:
            energy = sum(i*w*w/2 for i, w in zip(moments, velocity))
            momentum = math.sqrt(sum((i*w)**2 for i, w in zip(moments, velocity)))
            yield step*h, *velocity, energy, momentum
        if step < 600:
            velocity = method.advance(velocity, torque_balance, h, time=step*h)


def colour_chain(temperature, bias):
    # Energy per nearest-neighbour bond is -J δ_ab; colour zero gains an additional field energy -H.
    delta = np.eye(3)
    field = np.array([bias, 0., 0.])
    transfer = np.exp((delta + (field[:, None] + field[None, :]) / 2) / temperature)
    small, second, first = real_spectrum(transfer)
    return temperature, first, second, small, 1 / math.log(first / second), -temperature * math.log(first)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="model", required=True)
    pair = modes.add_parser("pair")
    pair.add_argument("--coupling", type=float, default=0.4)
    pair.add_argument("--separation", type=float, default=1.4)
    wells = modes.add_parser("wells")
    wells.add_argument("--barrier", type=float, default=4.0)
    wells.add_argument("--hopping", type=float, default=1.0)
    rotor = modes.add_parser("rotor")
    rotor.add_argument("--spin", type=float, default=12)
    rotor.add_argument("--seed", type=float, default=0.1)
    chain = modes.add_parser("chain")
    chain.add_argument("--temperature", type=float, default=1.0)
    chain.add_argument("--bias", type=float, default=0.15)
    args = parser.parse_args()
    out = csv.writer(sys.stdout, lineterminator="\n")
    if args.model == "pair":
        if not .05 <= args.coupling <= .9 or not 1 <= args.separation <= 2.5:
            parser.error("choose coupling between .05 and .9 and separation between 1 and 2.5")
        out.writerow(("R/R0", "Lower frequency / omega0", "Upper frequency / omega0", "Shift / hbar omega0"))
        out.writerow(paired_oscillators(args.coupling, args.separation))
    elif args.model == "wells":
        if not 2 <= args.barrier <= 12 or not .5 <= args.hopping <= 1.5:
            parser.error("choose barrier between 2 and 12 meV and hopping between .5 and 1.5 meV")
        out.writerow(("E0 (meV)", "E1 (meV)", "E2 (meV)", "E3 (meV)", "Low splitting (meV)"))
        out.writerow(double_well(args.barrier, args.hopping))
    elif args.model == "rotor":
        if not 4 <= args.spin <= 16 or not .02 <= args.seed <= .5:
            parser.error("choose spin between 4 and 16 rad/s and transverse seed between .02 and .5 rad/s")
        out.writerow(("Time (s)", "omega1 (rad/s)", "omega2 (rad/s)", "omega3 (rad/s)", "Energy (J)", "Momentum (kg m2/s)"))
        out.writerows(free_rotation(args.spin, args.seed))
    else:
        if not .4 <= args.temperature <= 3 or not -.5 <= args.bias <= .5:
            parser.error("choose reduced temperature between .4 and 3 and field between -.5 and .5")
        out.writerow(("kBT/J", "Leading eigenvalue", "Second eigenvalue", "Third eigenvalue", "Correlation length (sites)", "Free energy / J per site"))
        out.writerow(colour_chain(args.temperature, args.bias))


if __name__ == "__main__":
    main()
