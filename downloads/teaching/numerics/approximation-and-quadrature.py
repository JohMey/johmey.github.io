"""Interpolate point temperatures and integrate the resulting linear segments."""

import argparse

import numpy as np
from onesim.numerics.field_transfer import operators


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cells", type=int, default=4, help="Equal source intervals on the unit line")
parser.add_argument("--law", choices=("quadratic", "linear"), default="quadratic")
args = parser.parse_args()
if args.cells < 1:
    parser.error("use at least one source interval")

power = 2 if args.law == "quadratic" else 1
source = np.linspace(0.0, 1.0, args.cells + 1)
target = np.linspace(0.0, 1.0, 2 * args.cells + 1)
samples = 300 + 20 * source**power
mapping = operators.interpolation(
    "Rod temperature", "source nodes", "nodes and midpoints",
    source_coordinates=tuple(source), target_coordinates=tuple(target),
    source_unit="K", target_unit="K",
)
operators.validate(mapping, source=samples).require_accepted()
values = np.asarray(operators.apply(mapping, samples).value)
integral = np.trapezoid(values, x=target)
exact_integral = 300 + 20 / (power + 1)
print(f"Midpoint maximum error: {np.max(np.abs(values - (300 + 20 * target**power))):.9g} K")
print(f"Integrated interpolant: {integral:.9f} K m")
print(f"Signed integral error:  {integral - exact_integral:.9g} K m")
