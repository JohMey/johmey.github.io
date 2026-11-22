"""Simulate a sampled proportional controller using a continuous OneSimulation plant.

In a local OneSimulation environment run:
    python sampled-feedback.py --period 0.7 --steps 8
The controller samples once per hold. Its numerical plant substeps are independent.
"""
import argparse
import json
from math import exp, isfinite

from onesim.physics.systems import ControlSignalFlowProvider, SignalBlock


def simulate(period, steps=8, substeps=40):
    output = 0.0
    rows = []
    for index in range(steps + 1):
        # The unit-gain, unit-time-constant plant receives this value until the next update.
        command = 3.0 * (1.0 - output)
        rows.append({"time": index * period, "output": output, "command": command})
        if index == steps:
            break
        plant = ControlSignalFlowProvider(
            blocks=(
                SignalBlock("held_input", "constant", parameters={"value": command}),
                SignalBlock("plant", "first_order", inputs=("held_input",),
                            parameters={"time_constant": 1.0, "initial_value": output}),
            ),
            final_time=period, time_step=period / substeps, time_integration="classical_rk4",
        )
        run = plant.execute()
        output = float(run.signals["plant"][-1])
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", type=float, default=0.7, help="Controller sample interval in seconds")
    parser.add_argument("--steps", type=int, default=8, help="Number of controller updates")
    parser.add_argument("--substeps", type=int, default=40, help="RK4 plant steps per controller interval")
    parser.add_argument("--sweep", action="store_true", help="Write the lecture's sample interval sweep as JSON")
    args = parser.parse_args()
    if not isfinite(args.period) or args.period <= 0 or min(args.steps, args.substeps) < 1:
        parser.error("choose a finite positive period and positive step counts")
    if args.sweep:
        records = []
        for hundredths in range(15, 96, 5):
            period = hundredths / 100
            rows = simulate(period, args.steps, args.substeps)
            records.append({"period": period, "outputs": [round(row["output"], 10) for row in rows]})
        print(json.dumps(records, indent=2))
        raise SystemExit
    pole = 4 * exp(-args.period) - 3
    print("time (s)    plant output     held command     exact output")
    for k, row in enumerate(simulate(args.period, args.steps, args.substeps)):
        exact = 0.75 * (1 - pole**k)
        print(f"{row['time']:8.3f}  {row['output']:14.8f}  {row['command']:14.8f}  {exact:14.8f}")
