#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas
import argparse
import psutil

from simularium_models_util.microtubules import MicrotubulesSimulation
from simularium_models_util.visualization import MicrotubulesVisualization
from simularium_models_util import RepeatedTimer


def report_memory_usage():
    print(f"RAM percent used: {psutil.virtual_memory()[2]}")


def main():
    parser = argparse.ArgumentParser(
        description="Runs and visualizes a ReaDDy microtubules simulation"
    )
    parser.add_argument(
        "params_path", help="the file path of an excel file with parameters"
    )
    parser.add_argument(
        "data_column", help="the column index for the parameter set to use"
    )
    parser.add_argument(
        "model_name", help="prefix for output file names", nargs="?", default=""
    )
    args = parser.parse_args()
    parameters = pandas.read_excel(
        args.params_path, sheet_name="microtubules", usecols=[0, int(args.data_column)]
    )
    parameters.set_index("name", inplace=True)
    parameters.transpose()
    run_name = list(parameters)[0]
    parameters = parameters[run_name]
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    parameters["name"] = "outputs/" + args.model_name + "_" + run_name
    mt_simulation = MicrotubulesSimulation(parameters, True, True)
    mt_simulation.add_random_tubulin_dimers()
    mt_simulation.add_microtubule_seed()
    rt = RepeatedTimer(600, report_memory_usage)  # every 10 min
    try:
        mt_simulation.simulation.run(
            int(parameters["total_steps"]), parameters["timestep"]
        )
        MicrotubulesVisualization.visualize_microtubules(
            parameters["name"] + ".h5", parameters["box_size"], []
        )
    finally:
        rt.stop()
    sys.exit(0)


if __name__ == "__main__":
    main()
