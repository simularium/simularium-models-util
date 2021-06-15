#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas
import argparse
import psutil

from simularium_models_util.actin import ActinSimulation
from simularium_models_util.visualization import ActinVisualization
from simularium_models_util import RepeatedTimer


def report_hardware_usage():
    avg_load = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
    print(
        f"AVG load: {avg_load[0]} last min, {avg_load[1]} last 5 min, {avg_load[2]} last 15 min\n"
        f"RAM % used: {psutil.virtual_memory()[2]}\n"
        f"CPU % used: {psutil.cpu_percent()}\n"
        f"Disk % used: {psutil.disk_usage('/').percent}\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Runs and visualizes a ReaDDy branched actin simulation"
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
        args.params_path, sheet_name="actin", usecols=[0, int(args.data_column)]
    )
    parameters.set_index("name", inplace=True)
    parameters.transpose()
    run_name = list(parameters)[0]
    parameters = parameters[run_name]
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    parameters["name"] = "outputs/" + args.model_name + "_" + run_name
    actin_simulation = ActinSimulation(parameters, True, True)
    actin_simulation.add_random_monomers()
    actin_simulation.add_random_linear_fibers()
    rt = RepeatedTimer(300, report_hardware_usage)  # every 5 min
    try:
        actin_simulation.simulation.run(
            int(parameters["total_steps"]), parameters["timestep"]
        )
        try:
            plots = ActinVisualization.generate_plots(
                parameters["name"] + ".h5", parameters["box_size"], 10
            )
            ActinVisualization.visualize_actin(
                parameters["name"] + ".h5",
                parameters["box_size"],
                parameters["total_steps"],
                plots,
            )
        except Exception as e:
            print("Failed viz!!!!!!!!!!\n" + str(type(e)) + " " + str(e))
            report_hardware_usage()
            sys.exit(88888888)
    finally:
        rt.stop()
    sys.exit(0)


if __name__ == "__main__":
    main()
