#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from shutil import rmtree
import uuid

import numpy as np
import pandas
import psutil

from simularium_models_util.actin import (
    FiberData,
    ArpData,
    ActinSimulation,
    ActinGenerator,
    ActinTestData,
)
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
    parameters["name"] = "outputs/" + args.model_name + "_" + str(run_name)
    displacements = {
        0 : np.array([5e-4, 0., 0.]),
        1 : np.array([5e-4, 0., 0.]),
        2 : np.array([5e-4, 0., 0.]),
    }
    actin_simulation = ActinSimulation(parameters, True, False, displacements)
    actin_simulation.add_obstacles()
    actin_simulation.add_random_monomers()
    if "orthogonal_seed" in parameters and parameters["orthogonal_seed"]:
        print("ortho")
        fiber_data = [
            FiberData(
                28,
                [
                    np.array([-70, 0, 0]),
                    np.array([70, 0, 0]),
                ],
                "Actin-Polymer",
            )
        ]
        monomers = ActinGenerator.get_monomers(fiber_data, use_uuids=False)
        monomers["particles"][0]["type_name"] = "actin#pointed_fixed_ATP_1"
        monomers["particles"][1]["type_name"] = "actin#fixed_ATP_2"
        monomers["particles"][2]["type_name"] = "actin#mid_fixed_ATP_3"
        monomers["particles"][46]["type_name"] = "actin#mid_fixed_ATP_2"
        monomers["particles"][47]["type_name"] = "actin#mid_fixed_ATP_3"
        monomers["particles"][48]["type_name"] = "actin#fixed_barbed_ATP_1"
        actin_simulation.add_monomers_from_data(monomers)
    if "branched_seed" in parameters and parameters["branched_seed"]:
        print("branched")
        actin_simulation.add_monomers_from_data(
            ActinGenerator.get_monomers(ActinTestData.simple_branched_actin_fiber(), use_uuids=False)
        )
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
