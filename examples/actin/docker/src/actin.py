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
from simularium_models_util import RepeatedTimer, ReaddyUtil


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
    print("finished args")
    parameters = pandas.read_excel(
        args.params_path, sheet_name="actin", usecols=[0, int(args.data_column)]
    )
    print(f"parameter excel: {parameters}")
    parameters.set_index("name", inplace=True)
    parameters.transpose()
    run_name = list(parameters)[0]
    parameters = parameters[run_name]
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    parameters["name"] = "outputs/" + args.model_name + "_" + str(run_name)
    # actin_simulation = ActinSimulation(parameters, True, False) 
    # actin_simulation.add_obstacles()
    # actin_simulation.add_random_monomers()
    if parameters["orthogonal_seed"]:
        print("Starting with orthogonal seed")
        fiber_data = [
            FiberData(
                28,
                [
                    np.array([-75, 0, 0]),
                    np.array([75, 0, 0]),
                ],
                "Actin-Polymer",
            )
        ] 
        # raise Exception(int(parameters["actin_number_types"]))
        monomers = ActinGenerator.get_monomers(fiber_data, int(parameters["actin_number_types"]), use_uuids=False)
        monomers = ActinGenerator.setup_fixed_monomers(monomers, parameters)
        import ipdb; ipdb.set_trace()
        #"pp monomers" in terminal

        actin_simulation.add_monomers_from_data(monomers)
    print("success orthogonal if loop")
    if parameters["branched_seed"]:
        print("Starting with branched seed")
        actin_simulation.add_monomers_from_data(
            ActinGenerator.get_monomers(ActinTestData.simple_branched_actin_fiber(), int(parameters["actin_number_types"]), use_uuids=False)
        )
    rt = RepeatedTimer(300, report_hardware_usage)  # every 5 min
    try:
        actin_simulation.simulation.run(
            int(parameters["total_steps"]), parameters["internal_timestep"]
        )
        try:
            plots = None
            if parameters["plot_polymerization"]:
                plots = ActinVisualization.generate_polymerization_plots(
                    parameters["name"] + ".h5", parameters["box_size"], 10, parameters["periodic_boundary"], plots
                )
            if parameters["plot_bend_twist"]:
                plots = ActinVisualization.generate_bend_twist_plots(
                    parameters["name"] + ".h5", parameters["box_size"], 10, parameters["periodic_boundary"], plots
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
