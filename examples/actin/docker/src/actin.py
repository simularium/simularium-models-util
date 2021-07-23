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

from simularium_models_util.actin import FiberData, ArpData, ActinSimulation, ActinGenerator
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
    actin_simulation = ActinSimulation(parameters, True, True)
    actin_simulation.add_random_monomers()
    actin_simulation.add_random_linear_fibers()
    if parameters["orthogonal_seed"]:
        orthogonal_seed = [
            FiberData(
                0,
                [
                    np.array([-70, 0, 0]),
                    np.array([10, 0, 0]),
                ],
            )
        ]
        branched_seed = [
            FiberData(
                fiber_id=0,
                points=[
                    np.array([-50, 0, 0]),
                    np.array([50, 0, 0]),
                ],
                nucleated_arps=[
                    ArpData(
                        arp_id=0,
                        position=np.array([-19, 0, 0]),
                        nucleated=True,
                        daughter_fiber=FiberData(
                            fiber_id=1,
                            points=[
                                np.array([-19, 0, 0]),
                                np.array([0.8, 54.5, 0]),
                            ],
                            is_daughter=True,
                            nucleated_arps=[
                                ArpData(
                                    arp_id=1,
                                    position=np.array([-6, 35.7, 0]),
                                    nucleated=True,
                                    daughter_fiber=FiberData(
                                        fiber_id=2,
                                        points=[
                                            np.array([-6, 35.7, 0]),
                                            np.array([14, 35.7, 0]),
                                        ],
                                        is_daughter=True,
                                    ),
                                ),
                            ],
                        ),
                    ),
                    ArpData(
                        arp_id=3,
                        position=np.array([0, 0, 0]),
                        nucleated=True,
                        daughter_fiber=FiberData(
                            fiber_id=3,
                            points=[
                                np.array([0, 0, 0]),
                                np.array([16.4, -45.1, 0]),
                            ],
                            is_daughter=True,
                            nucleated_arps=[
                                ArpData(
                                    arp_id=4,
                                    position=np.array([6.5, -17.9, 0]),
                                    nucleated=True,
                                    daughter_fiber=FiberData(
                                        fiber_id=4,
                                        points=[
                                            np.array([6.5, -17.9, 0]),
                                            np.array([-30.3, -48.7, 0]),
                                        ],
                                        is_daughter=True,
                                        nucleated_arps=[
                                            ArpData(
                                                arp_id=5,
                                                position=np.array([-22.6, -42.3, 0]),
                                                nucleated=True,
                                                daughter_fiber=FiberData(
                                                    fiber_id=5,
                                                    points=[
                                                        np.array([-22.6, -42.3, 0]),
                                                        np.array([-15.8, -61.1, 0]),
                                                    ],
                                                    is_daughter=True,
                                                ),
                                            ),
                                        ],
                                    ),
                                ),
                                ArpData(
                                    arp_id=6,
                                    position=np.array([13, -35.7, 0]),
                                    nucleated=True,
                                    daughter_fiber=FiberData(
                                        fiber_id=6,
                                        points=[
                                            np.array([13, -35.7, 0]),
                                            np.array([33, -35.7, 0]),
                                        ],
                                        is_daughter=True,
                                    ),
                                ),
                            ],
                        ),
                    ),
                    ArpData(
                        arp_id=7,
                        position=np.array([9.5, 0, 0]),
                        nucleated=True,
                        daughter_fiber=FiberData(
                            fiber_id=7,
                            points=[
                                np.array([9.5, 0, 0]),
                                np.array([16.3, 0, 18.8]),
                            ],
                            is_daughter=True,
                        ),
                    ),
                    ArpData(
                        arp_id=8,
                        position=np.array([19, 0, 0]),
                        nucleated=True,
                        daughter_fiber=FiberData(
                            fiber_id=8,
                            points=[
                                np.array([19, 0, 0]),
                                np.array([29.3, 28.2, 0]),
                            ],
                            is_daughter=True,
                            bound_arps=[
                                ArpData(
                                    arp_id=9,
                                    position=np.array([25.9, 18.8, 0]),
                                ),
                            ],
                        ),
                    ),
                ],
                bound_arps=[
                    ArpData(
                        arp_id=2,
                        position=np.array([-10, 0, 0]),
                    ),
                    ArpData(
                        arp_id=10,
                        position=np.array([25, 0, 0]),
                    )
                ],
            )
        ]
        # actin_simulation.add_fibers_from_data(orthogonal_seed)
        actin_simulation.add_monomers_from_data(ActinGenerator.get_monomers(branched_seed))
    rt = RepeatedTimer(300, report_hardware_usage)  # every 5 min
    try:
        actin_simulation.simulation.run(
            int(parameters["total_steps"]), parameters["timestep"]
        )
        # remove checkpoints once succeeded to save space
        rmtree("checkpoints/")
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
