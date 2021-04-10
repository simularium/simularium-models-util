#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import argparse

from simularium_models_util.actin import ActinSimulation
from simularium_models_util.visualization import ActinVisualization


def main():
    parser = argparse.ArgumentParser(
        description="Runs and visualizes a ReaDDy branched actin simulation"
    )
    parser.add_argument(
        "params_path", help="the file path of an excel file with parameters")
    parser.add_argument(
        "data_column", help="the column index for the parameter set to use")
    args = parser.parse_args()
    parameters = pandas.read_excel(
        args.params_path, sheet_name="actin", usecols=[0, int(args.data_column)])
    parameters.set_index('name', inplace=True)
    parameters.transpose()
    run_name = list(parameters)[0]
    parameters = parameters[run_name]
    parameters["name"] = run_name
    actin_simulation = ActinSimulation(parameters, True, True)
    actin_simulation.add_random_monomers()
    actin_simulation.add_random_linear_fibers()
    actin_simulation.simulation.run(
        int(parameters["total_steps"]), parameters["timestep"])
    ActinVisualization.visualize_actin(
        "{}.h5".format(parameters["name"]), parameters["box_size"], [])


if __name__ == '__main__':
    main()
