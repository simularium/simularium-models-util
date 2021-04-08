#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import argparse

from simularium_models_util.actin import (
    ActinSimulation,
    ActinVisualization,
)


def main():
    parser = argparse.ArgumentParser(
        description="Runs and visualizes a ReaDDy branched actin simulation"
    )
    parser.add_argument(
        "params_path", help="the file path of an excel file with parameters")
    args = parser.parse_args()
    parameters = pandas.read_excel(args.params_path, sheet_name="actin", usecols=[0, 1])
    parameters.set_index('name', inplace=True)
    parameters.transpose()
    run_name = list(parameters)[0]
    parameters = parameters[run_name]
    parameters["name"] = run_name
    actin_simulation = ActinSimulation(parameters, True, True)
    actin_simulation.add_random_monomers()
    actin_simulation.add_random_linear_fibers()
    actin_simulation.simulation.run(int(parameters["total_steps"]), parameters["timestep"])
    ActinVisualization.visualize_actin(
        "{}.h5".format(parameters["name"]), parameters["box_size"], [])


if __name__ == '__main__':
    main()
