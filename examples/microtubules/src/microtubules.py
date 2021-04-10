#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import argparse

from simularium_models_util.microtubules import MicrotubulesSimulation
from simularium_models_util.visualization import MicrotubulesVisualization


def main():
    parser = argparse.ArgumentParser(
        description="Runs and visualizes a ReaDDy microtubules simulation"
    )
    parser.add_argument(
        "params_path", help="the file path of an excel file with parameters")
    parser.add_argument(
        "data_column", help="the column index for the parameter set to use")
    args = parser.parse_args()
    parameters = pandas.read_excel(
        args.params_path, sheet_name="microtubules", usecols=[0, int(args.data_column)])
    parameters.set_index('name', inplace=True)
    parameters.transpose()
    run_name = list(parameters)[0]
    parameters = parameters[run_name]
    parameters["name"] = run_name
    mt_simulation = MicrotubulesSimulation(parameters, True, True)
    mt_simulation.add_random_tubulin_dimers()
    mt_simulation.add_microtubule_seed()
    mt_simulation.simulation.run(
        int(parameters["total_steps"]), parameters["timestep"])
    MicrotubulesVisualization.visualize_microtubules(
        "{}.h5".format(parameters["name"]), parameters["box_size"], [])


if __name__ == '__main__':
    main()
