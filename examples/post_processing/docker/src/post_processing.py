#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas
import argparse
import psutil
import numpy as np
import boto3

from simularium_models_util.microtubules import MicrotubulesSimulation
from simularium_models_util.visualization import MicrotubulesVisualization
from simularium_models_util import RepeatedTimer, ReaddyUtil


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
        args.params_path,
        sheet_name="post_processing",
        usecols=[0, int(args.data_column)],
        dtype=object,
    )
    parameters.set_index("name", inplace=True)
    parameters.transpose()
    run_name = list(parameters)[0]
    parameters = parameters[run_name]
    # read in box size
    parameters["box_size"] = ReaddyUtil.get_box_size(parameters["box_size"])
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    parameters["name"] = "outputs/" + args.model_name + "_" + run_name

    stride = 10
    h5_path = parameters["name"] + ".h5"

    s3 = boto3.client("s3")
    bucket_name = "readdy-working-bucket"
    s3.download_file(bucket_name, h5_path, h5_path)

    plots = MicrotubulesVisualization.generate_plots(
        h5_path=h5_path,
        box_size=parameters["box_size"],
        stride=stride,
        save_pickle_file=True,
        pickle_file_path=None,
    )

    viz_stepsize = max(int(parameters["total_steps"] / 1000.0), 1)
    scaled_time_step_us = parameters["timestep"] * 1e-3 * viz_stepsize

    MicrotubulesVisualization.visualize_microtubules(
        parameters["name"] + ".h5",
        parameters["box_size"],
        scaled_time_step_us=scaled_time_step_us,
        plots=plots,
    )


if __name__ == "__main__":
    main()
