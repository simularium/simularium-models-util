#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

from simularium_models_util.visualization import ActinVisualization


def main():
    parser = argparse.ArgumentParser(
        description="Parses an actin hdf5 (*.h5) trajectory file produced\
         by the ReaDDy software and converts it into the Simularium\
         visualization-data-format with plots"
    )
    parser.add_argument(
        "dir_path",
        help="the file path of the directory\
         containing the trajectories to parse",
    )
    parser.add_argument("box_size", help="width of simulation cube")
    parser.add_argument(
        "total_steps", help="total number of iterations during model run"
    )
    parser.add_argument(
        "periodic_boundary", help="is there a periodic boundary condition?"
    )
    parser.add_argument(
        "plot_bend_twist",
        help="calculate bend/twist plots? otherwise calculate polymerization plots",
    )
    parser.add_argument(
        "actin_number_types",
        help="number of possible actin monomer types. can be either 3 or 5.",
    )
    args = parser.parse_args()
    dir_path = args.dir_path
    for file in os.listdir(dir_path):
        if file.endswith(".h5"):
            file_path = os.path.join(dir_path, file)
            print(f"visualize {file_path}")
            if args.plot_bend_twist:
                plots = ActinVisualization.generate_bend_twist_plots(
                    file_path, float(args.box_size), 10, args.periodic_boundary
                )
            else:

                plots = ActinVisualization.generate_polymerization_plots(
                    int(args.actin_number_types),
                    file_path,
                    float(args.box_size),
                    10,
                    args.periodic_boundary,
                )
            ActinVisualization.visualize_actin(
                file_path,
                float(args.box_size),
                float(args.total_steps),
                plots,
            )


if __name__ == "__main__":
    main()
