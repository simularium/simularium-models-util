#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import argparse

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import MetaData, UnitData, DisplayData, DISPLAY_TYPE
from ..microtubules import MicrotubulesUtil


class MicrotubulesVisualization:
    """
    visualize a microtubules trajectory in Simularium
    """

    @staticmethod
    def get_display_data_for_all_polymer_types(raw_display_data):
        """
        creates a dictionary mapping particle type for all polymer types to a value
        for a dictionary of types and values

        returns dictionary mapping all types to values
        """
        result = {}
        for particle_type in raw_display_data:
            types = MicrotubulesUtil.get_all_polymer_tubulin_types(particle_type)
            for t in types:
                result[t] = raw_display_data[particle_type]
        return result

    @staticmethod
    def visualize_microtubules(path_to_readdy_h5, box_size, plots):
        """
        visualize a microtubule trajectory in Simularium
        """
        # radii
        tubulin_radius = 2.0
        display_type = DISPLAY_TYPE.SPHERE
        polymer_display_data = {
            "tubulinA#GTP_": DisplayData(
                name="tubulinA#GTP",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinA#GDP_": DisplayData(
                name="tubulinA#GDP",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinB#GTP_": DisplayData(
                name="tubulinB#GTP",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinB#GDP_": DisplayData(
                name="tubulinB#GDP",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinA#GTP_bent_": DisplayData(
                name="tubulinA#GTP_bent",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinA#GDP_bent_": DisplayData(
                name="tubulinA#GDP_bent",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinB#GTP_bent_": DisplayData(
                name="tubulinB#GTP_bent",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinB#GDP_bent_": DisplayData(
                name="tubulinB#GDP_bent",
                radius=tubulin_radius,
                display_type=display_type,
            ),
        }
        display_data = MicrotubulesVisualization.get_display_data_for_all_polymer_types(
            polymer_display_data
        )
        dimer_display_data = {
            "tubulinA#free": DisplayData(
                name="tubulinA#free",
                radius=tubulin_radius,
                display_type=display_type,
            ),
            "tubulinB#free": DisplayData(
                name="tubulinB#free",
                radius=tubulin_radius,
                display_type=display_type,
            ),
        }
        display_data = dict(display_data, **dimer_display_data)
        # types to ignore
        ignore_types = [
            "site#out",
            "site#1",
            "site#1_GTP",
            "site#1_GDP",
            "site#1_detach",
            "site#2",
            "site#2_GTP",
            "site#2_GDP",
            "site#2_detach",
            "site#3",
            "site#4",
            "site#4_GTP",
            "site#4_GDP",
            "site#new",
            "site#remove",
        ]
        # convert
        data = ReaddyData(
            meta_data=MetaData(
                box_size=box_size,
            ),
            timestep=0.1,
            path_to_readdy_h5=path_to_readdy_h5,
            display_data=display_data,
            time_units=UnitData("ns"),
            spatial_units=UnitData("nm"),
            ignore_types=ignore_types,
            plots=plots,
        )
        ReaddyConverter(data).write_JSON(path_to_readdy_h5)


def main():
    parser = argparse.ArgumentParser(
        description="Parses a microtuble hdf5 (*.h5) trajectory file produced\
         by the ReaDDy software and converts it into the Simularium\
         visualization-data-format"
    )
    parser.add_argument(
        "dir_path",
        help="the file path of the directory\
         containing the trajectories to parse",
    )
    parser.add_argument("box_size", help="width of simulation cube")
    args = parser.parse_args()
    dir_path = args.dir_path
    for file in os.listdir(dir_path):
        if file.endswith(".h5"):
            file_path = os.path.join(dir_path, file)
            print(f"visualize {file_path}")
            MicrotubulesVisualization.visualize_microtubules(
                file_path, args.box_size, []
            )


if __name__ == "__main__":
    main()
