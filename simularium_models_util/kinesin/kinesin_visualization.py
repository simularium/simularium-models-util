#!/usr/bin/env python
# -*- coding: utf-8 -*-

import readdy
import numpy as np
import json
import os
import argparse

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import MetaData, UnitData


class KinesinVisualization():
    """
    visualize a kinesin trajectory in Simularium
    """

    @staticmethod
    def get_all_polymer_types(particle_type):
        """
        gets a list of all polymer numbers
        ("type1_1", "type1_2", "type1_3", "type2_1", ... "type3_3")
        for type particle_type

        returns list of types
        """
        result = []
        for x in range(1,4):
            for y in range(1,4):
                result.append("{}{}_{}".format(particle_type, x, y))
        return result

    @staticmethod
    def get_mapping_for_all_polymer_types(types_mapping):
        """
        creates a dictionary mapping particle type for all polymer types to a value
        for a dictionary of types and values

        returns dictionary mapping all types to values
        """
        result = {}
        for particle_type in types_mapping:
            types = KinesinVisualization.get_all_polymer_types(particle_type)
            for t in types:
                result[t] = types_mapping[particle_type]
        return result

    @staticmethod
    def visualize_kinesin(path_to_readdy_h5, box_size, plots):
        """
        visualize a kinesin trajectory in Simularium
        """
        # radii
        tubulin_radius = 2.
        motor_radius = 2.
        hips_radius = 1.
        cargo_radius = 15.
        radii = {
            "tubulinA#" : tubulin_radius,
            "tubulinB#" : tubulin_radius,
            "tubulinB#bound_" : tubulin_radius
        }
        radii = KinesinVisualization.get_mapping_for_all_polymer_types(radii)
        radii["hips"] = hips_radius
        radii["cargo"] = cargo_radius
        radii["motor#ADP"] = motor_radius
        radii["motor#ATP"] = motor_radius
        radii["motor#apo"] = motor_radius
        radii["motor#new"] = motor_radius
        # type grouping
        type_grouping = {}
        group_types = {
            "tubulinA#",
            "tubulinB#",
            "tubulinB#bound_",
        }
        for group_type in group_types:
            type_grouping[group_type[:-1]] = KinesinVisualization.get_all_polymer_types(group_type)
        # convert
        data = ReaddyData(
            meta_data=MetaData(
                box_size=np.array([box_size, box_size, box_size]),
            ),
            timestep=0.05,
            path_to_readdy_h5=path_to_readdy_h5,
            radii=radii,
            type_grouping=type_grouping,
            time_units=UnitData("ns"),
            spatial_units=UnitData("nm"),
            plots=plots,
        )
        ReaddyConverter(data).write_JSON(path_to_readdy_h5)

def main():
    parser = argparse.ArgumentParser(
        description="Parses a kinesin hdf5 (*.h5) trajectory file produced\
         by the ReaDDy software and converts it into the Simularium\
         visualization-data-format"
    )
    parser.add_argument(
        "dir_path", help="the file path of the directory\
         containing the trajectories to parse")
    parser.add_argument(
        "box_size", help="width of simulation cube")
    args = parser.parse_args()
    dir_path = args.dir_path
    for file in os.listdir(dir_path):
        if file.endswith(".h5"):
            file_path = os.path.join(dir_path, file)
            print("visualize {}".format(file_path))
            KinesinVisualization.visualize_kinesin(file_path, args.box_size, [])

if __name__ == '__main__':
    main()
