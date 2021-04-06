#!/usr/bin/env python
# -*- coding: utf-8 -*-

import readdy
import numpy as np
import json
import os
import argparse

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import UnitData


class MicrotubulesVisualization():
    """
    visualize a microtubules trajectory in Simularium
    """

    @staticmethod
    def get_all_polymer_types(particle_type):
        '''
        gets a list of all polymer numbers
        ("type1_1", "type1_2", "type1_3", "type2_1", ... "type3_3")
        for type particle_type

        returns list of types
        '''
        result = []
        for x in range(1,4):
            for y in range(1,4):
                result.append("{}{}_{}".format(particle_type, x, y))
        return result

    @staticmethod
    def get_mapping_for_all_polymer_types(types_mapping):
        '''
        creates a dictionary mapping particle type for all polymer types to a value
        for a dictionary of types and values

        returns dictionary mapping all types to values
        '''
        result = {}
        for particle_type in types_mapping:
            types = get_all_polymer_types(particle_type)
            for t in types:
                result[t] = types_mapping[particle_type]
        return result

    @staticmethod
    def visualize_microtubules(path_to_readdy_h5, box_size, plots):
        '''
        visualize a microtubule trajectory in Simularium
        '''
        # radii
        tubulin_radius = 2.
        radii = {
            "tubulinA#GTP_" : tubulin_radius,
            "tubulinA#GDP_" : tubulin_radius,
            "tubulinB#GTP_" : tubulin_radius,
            "tubulinB#GDP_" : tubulin_radius,
            "tubulinA#GTP_bent_" : tubulin_radius,
            "tubulinA#GDP_bent_" : tubulin_radius,
            "tubulinB#GTP_bent_" : tubulin_radius,
            "tubulinB#GDP_bent_" : tubulin_radius
        }
        radii = get_mapping_for_all_polymer_types(radii)
        radii["tubulinA#free"] = tubulin_radius
        radii["tubulinB#free"] = tubulin_radius
        # type grouping
        type_grouping = {}
        group_types = {
            "tubulinA#GTP",
            "tubulinA#GDP",
            "tubulinB#GTP",
            "tubulinB#GDP",
            "tubulinA#GTP_bent",
            "tubulinA#GDP_bent",
            "tubulinB#GTP_bent",
            "tubulinB#GDP_bent",
        }
        for group_type in group_types:
            type_grouping[group_type] = get_all_polymer_types(group_type + "_")
        # types to ignore
        ignore_types = [
            "site#out", "site#1", "site#1_GTP", "site#1_GDP", "site#1_detach",
            "site#2", "site#2_GTP", "site#2_GDP", "site#2_detach", "site#3",
            "site#4", "site#4_GTP", "site#4_GDP", "site#new", "site#remove"
        ]
        # convert
        data = ReaddyData(
            box_size=np.array([box_size, box_size, box_size]),
            timestep=0.1,
            path_to_readdy_h5=path_to_readdy_h5,
            radii=radii,
            type_grouping=type_grouping,
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
            MicrotubulesVisualization.visualize_microtubules(file_path, args.box_size, [])

if __name__ == '__main__':
    main()
