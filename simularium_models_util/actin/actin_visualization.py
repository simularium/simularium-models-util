#!/usr/bin/env python
# -*- coding: utf-8 -*-

import readdy
import numpy as np
import json
import os
import argparse

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import MetaData, UnitData


class ActinVisualization():
    """
    visualize an actin trajectory in Simularium
    """

    @staticmethod
    def visualize_actin(path_to_readdy_h5, box_size, plots):
        """
        visualize an actin trajectory in Simularium
        """
        # radii
        extra_radius = 1.5
        actin_radius = 2. + extra_radius
        arp23_radius = 2. + extra_radius
        cap_radius = 3. + extra_radius
        radii = {
            "arp2" : arp23_radius,
            "arp2#branched" : arp23_radius,
            "arp3" : arp23_radius,
            "arp3#ATP" : arp23_radius,
            "arp3#new" : arp23_radius,
            "arp3#new_ATP" : arp23_radius,
            "cap" : cap_radius,
            "cap#new" : cap_radius,
            "cap#bound" : cap_radius,
            "actin#free" : actin_radius,
            "actin#free_ATP" : actin_radius,
            "actin#new" : actin_radius,
            "actin#new_ATP" : actin_radius,
            "actin#1" : actin_radius,
            "actin#2" : actin_radius,
            "actin#3" : actin_radius,
            "actin#ATP_1" : actin_radius,
            "actin#ATP_2" : actin_radius,
            "actin#ATP_3" : actin_radius,
            "actin#pointed_1" : actin_radius,
            "actin#pointed_2" : actin_radius,
            "actin#pointed_3" : actin_radius,
            "actin#pointed_ATP_1" : actin_radius,
            "actin#pointed_ATP_2" : actin_radius,
            "actin#pointed_ATP_3" : actin_radius,
            "actin#barbed_1" : actin_radius,
            "actin#barbed_2" : actin_radius,
            "actin#barbed_3" : actin_radius,
            "actin#barbed_ATP_1" : actin_radius,
            "actin#barbed_ATP_2" : actin_radius,
            "actin#barbed_ATP_3" : actin_radius,
            "actin#branch_1" : actin_radius,
            "actin#branch_ATP_1" : actin_radius,
            "actin#branch_barbed_1" : actin_radius,
            "actin#branch_barbed_ATP_1" : actin_radius,
        }
        # type grouping
        type_grouping = {
            "arp2" : ["arp2", "arp2#new"],
            "arp2#ATP" : ["arp2#ATP", "arp2#new_ATP"],
            "cap" : ["cap", "cap#new"],
            "actin#free" : ["actin#free", "actin#new"],
            "actin#free_ATP" : ["actin#free_ATP", "actin#new_ATP"],
            "actin" : ["actin#1", "actin#2", "actin#3"],
            "actin#ATP" : ["actin#ATP_1", "actin#ATP_2", "actin#ATP_3"],
            "actin#pointed" : ["actin#pointed_1", "actin#pointed_2", "actin#pointed_3"],
            "actin#pointed_ATP" : ["actin#pointed_ATP_1", "actin#pointed_ATP_2", "actin#pointed_ATP_3"],
            "actin#barbed" : ["actin#barbed_1", "actin#barbed_2", "actin#barbed_3"],
            "actin#barbed_ATP" : ["actin#barbed_ATP_1", "actin#barbed_ATP_2", "actin#barbed_ATP_3"],
            "actin" : ["actin#1", "actin#2", "actin#3"],
            "actin#branch" : ["actin#branch_1"],
            "actin#branch_ATP" : ["actin#branch_ATP_1"],
            "actin#branch_barbed" : ["actin#branch_barbed_1"],
            "actin#branch_barbed_ATP" : ["actin#branch_barbed_ATP_1"],
        }
        # convert
        data = ReaddyData(
            meta_data=MetaData(
                box_size=np.array([box_size, box_size, box_size]),
            ),
            timestep=0.1,
            path_to_readdy_h5=path_to_readdy_h5,
            radii=radii,
            type_grouping=type_grouping,
            time_units=UnitData("ns"),
            spatial_units=UnitData("nm"),
            plots=plots,
        )
        converter = ReaddyConverter(data)
        converter.write_JSON(path_to_readdy_h5)

def main():
    parser = argparse.ArgumentParser(
        description="Parses an actin hdf5 (*.h5) trajectory file produced\
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
            ActinVisualization.visualize_actin(file_path, args.box_size, [])

if __name__ == '__main__':
    main()
