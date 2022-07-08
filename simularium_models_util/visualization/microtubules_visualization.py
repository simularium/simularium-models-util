#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import MetaData, UnitData, DisplayData, ScatterPlotData, DISPLAY_TYPE
from ..microtubules import (
    MicrotubulesUtil,
    MicrotubulesAnalyzer,
    MICROTUBULES_REACTIONS,
)
from ..common import ReaddyUtil


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
    def get_protofilament_length_plot(monomer_data, times):
        """
        Add a plot of protofilament lengths
        """
        protofilament_list = MicrotubulesAnalyzer.analyze_protofilament_lengths(
            monomer_data
        )
        protofilaments = {}
        for count, protofilament in enumerate(protofilament_list):
            protofilaments["Filament " + str(count + 1)] = protofilament

        return ScatterPlotData(
            title="Length of protofilaments",
            xaxis_title="Time (µs)",
            yaxis_title="Protofilament length (monomers)",
            xtrace=times,
            ytraces=protofilaments,
            render_mode="lines",
        )

    @staticmethod
    def get_avg_microtubule_length_plot(monomer_data, times):
        """
        Add a plot of average microtubule length
        """
        protofilament_list = np.array(
            MicrotubulesAnalyzer.analyze_protofilament_lengths(monomer_data),
            dtype=float,
        )
        mean_lengths = np.nanmean(protofilament_list, axis=0)
        mean_lengths[np.isnan(mean_lengths)] = 0

        return ScatterPlotData(
            title="Average microtubule length",
            xaxis_title="Time (µs)",
            yaxis_title="Average microtubule length (monomers)",
            xtrace=times,
            ytraces={"Average microtubule length": mean_lengths},
            render_mode="lines",
        )

    @staticmethod
    def get_growth_reactions_plot(reactions, times):
        """
        Add a plot of reaction events over time
        for each total growth reaction
        """
        reaction_events = {}
        for total_rxn_name in MICROTUBULES_REACTIONS["Grow"]:
            reaction_counts = ReaddyUtil.analyze_reaction_count_over_time(
                reactions, total_rxn_name
            )
            if reaction_counts is not None:
                reaction_events[total_rxn_name] = reaction_counts
                if "Total Grow" not in reaction_events:
                    tmp = reaction_counts
                else:
                    tmp += reaction_counts
        reaction_events["Total Grow"] = tmp

        return ScatterPlotData(
            title="Growth reactions",
            xaxis_title="Time (µs)",
            yaxis_title="Reaction events",
            xtrace=times,
            ytraces=reaction_events,
            render_mode="lines",
        )

    @staticmethod
    def get_shrink_reactions_plot(reactions, times):
        """
        Add a plot of reaction events over time
        for each total shrink reaction
        """
        reaction_events = {}

        reaction_counts = ReaddyUtil.analyze_reaction_count_over_time(
            reactions, MICROTUBULES_REACTIONS["MT Shrink"][0]
        )
        if reaction_counts is not None:
            reaction_events[MICROTUBULES_REACTIONS["MT Shrink"][0]] = reaction_counts

        return ScatterPlotData(
            title="Shrink reactions",
            xaxis_title="Time (µs)",
            yaxis_title="Reaction events",
            xtrace=times,
            ytraces=reaction_events,
            render_mode="lines",
        )

    @staticmethod
    def get_attach_reactions_plot(reactions, times):
        """
        Add a plot of attachment events over time
        for each total attach reaction
        """
        reaction_events = {}
        for total_rxn_name in MICROTUBULES_REACTIONS["Lateral Attach"]:
            reaction_counts = ReaddyUtil.analyze_reaction_count_over_time(
                reactions, total_rxn_name
            )
            if reaction_counts is not None:
                if "Total Attach" not in reaction_events:
                    tmp = reaction_counts
                else:
                    tmp += reaction_counts
        reaction_events["Total Attach"] = tmp

        return ScatterPlotData(
            title="Attach reactions",
            xaxis_title="Time (µs)",
            yaxis_title="Reaction events",
            xtrace=times,
            ytraces=reaction_events,
            render_mode="lines",
        )

    @staticmethod
    def generate_plots(
        path_to_readdy_h5,
        box_size,
        stride=1,
        periodic_boundary=True,
        save_pickle_file=False,
        pickle_file_path=None,
    ):
        """
        Use an MicrotubulesAnalyzer to generate plots of observables
        """
        (
            monomer_data,
            reactions,
            times,
            _,
        ) = ReaddyUtil.monomer_data_and_reactions_from_file(
            h5_file_path=path_to_readdy_h5,
            stride=stride,
            timestep=0.1,
            reaction_names=MICROTUBULES_REACTIONS,
            save_pickle_file=save_pickle_file,
            pickle_file_path=pickle_file_path,
        )
        return {
            "scatter": [
                MicrotubulesVisualization.get_avg_microtubule_length_plot(
                    monomer_data, times
                ),
                MicrotubulesVisualization.get_protofilament_length_plot(
                    monomer_data, times
                ),
                MicrotubulesVisualization.get_growth_reactions_plot(reactions, times),
                MicrotubulesVisualization.get_shrink_reactions_plot(reactions, times),
                MicrotubulesVisualization.get_attach_reactions_plot(reactions, times),
            ],
        }

    @staticmethod
    def visualize_microtubules(path_to_readdy_h5, box_size, scaled_time_step_us, plots):
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
            "tubulinA#free",
            "tubulinB#free",
        ]
        # convert
        data = ReaddyData(
            meta_data=MetaData(
                box_size=box_size,
            ),
            timestep=scaled_time_step_us,
            path_to_readdy_h5=path_to_readdy_h5,
            display_data=display_data,
            time_units=UnitData("µs"),
            spatial_units=UnitData("nm"),
            ignore_types=ignore_types,
            plots=[],
        )
        try:
            converter = ReaddyConverter(data)
        except Exception as e:
            print(str(e))
        if plots is not None:
            for plot_type in plots:
                for plot in plots[plot_type]:
                    converter.add_plot(plot, plot_type)

        converter.write_JSON(path_to_readdy_h5)


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
