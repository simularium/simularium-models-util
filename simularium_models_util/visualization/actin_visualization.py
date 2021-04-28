#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import MetaData, UnitData, ScatterPlotData
from simulariumio.filters import MultiplyTimeFilter
from ..actin import ActinAnalyzer

TIMESTEP = 0.1 #ns


class ActinVisualization:
    """
    visualize an actin trajectory in Simularium
    """

    @staticmethod
    def get_bound_actin_plot(analyzer):
        """
        Add a plot of percent actin in filaments
        """
        return ScatterPlotData(
            title="Filamentous actin",
            xaxis_title="Time (µs)",
            yaxis_title="Bound actin (%)",
            xtrace=analyzer.times,
            ytraces={
                "Bound actin": 100.0
                * analyzer.analyze_ratio_of_filamentous_to_total_actin(),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_ATP_actin_plot(analyzer):
        """
        Add a plot of percent actin with ATP bound (vs ADP)
        """
        return ScatterPlotData(
            title="Actin nucleotide state",
            xaxis_title="Time (µs)",
            yaxis_title="ATP actin (%)",
            xtrace=analyzer.times,
            ytraces={
                "ATP actin": 100.0
                * analyzer.analyze_ratio_of_ATP_actin_to_total_actin(),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_daughter_actin_plot(analyzer):
        """
        Add a plot of percent filamentous actin in daughter filaments
        """
        return ScatterPlotData(
            title="Filamentous actin in branches",
            xaxis_title="Time (µs)",
            yaxis_title="Daughter actin (%)",
            xtrace=analyzer.times,
            ytraces={
                "Daughter actin": 100.0
                * analyzer.analyze_ratio_of_daughter_filament_actin_to_total_filamentous_actin(),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_avg_mother_length_plot(analyzer):
        """
        Add a plot of average mother filament length
        """
        return ScatterPlotData(
            title="Average length of mother filaments",
            xaxis_title="Time (µs)",
            yaxis_title="Average length (nm)",
            xtrace=analyzer.times,
            ytraces={
                "Average length": ActinAnalyzer.analyze_average_over_time(
                    analyzer.analyze_mother_filament_lengths()),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_avg_daughter_length_plot(analyzer):
        """
        Add a plot of average daughter filament length
        """
        return ScatterPlotData(
            title="Average length of daughter filaments",
            xaxis_title="Time (µs)",
            yaxis_title="Average length (nm)",
            xtrace=analyzer.times,
            ytraces={
                "Average length": ActinAnalyzer.analyze_average_over_time(
                    analyzer.analyze_daughter_filament_lengths()),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_bound_arp_plot(analyzer):
        """
        Add a plot of percent arp in filaments
        """
        return ScatterPlotData(
            title="Bound arp2/3 complexes",
            xaxis_title="Time (µs)",
            yaxis_title="Bound arp2/3 (%)",
            xtrace=analyzer.times,
            ytraces={
                "Bound arp2/3": 100.0
                * analyzer.analyze_ratio_of_bound_to_total_arp23(),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_capped_ends_plot(analyzer):
        """
        Add a plot of percent barbed ends that are capped
        """
        return ScatterPlotData(
            title="Capped barbed ends",
            xaxis_title="Time (µs)",
            yaxis_title="Capped ends (%)",
            xtrace=analyzer.times,
            ytraces={
                "Capped ends": 100.0
                * analyzer.analyze_ratio_of_capped_ends_to_total_ends(),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_branch_angle_plot(analyzer):
        """
        Add a plot of branch angle mean and std dev
        """
        angles = analyzer.analyze_branch_angles()
        mean = ActinAnalyzer.analyze_average_over_time(angles)
        stddev = ActinAnalyzer.analyze_stddev_over_time(angles)
        return ScatterPlotData(
            title="Average branch angle",
            xaxis_title="Time (µs)",
            yaxis_title="Branch angle (°)",
            xtrace=analyzer.times,
            ytraces={
                "Ideal": np.array(analyzer.times.shape[0] * [70.9]),
                "Mean": mean,
                "Mean - std" : mean - stddev,
                "Mean + std" : mean + stddev,
            },
            render_mode="lines",
        )

    @staticmethod
    def get_helix_pitch_plot(analyzer):
        """
        Add a plot of average helix pitch
        for both the short and long helices
        ideal Ref: http://www.jbc.org/content/266/1/1.full.pdf
        """
        return ScatterPlotData(
            title="Average helix pitch",
            xaxis_title="Time (µs)",
            yaxis_title="Pitch (nm)",
            xtrace=analyzer.times,
            ytraces={
                "Ideal short pitch": np.array(analyzer.times.shape[0] * [5.9]),
                "Mean short pitch": ActinAnalyzer.analyze_average_over_time(
                    analyzer.analyze_short_helix_pitches()),
                "Ideal long pitch": np.array(analyzer.times.shape[0] * [72]),
                "Mean long pitch": ActinAnalyzer.analyze_average_over_time(
                    analyzer.analyze_long_helix_pitches()),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_filament_straightness_plot(analyzer):
        """
        Add a plot of how many nm each monomer is away
        from ideal position in a straight filament
        """
        return ScatterPlotData(
            title="Filament bending",
            xaxis_title="Time (µs)",
            yaxis_title="Filament bending",
            xtrace=analyzer.times,
            ytraces={
                "Filament bending": (
                    ActinAnalyzer.analyze_average_over_time(
                        analyzer.analyze_filament_straightness()
                    )
                ),
            },
            render_mode="lines",
        )

    # @staticmethod
    # def get_bound_actin_plot(analyzer):
    #     """
    #     Add a plot of percent actin in filaments
    #     """
    #     return ScatterPlotData(
    #         title="Filamentous actin",
    #         xaxis_title="Time (µs)",
    #         yaxis_title="Bound actin (%)",
    #         xtrace=analyzer.times,
    #         ytraces={
    #             "Bound actin": 100.0
    #             * analyzer.analyze_ratio_of_filamentous_to_total_actin(),
    #         },
    #         render_mode="lines",
    #     )

    @staticmethod
    def generate_plots(path_to_readdy_h5, box_size, stride=1):
        """
        Use an ActinAnalyzer to generate plots of observables
        """
        analyzer = ActinAnalyzer(path_to_readdy_h5, box_size, stride)
        analyzer.times = TIMESTEP * 1e-3 * analyzer.times
        return {
            "scatter" : [
                ActinVisualization.get_bound_actin_plot(analyzer),
                ActinVisualization.get_ATP_actin_plot(analyzer),
                ActinVisualization.get_daughter_actin_plot(analyzer),
                ActinVisualization.get_avg_mother_length_plot(analyzer),
                ActinVisualization.get_avg_daughter_length_plot(analyzer),
                ActinVisualization.get_bound_arp_plot(analyzer),
                ActinVisualization.get_capped_ends_plot(analyzer),
                ActinVisualization.get_branch_angle_plot(analyzer),
                ActinVisualization.get_helix_pitch_plot(analyzer),
                ActinVisualization.get_filament_straightness_plot(analyzer),
            ],
            "histogram" : [],
        }
        # reactions = analyzer.analyze_all_reaction_events_over_time()
        # free_actin = analyzer.analyze_free_actin_concentration_over_time()

    @staticmethod
    def visualize_actin(path_to_readdy_h5, box_size, total_steps, plots={}):
        """
        visualize an actin trajectory in Simularium
        """
        # radii
        extra_radius = 1.5
        actin_radius = 2.0 + extra_radius
        arp23_radius = 2.0 + extra_radius
        cap_radius = 3.0 + extra_radius
        radii = {
            "arp2": arp23_radius,
            "arp2#branched": arp23_radius,
            "arp3": arp23_radius,
            "arp3#ATP": arp23_radius,
            "arp3#new": arp23_radius,
            "arp3#new_ATP": arp23_radius,
            "cap": cap_radius,
            "cap#new": cap_radius,
            "cap#bound": cap_radius,
            "actin#free": actin_radius,
            "actin#free_ATP": actin_radius,
            "actin#new": actin_radius,
            "actin#new_ATP": actin_radius,
            "actin#1": actin_radius,
            "actin#2": actin_radius,
            "actin#3": actin_radius,
            "actin#ATP_1": actin_radius,
            "actin#ATP_2": actin_radius,
            "actin#ATP_3": actin_radius,
            "actin#pointed_1": actin_radius,
            "actin#pointed_2": actin_radius,
            "actin#pointed_3": actin_radius,
            "actin#pointed_ATP_1": actin_radius,
            "actin#pointed_ATP_2": actin_radius,
            "actin#pointed_ATP_3": actin_radius,
            "actin#barbed_1": actin_radius,
            "actin#barbed_2": actin_radius,
            "actin#barbed_3": actin_radius,
            "actin#barbed_ATP_1": actin_radius,
            "actin#barbed_ATP_2": actin_radius,
            "actin#barbed_ATP_3": actin_radius,
            "actin#branch_1": actin_radius,
            "actin#branch_ATP_1": actin_radius,
            "actin#branch_barbed_1": actin_radius,
            "actin#branch_barbed_ATP_1": actin_radius,
        }
        # type grouping
        type_grouping = {
            "arp2": ["arp2", "arp2#new"],
            "arp2#ATP": ["arp2#ATP", "arp2#new_ATP"],
            "cap": ["cap", "cap#new"],
            "actin#free": ["actin#free", "actin#new"],
            "actin#free_ATP": ["actin#free_ATP", "actin#new_ATP"],
            "actin": ["actin#1", "actin#2", "actin#3"],
            "actin#ATP": ["actin#ATP_1", "actin#ATP_2", "actin#ATP_3"],
            "actin#pointed": ["actin#pointed_1", "actin#pointed_2", "actin#pointed_3"],
            "actin#pointed_ATP": [
                "actin#pointed_ATP_1",
                "actin#pointed_ATP_2",
                "actin#pointed_ATP_3",
            ],
            "actin#barbed": ["actin#barbed_1", "actin#barbed_2", "actin#barbed_3"],
            "actin#barbed_ATP": [
                "actin#barbed_ATP_1",
                "actin#barbed_ATP_2",
                "actin#barbed_ATP_3",
            ],
            "actin#branch": ["actin#branch_1"],
            "actin#branch_ATP": ["actin#branch_ATP_1"],
            "actin#branch_barbed": ["actin#branch_barbed_1"],
            "actin#branch_barbed_ATP": ["actin#branch_barbed_ATP_1"],
        }
        # convert
        data = ReaddyData(
            meta_data=MetaData(
                box_size=np.array([box_size, box_size, box_size]),
            ),
            # assume 1e3 recorded steps
            timestep=TIMESTEP * total_steps * 1e-3, 
            path_to_readdy_h5=path_to_readdy_h5,
            radii=radii,
            type_grouping=type_grouping,
            time_units=UnitData("µs"),
            spatial_units=UnitData("nm"),
        )
        converter = ReaddyConverter(data)
        for plot_type in plots:
            for plot in plots[plot_type]:
                converter.add_plot(plot, plot_type)
        filtered_data = converter.filter_data([
            MultiplyTimeFilter(
                multiplier=1e-3,
                apply_to_plots=False,
            ),
        ])
        converter.write_external_JSON(filtered_data, path_to_readdy_h5)
