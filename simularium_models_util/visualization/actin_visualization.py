#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import MetaData, UnitData, ScatterPlotData, DisplayData
from simulariumio.filters import MultiplyTimeFilter
from ..actin import ActinAnalyzer

TIMESTEP = 0.1  # ns
GROWTH_RXNS = [
    "Dimerize",
    "Trimerize",
    "Grow Pointed",
    "Grow Barbed",
    "Bind Arp2/3",
    "Start Branch",
    "Bind Cap",
]
GROUPED_GROWTH_RXNS = {
    "Dimerize Actin": ["Dimerize"],
    "Polymerize Actin": ["Trimerize", "Grow Pointed", "Grow Barbed", "Start Branch"],
    "Bind Arp2/3": ["Bind Arp2/3"],
    "Bind Cap": ["Bind Cap"],
}
STRUCTURAL_RXNS = [
    "Reverse Dimerize",
    "Reverse Trimerize",
    "Shrink Pointed",
    "Shrink Barbed",
    "Unbind Arp2/3",
    "Debranch",
    "Unbind Cap",
    "Hydrolyze Actin",
    "Hydrolyze Arp2/3",
    "Bind ATP (actin)",
    "Bind ATP (arp2/3)",
]


class ActinVisualization:
    """
    visualize an actin trajectory in Simularium
    """

    @staticmethod
    def get_bound_monomers_plot(analyzer):
        """
        Add a plot of percent actin in filaments
        """
        return ScatterPlotData(
            title="Monomers over time",
            xaxis_title="Time (µs)",
            yaxis_title="Monomers (%)",
            xtrace=analyzer.times,
            ytraces={
                "Actin in filaments": 100.0
                * analyzer.analyze_ratio_of_filamentous_to_total_actin(),
                "Arp2/3 in filaments": 100.0
                * analyzer.analyze_ratio_of_bound_to_total_arp23(),
                "Actin in daughter filaments": 100.0
                * analyzer.analyze_ratio_of_daughter_to_total_actin(),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_avg_length_plot(analyzer):
        """
        Add a plot of average mother and daughter filament length
        """
        return ScatterPlotData(
            title="Average length of filaments",
            xaxis_title="Time (µs)",
            yaxis_title="Average length (monomers)",
            xtrace=analyzer.times,
            ytraces={
                "Mother filaments": ActinAnalyzer.analyze_average_over_time(
                    analyzer.analyze_mother_filament_lengths()
                ),
                "Daughter filaments": ActinAnalyzer.analyze_average_over_time(
                    analyzer.analyze_daughter_filament_lengths()
                ),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_growth_reactions_plot(analyzer):
        """
        Add a plot of reaction events over time
        for each total growth reaction
        """
        ytraces = {}
        for total_rxn_name in GROWTH_RXNS:
            rxn_events = analyzer.analyze_reaction_count_over_time(total_rxn_name)
            if rxn_events is not None:
                ytraces[total_rxn_name] = rxn_events
        return ScatterPlotData(
            title="Growth reactions",
            xaxis_title="Time (µs)",
            yaxis_title="Reaction events",
            xtrace=analyzer.times,
            ytraces=ytraces,
            render_mode="lines",
        )

    @staticmethod
    def get_structural_reactions_plot(analyzer):
        """
        Add a plot of the number of times a structural reaction
        was triggered over time
        Note: triggered != completed, the reaction may have failed
        to find the required reactants
        """
        ytraces = {}
        for total_rxn_name in STRUCTURAL_RXNS:
            rxn_events = analyzer.analyze_reaction_count_over_time(total_rxn_name)
            if rxn_events is not None:
                ytraces[total_rxn_name] = rxn_events
        return ScatterPlotData(
            title="Structural reaction triggers",
            xaxis_title="Time (µs)",
            yaxis_title="Reactions triggered",
            xtrace=analyzer.times,
            ytraces=ytraces,
            render_mode="lines",
        )

    @staticmethod
    def get_growth_reactions_vs_actin_plot(analyzer):
        """
        Add a plot of average reaction events over time
        for each total growth reaction
        """
        ytraces = {}
        for rxn_group_name in GROUPED_GROWTH_RXNS:
            group_reaction_events = []
            for total_rxn_name in GROUPED_GROWTH_RXNS[rxn_group_name]:
                group_reaction_events.append(
                    analyzer.analyze_reaction_count_over_time(total_rxn_name)
                )
            if len(group_reaction_events) > 0:
                ytraces[rxn_group_name] = np.sum(
                    np.array(group_reaction_events), axis=0
                )
        return ScatterPlotData(
            title="Growth vs [actin]",
            xaxis_title="[Actin] (µM)",
            yaxis_title="Reaction events",
            xtrace=analyzer.analyze_free_actin_concentration_over_time(),
            ytraces=ytraces,
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
                "Mean - std": mean - stddev,
                "Mean + std": mean + stddev,
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
                    analyzer.analyze_short_helix_pitches()
                ),
                "Ideal long pitch": np.array(analyzer.times.shape[0] * [72]),
                "Mean long pitch": ActinAnalyzer.analyze_average_over_time(
                    analyzer.analyze_long_helix_pitches()
                ),
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

    @staticmethod
    def generate_plots(path_to_readdy_h5, box_size, stride=1, periodic_boundary=True):
        """
        Use an ActinAnalyzer to generate plots of observables
        """
        analyzer = ActinAnalyzer(path_to_readdy_h5, box_size, stride, periodic_boundary)
        return {
            "scatter": [
                ActinVisualization.get_bound_monomers_plot(analyzer),
                ActinVisualization.get_avg_length_plot(analyzer),
                ActinVisualization.get_growth_reactions_plot(analyzer),
                ActinVisualization.get_growth_reactions_vs_actin_plot(analyzer),
                # ActinVisualization.get_capped_ends_plot(analyzer),
                ActinVisualization.get_branch_angle_plot(analyzer),
                ActinVisualization.get_helix_pitch_plot(analyzer),
                ActinVisualization.get_filament_straightness_plot(analyzer),
                ActinVisualization.get_structural_reactions_plot(analyzer),
            ],
            "histogram": [],
        }

    @staticmethod
    def visualize_actin(path_to_readdy_h5, box_size, total_steps, plots=None):
        """
        visualize an actin trajectory in Simularium
        """
        # radii
        extra_radius = 1.5
        actin_radius = 2.0 + extra_radius
        arp23_radius = 2.0 + extra_radius
        cap_radius = 3.0 + extra_radius
        obstacle_radius = 35.0
        display_data = {
            "arp2": DisplayData(
                name="arp2",
                radius=arp23_radius,
                color="#c9df8a",
            ),
            "arp2#branched": DisplayData(
                name="arp2#branched",
                radius=arp23_radius,
                color="#c9df8a",
            ),
            "arp2#free": DisplayData(
                name="arp2#free",
                radius=arp23_radius,
                color="#234d20",
            ),
            "arp3": DisplayData(
                name="arp3",
                radius=arp23_radius,
                color="#36802d",
            ),
            "arp3#new": DisplayData(
                name="arp3",
                radius=arp23_radius,
                color="#36802d",
            ),
            "arp3#ATP": DisplayData(
                name="arp3#ATP",
                radius=arp23_radius,
                color="#77ab59",
            ),
            "arp3#new_ATP": DisplayData(
                name="arp3#ATP",
                radius=arp23_radius,
                color="#77ab59",
            ),
            "cap": DisplayData(
                name="cap",
                radius=cap_radius,
                color="#005073",
            ),
            "cap#new": DisplayData(
                name="cap",
                radius=cap_radius,
                color="#189ad3",
            ),
            "cap#bound": DisplayData(
                name="cap#bound",
                radius=cap_radius,
                color="#189ad3",
            ),
            "actin#free": DisplayData(
                name="actin#free",
                radius=actin_radius,
                color="#8d5524",
            ),
            "actin#free_ATP": DisplayData(
                name="actin#free_ATP",
                radius=actin_radius,
                color="#cd8500",
            ),
            "actin#new": DisplayData(
                name="actin",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#new_ATP": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#1": DisplayData(
                name="actin",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#2": DisplayData(
                name="actin",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#3": DisplayData(
                name="actin",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#ATP_1": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#ATP_2": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#ATP_3": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#mid_1": DisplayData(
                name="actin#mid",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#mid_2": DisplayData(
                name="actin#mid",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#mid_3": DisplayData(
                name="actin#mid",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#mid_ATP_1": DisplayData(
                name="actin#mid_ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#mid_ATP_2": DisplayData(
                name="actin#mid_ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#mid_ATP_3": DisplayData(
                name="actin#mid_ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#pointed_1": DisplayData(
                name="actin#pointed",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#pointed_2": DisplayData(
                name="actin#pointed",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#pointed_3": DisplayData(
                name="actin#pointed",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#pointed_ATP_1": DisplayData(
                name="actin#pointed_ATP",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#pointed_ATP_2": DisplayData(
                name="actin#pointed_ATP",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#pointed_ATP_3": DisplayData(
                name="actin#pointed_ATP",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#barbed_1": DisplayData(
                name="actin#barbed",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#barbed_2": DisplayData(
                name="actin#barbed",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#barbed_3": DisplayData(
                name="actin#barbed",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#barbed_ATP_1": DisplayData(
                name="actin#barbed_ATP",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#barbed_ATP_2": DisplayData(
                name="actin#barbed_ATP",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#barbed_ATP_3": DisplayData(
                name="actin#barbed_ATP",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#fixed_1": DisplayData(
                name="actin#fixed",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#fixed_2": DisplayData(
                name="actin#fixed",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#fixed_3": DisplayData(
                name="actin#fixed",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#fixed_ATP_1": DisplayData(
                name="actin#fixed_ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#fixed_ATP_2": DisplayData(
                name="actin#fixed_ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#fixed_ATP_3": DisplayData(
                name="actin#fixed_ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#mid_fixed_1": DisplayData(
                name="actin#mid_fixed",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#mid_fixed_2": DisplayData(
                name="actin#mid_fixed",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#mid_fixed_3": DisplayData(
                name="actin#mid_fixed",
                radius=actin_radius,
                color="#bf9b30",
            ),
            "actin#mid_fixed_ATP_1": DisplayData(
                name="actin#mid_fixed_ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#mid_fixed_ATP_2": DisplayData(
                name="actin#mid_fixed_ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#mid_fixed_ATP_3": DisplayData(
                name="actin#mid_fixed_ATP",
                radius=actin_radius,
                color="#ffbf00",
            ),
            "actin#pointed_fixed_1": DisplayData(
                name="actin#pointed_fixed",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#pointed_fixed_2": DisplayData(
                name="actin#pointed_fixed",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#pointed_fixed_3": DisplayData(
                name="actin#pointed_fixed",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#pointed_fixed_ATP_1": DisplayData(
                name="actin#pointed_fixed_ATP",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#pointed_fixed_ATP_2": DisplayData(
                name="actin#pointed_fixed_ATP",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#pointed_fixed_ATP_3": DisplayData(
                name="actin#pointed_fixed_ATP",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#fixed_barbed_1": DisplayData(
                name="actin#fixed_barbed",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#fixed_barbed_2": DisplayData(
                name="actin#fixed_barbed",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#fixed_barbed_3": DisplayData(
                name="actin#fixed_barbed",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#fixed_barbed_ATP_1": DisplayData(
                name="actin#fixed_barbed_ATP",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#fixed_barbed_ATP_2": DisplayData(
                name="actin#fixed_barbed_ATP",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#fixed_barbed_ATP_3": DisplayData(
                name="actin#fixed_barbed_ATP",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#branch_1": DisplayData(
                name="actin#branch",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#branch_ATP_1": DisplayData(
                name="actin#branch_ATP",
                radius=actin_radius,
                color="#a67c00",
            ),
            "actin#branch_barbed_1": DisplayData(
                name="actin#branch_barbed",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "actin#branch_barbed_ATP_1": DisplayData(
                name="actin#branch_barbed_ATP",
                radius=actin_radius,
                color="#ffdc73",
            ),
            "obstacle": DisplayData(
                name="obstacle",
                radius=obstacle_radius,
                color="#666666",
            ),
        }
        # convert
        data = ReaddyData(
            # assume 1e3 recorded steps
            timestep=TIMESTEP * total_steps * 1e-3,
            path_to_readdy_h5=path_to_readdy_h5,
            meta_data=MetaData(
                box_size=np.array([box_size, box_size, box_size]),
            ),
            display_data=display_data,
            time_units=UnitData("µs"),
            spatial_units=UnitData("nm"),
            plots=[],
        )
        try:
            converter = ReaddyConverter(data)
        except OverflowError as e:
            print(
                "OverflowError during SimulariumIO conversion !!!!!!!!!!!!!!\n" + str(e)
            )
            return
        if plots is not None:
            for plot_type in plots:
                for plot in plots[plot_type]:
                    converter.add_plot(plot, plot_type)
        filtered_data = converter.filter_data(
            [
                MultiplyTimeFilter(
                    multiplier=1e-3,
                    apply_to_plots=False,
                ),
            ]
        )
        converter.write_external_JSON(filtered_data, path_to_readdy_h5)
