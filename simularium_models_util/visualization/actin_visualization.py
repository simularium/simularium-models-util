#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import MetaData, UnitData, ScatterPlotData, DisplayData
from simulariumio.filters import MultiplyTimeFilter
from ..actin import ActinAnalyzer

TIMESTEP = 0.1  # ns
SPECIES_COUNT_RXNS = ["Dimers", "Trimers"]
POLYMERIZATION_RXNS = [
    "Barbed growth ATP",
    "Barbed growth ADP",
    "Pointed growth ATP",
    "Pointed growth ADP",
]
DEPOLYMERIZATION_RXNS = [
    "Pointed shrink ATP",
    "Pointed shrink ADP",
    "Barbed shrink ATP",
    "Barbed shrink ADP",
]
GROWTH_REACTIONS = [
    "Dimerize",
    "Trimerize",
    "Barbed growth ATP",
    "Barbed growth ADP",
    "Pointed growth ATP",
    "Pointed growth ADP",
    "Branch ATP",
    "Branch ADP",
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
                "ATP-Actin in filaments": 100.0
                * analyzer.analyze_ratio_of_bound_ATP_actin_to_total_actin(),
                "Arp2/3 in filaments": 100.0
                * analyzer.analyze_ratio_of_bound_to_total_arp23(),
                "Actin in daughter filaments": 100.0
                * analyzer.analyze_ratio_of_daughter_to_total_actin(),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_dimers_trimers_plot(analyzer):
        """
        Add a plot of number of dimer and trimer complexes over time
        """
        ytraces = {}
        for total_rxn_name in SPECIES_COUNT_RXNS:
            rate = analyzer.reactions[total_rxn_name].to_numpy()
            ytraces[total_rxn_name] = np.zeros(rate.shape[0] + 1)
            for t in range(len(rate) + 1):
                ytraces[total_rxn_name][t] = np.sum(rate[:t])
        return ScatterPlotData(
            title="Actin dimers and trimers",
            xaxis_title="Time (µs)",
            yaxis_title="Number of complexes",
            xtrace=analyzer.times,
            ytraces=ytraces,
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
    def get_polymerization_reactions_plot(analyzer):
        """
        Add a plot of cumulative reaction events over time
        for each total polymerization reaction over time
        """
        ytraces = {}
        for total_rxn_name in POLYMERIZATION_RXNS:
            rxn_rate = analyzer.analyze_reaction_rate_over_time(total_rxn_name)
            if rxn_rate is not None:
                ytraces[total_rxn_name] = rxn_rate
        return ScatterPlotData(
            title="Polymerization reaction rates",
            xaxis_title="Time (µs)",
            yaxis_title="Rate (s\u207B\u00B9)",
            xtrace=analyzer.times,
            ytraces=ytraces,
            render_mode="lines",
        )

    @staticmethod
    def get_depolymerization_reactions_plot(analyzer):
        """
        Add a plot of cumulative reaction events over time
        for each total polymerization reaction over time
        """
        ytraces = {}
        for total_rxn_name in DEPOLYMERIZATION_RXNS:
            rxn_rate = analyzer.analyze_reaction_rate_over_time(total_rxn_name)
            if rxn_rate is not None:
                ytraces[total_rxn_name] = rxn_rate
        return ScatterPlotData(
            title="Depolymerization reaction rates",
            xaxis_title="Time (µs)",
            yaxis_title="Rate (s\u207B\u00B9)",
            xtrace=analyzer.times,
            ytraces=ytraces,
            render_mode="lines",
        )

    @staticmethod
    def get_actin_growth_reactions_vs_concentration_plot(analyzer):
        """
        Add a plot of cumulative reaction events over time
        for each total polymerization reaction over time
        """
        ytraces = {}
        for total_rxn_name in GROWTH_REACTIONS:
            rxn_rate = analyzer.analyze_reaction_rate_over_time(total_rxn_name)
            if rxn_rate is not None:
                ytraces[total_rxn_name] = rxn_rate
        return ScatterPlotData(
            title="Actin growth vs concentration",
            xaxis_title="[Actin] (µM)",
            yaxis_title="Rate (s\u207B\u00B9)",
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
    def generate_plots(path_to_readdy_h5, box_size, stride=1):
        """
        Use an ActinAnalyzer to generate plots of observables
        """
        analyzer = ActinAnalyzer(path_to_readdy_h5, box_size, stride)
        return {
            "scatter": [
                ActinVisualization.get_bound_monomers_plot(analyzer),
                ActinVisualization.get_dimers_trimers_plot(analyzer),
                ActinVisualization.get_avg_length_plot(analyzer),
                ActinVisualization.get_polymerization_reactions_plot(analyzer),
                ActinVisualization.get_depolymerization_reactions_plot(analyzer),
                ActinVisualization.get_actin_growth_reactions_vs_concentration_plot(
                    analyzer
                ),
                ActinVisualization.get_capped_ends_plot(analyzer),
                ActinVisualization.get_branch_angle_plot(analyzer),
                ActinVisualization.get_helix_pitch_plot(analyzer),
                ActinVisualization.get_filament_straightness_plot(analyzer),
            ],
            "histogram": [],
        }

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
        display_data = {
            "arp2": DisplayData(
                name="arp2",
                radius=arp23_radius,
            ),
            "arp2#branched": DisplayData(
                name="arp2#branched",
                radius=arp23_radius,
            ),
            "arp2#free": DisplayData(
                name="arp2#free",
                radius=arp23_radius,
            ),
            "arp3": DisplayData(
                name="arp3",
                radius=arp23_radius,
            ),
            "arp3#new": DisplayData(
                name="arp3",
                radius=arp23_radius,
            ),
            "arp3#ATP": DisplayData(
                name="arp3#ATP",
                radius=arp23_radius,
            ),
            "arp3#new_ATP": DisplayData(
                name="arp3#ATP",
                radius=arp23_radius,
            ),
            "cap": DisplayData(
                name="cap",
                radius=cap_radius,
            ),
            "cap#new": DisplayData(
                name="cap",
                radius=cap_radius,
            ),
            "cap#bound": DisplayData(
                name="cap#bound",
                radius=cap_radius,
            ),
            "actin#free": DisplayData(
                name="actin#free",
                radius=actin_radius,
            ),
            "actin#free_ATP": DisplayData(
                name="actin#free_ATP",
                radius=actin_radius,
            ),
            "actin#new": DisplayData(
                name="actin",
                radius=actin_radius,
            ),
            "actin#new_ATP": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
            ),
            "actin#1": DisplayData(
                name="actin",
                radius=actin_radius,
            ),
            "actin#2": DisplayData(
                name="actin",
                radius=actin_radius,
            ),
            "actin#3": DisplayData(
                name="actin",
                radius=actin_radius,
            ),
            "actin#ATP_1": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
            ),
            "actin#ATP_2": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
            ),
            "actin#ATP_3": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
            ),
            "actin#pointed_1": DisplayData(
                name="actin#pointed",
                radius=actin_radius,
            ),
            "actin#pointed_2": DisplayData(
                name="actin#pointed",
                radius=actin_radius,
            ),
            "actin#pointed_3": DisplayData(
                name="actin#pointed",
                radius=actin_radius,
            ),
            "actin#pointed_ATP_1": DisplayData(
                name="actin#pointed_ATP",
                radius=actin_radius,
            ),
            "actin#pointed_ATP_2": DisplayData(
                name="actin#pointed_ATP",
                radius=actin_radius,
            ),
            "actin#pointed_ATP_3": DisplayData(
                name="actin#pointed_ATP",
                radius=actin_radius,
            ),
            "actin#barbed_1": DisplayData(
                name="actin#barbed",
                radius=actin_radius,
            ),
            "actin#barbed_2": DisplayData(
                name="actin#barbed",
                radius=actin_radius,
            ),
            "actin#barbed_3": DisplayData(
                name="actin#barbed",
                radius=actin_radius,
            ),
            "actin#barbed_ATP_1": DisplayData(
                name="actin#barbed_ATP",
                radius=actin_radius,
            ),
            "actin#barbed_ATP_2": DisplayData(
                name="actin#barbed_ATP",
                radius=actin_radius,
            ),
            "actin#barbed_ATP_3": DisplayData(
                name="actin#barbed_ATP",
                radius=actin_radius,
            ),
            "actin#branch_1": DisplayData(
                name="actin#branch",
                radius=actin_radius,
            ),
            "actin#branch_ATP_1": DisplayData(
                name="actin#branch_ATP",
                radius=actin_radius,
            ),
            "actin#branch_barbed_1": DisplayData(
                name="actin#branch_barbed",
                radius=actin_radius,
            ),
            "actin#branch_barbed_ATP_1": DisplayData(
                name="actin#branch_barbed_ATP",
                radius=actin_radius,
            ),
        }
        # convert
        data = ReaddyData(
            meta_data=MetaData(
                box_size=np.array([box_size, box_size, box_size]),
            ),
            # assume 1e3 recorded steps
            timestep=TIMESTEP * total_steps * 1e-3,
            path_to_readdy_h5=path_to_readdy_h5,
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
