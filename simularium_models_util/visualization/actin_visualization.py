#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import MetaData, UnitData, ScatterPlotData, DisplayData, DISPLAY_TYPE
from simulariumio.filters import MultiplyTimeFilter
from simulariumio.orientations import OrientationData, NeighborData
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
        # obstacle_radius = 35.0
        bucket_url = "https://aics-simularium-data.s3.us-east-2.amazonaws.com"
        test_geo = f"{bucket_url}/geometry/gizmo_small.obj"
        actin_geometry_url = test_geo  # f"{bucket_url}/geometry/actin.pdb"
        arp2_geometry_url = test_geo  # f"{bucket_url}/geometry/arp2.pdb"
        arp3_geometry_url = test_geo  # f"{bucket_url}/geometry/arp3.pdb"
        display_type = DISPLAY_TYPE.OBJ  # DISPLAY_TYPE.PDB
        display_data = {
            "arp2": DisplayData(
                name="arp2",
                radius=arp23_radius,
                display_type=display_type,
                url=arp2_geometry_url,
                color="#c9df8a",
            ),
            "arp2#branched": DisplayData(
                name="arp2#branched",
                radius=arp23_radius,
                display_type=display_type,
                url=arp2_geometry_url,
                color="#c9df8a",
            ),
            "arp2#free": DisplayData(
                name="arp2#free",
                radius=arp23_radius,
                display_type=display_type,
                url=arp2_geometry_url,
                color="#234d20",
            ),
            "arp3": DisplayData(
                name="arp3",
                radius=arp23_radius,
                display_type=display_type,
                url=arp3_geometry_url,
                color="#36802d",
            ),
            "arp3#new": DisplayData(
                name="arp3",
                radius=arp23_radius,
                display_type=display_type,
                url=arp3_geometry_url,
                color="#36802d",
            ),
            "arp3#ATP": DisplayData(
                name="arp3#ATP",
                radius=arp23_radius,
                display_type=display_type,
                url=arp3_geometry_url,
                color="#77ab59",
            ),
            "arp3#new_ATP": DisplayData(
                name="arp3#ATP",
                radius=arp23_radius,
                display_type=display_type,
                url=arp3_geometry_url,
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
                display_type=display_type,
                url=actin_geometry_url,
                color="#8d5524",
            ),
            "actin#free_ATP": DisplayData(
                name="actin#free_ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#cd8500",
            ),
            "actin#new": DisplayData(
                name="actin",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#bf9b30",
            ),
            "actin#new_ATP": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffbf00",
            ),
            "actin#1": DisplayData(
                name="actin",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#bf9b30",
            ),
            "actin#2": DisplayData(
                name="actin",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#bf9b30",
            ),
            "actin#3": DisplayData(
                name="actin",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#bf9b30",
            ),
            "actin#ATP_1": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffbf00",
            ),
            "actin#ATP_2": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffbf00",
            ),
            "actin#ATP_3": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffbf00",
            ),
            "actin#mid_1": DisplayData(
                name="actin#mid",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#bf9b30",
            ),
            "actin#mid_2": DisplayData(
                name="actin#mid",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#bf9b30",
            ),
            "actin#mid_3": DisplayData(
                name="actin#mid",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#bf9b30",
            ),
            "actin#mid_ATP_1": DisplayData(
                name="actin#mid_ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffbf00",
            ),
            "actin#mid_ATP_2": DisplayData(
                name="actin#mid_ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffbf00",
            ),
            "actin#mid_ATP_3": DisplayData(
                name="actin#mid_ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffbf00",
            ),
            "actin#pointed_1": DisplayData(
                name="actin#pointed",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#a67c00",
            ),
            "actin#pointed_2": DisplayData(
                name="actin#pointed",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#a67c00",
            ),
            "actin#pointed_3": DisplayData(
                name="actin#pointed",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#a67c00",
            ),
            "actin#pointed_ATP_1": DisplayData(
                name="actin#pointed_ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#a67c00",
            ),
            "actin#pointed_ATP_2": DisplayData(
                name="actin#pointed_ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#a67c00",
            ),
            "actin#pointed_ATP_3": DisplayData(
                name="actin#pointed_ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#a67c00",
            ),
            "actin#barbed_1": DisplayData(
                name="actin#barbed",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffdc73",
            ),
            "actin#barbed_2": DisplayData(
                name="actin#barbed",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffdc73",
            ),
            "actin#barbed_3": DisplayData(
                name="actin#barbed",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffdc73",
            ),
            "actin#barbed_ATP_1": DisplayData(
                name="actin#barbed_ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffdc73",
            ),
            "actin#barbed_ATP_2": DisplayData(
                name="actin#barbed_ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffdc73",
            ),
            "actin#barbed_ATP_3": DisplayData(
                name="actin#barbed_ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffdc73",
            ),
            "actin#branch_1": DisplayData(
                name="actin#branch",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#a67c00",
            ),
            "actin#branch_ATP_1": DisplayData(
                name="actin#branch_ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#a67c00",
            ),
            "actin#branch_barbed_1": DisplayData(
                name="actin#branch_barbed",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffdc73",
            ),
            "actin#branch_barbed_ATP_1": DisplayData(
                name="actin#branch_barbed_ATP",
                radius=actin_radius,
                display_type=display_type,
                url=actin_geometry_url,
                color="#ffdc73",
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
            zero_orientations=[
                OrientationData(
                    type_name_substrings=["actin", "2"],
                    neighbor_data=[
                        NeighborData(
                            type_name_substrings=["actin", "1"],
                            relative_position=np.array([19.126, 20.838, 27.757])
                            - np.array([21.847, 24.171, 27.148]),
                            neighbor_type_name_substrings=["actin", "3"],
                            neighbor_relative_position=np.array(
                                [16.236, 23.926, 26.754]
                            ),
                        ),
                        NeighborData(
                            type_name_substrings=["actin", "3"],
                            relative_position=np.array([24.738, 20.881, 26.671])
                            - np.array([21.847, 24.171, 27.148]),
                            neighbor_type_name_substrings=["actin", "1"],
                            neighbor_relative_position=np.array(
                                [27.609, 24.061, 27.598]
                            ),
                        ),
                    ],
                ),
                OrientationData(
                    type_name_substrings=["actin", "3"],
                    neighbor_data=[
                        NeighborData(
                            type_name_substrings=["actin", "2"],
                            relative_position=np.array([19.126, 20.838, 27.757])
                            - np.array([21.847, 24.171, 27.148]),
                            neighbor_type_name_substrings=["actin", "1"],
                            neighbor_relative_position=np.array(
                                [16.236, 23.926, 26.754]
                            ),
                        ),
                        NeighborData(
                            type_name_substrings=["actin", "1"],
                            relative_position=np.array([24.738, 20.881, 26.671])
                            - np.array([21.847, 24.171, 27.148]),
                            neighbor_type_name_substrings=["actin", "2"],
                            neighbor_relative_position=np.array(
                                [27.609, 24.061, 27.598]
                            ),
                        ),
                    ],
                ),
                OrientationData(
                    type_name_substrings=["actin", "1"],
                    neighbor_data=[
                        NeighborData(
                            type_name_substrings=["actin", "3"],
                            relative_position=np.array([19.126, 20.838, 27.757])
                            - np.array([21.847, 24.171, 27.148]),
                            neighbor_type_name_substrings=["actin", "2"],
                            neighbor_relative_position=np.array(
                                [16.236, 23.926, 26.754]
                            ),
                        ),
                        NeighborData(
                            type_name_substrings=["actin", "2"],
                            relative_position=np.array([24.738, 20.881, 26.671])
                            - np.array([21.847, 24.171, 27.148]),
                            neighbor_type_name_substrings=["actin", "3"],
                            neighbor_relative_position=np.array(
                                [27.609, 24.061, 27.598]
                            ),
                        ),
                    ],
                ),
                OrientationData(
                    type_name_substrings=["arp3"],
                    neighbor_data=[
                        NeighborData(
                            type_name_substrings=["arp2"],
                            relative_position=np.array([28.087, 30.872, 26.657])
                            - np.array([29.275, 27.535, 23.944]),
                        ),
                        NeighborData(
                            type_name_substrings=["actin"],
                            relative_position=np.array([30.382, 21.190, 25.725])
                            - np.array([29.275, 27.535, 23.944]),
                        ),
                    ],
                ),
                OrientationData(
                    type_name_substrings=["arp2#branched"],
                    neighbor_data=[
                        NeighborData(
                            type_name_substrings=["arp3"],
                            relative_position=np.array([29.275, 27.535, 23.944])
                            - np.array([28.087, 30.872, 26.657]),
                        ),
                        NeighborData(
                            type_name_substrings=["actin#branch"],
                            relative_position=np.array([29.821, 33.088, 23.356])
                            - np.array([28.087, 30.872, 26.657]),
                        ),
                    ],
                ),
                OrientationData(
                    type_name_substrings=["actin#branch"],
                    neighbor_data=[
                        NeighborData(
                            type_name_substrings=["arp2#branched"],
                            relative_position=np.array([28.087, 30.872, 26.657])
                            - np.array([29.821, 33.088, 23.356]),
                        ),
                        NeighborData(
                            type_name_substrings=["actin", "2"],
                            relative_position=np.array([30.476, 36.034, 26.528])
                            - np.array([29.821, 33.088, 23.356]),
                        ),
                    ],
                ),
            ],
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
