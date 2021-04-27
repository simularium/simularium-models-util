#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import sys
import numpy as np
import readdy

from ..common import ReaddyUtil
from .actin_util import ActinUtil


class ActinAnalyzer:
    def __init__(self, h5_file_path, box_size, analyze_reactions=True):
        """
        Load data from a ReaDDy trajectory
        """
        self.box_size = box_size
        self.traj = readdy.Trajectory(h5_file_path)
        self.times, self.topology_records = self.traj.read_observable_topologies()
        (
            self.times,
            self.types,
            self.ids,
            self.positions,
        ) = self.traj.read_observable_particles()
        self.particle_data = ReaddyUtil.shape_particle_data(
            0,
            self.times.shape[0],
            1,
            self.times,
            self.topology_records,
            self.ids,
            self.types,
            self.positions,
            self.traj,
        )
        self.reactions = None

    @staticmethod
    def _free_actin_types():
        """
        Get all the types for actins freely diffusing
        """
        return ["actin#free", "actin#free_ATP"]

    @staticmethod
    def _pointed_actin_types():
        """
        Get all the types for actins at the pointed end of a filament
        """
        result = []
        for i in range(1, 4):
            result.append(f"actin#pointed_{i}")
            result.append(f"actin#pointed_ATP_{i}")
        return result

    @staticmethod
    def _middle_actin_types():
        """
        Get all the types for actins in the middle of a filament
        """
        result = []
        for i in range(1, 4):
            result.append(f"actin#{i}")
            result.append(f"actin#ATP_{i}")
        return result

    @staticmethod
    def _barbed_actin_types():
        """
        Get all the types for actins at the barbed end of a filament
        """
        result = []
        for i in range(1, 4):
            result.append(f"actin#barbed_{i}")
            result.append(f"actin#barbed_ATP_{i}")
        return result

    @staticmethod
    def _branch_actin_types():
        """
        Get all the types for actins at the pointed end of a branch
        """
        return [
            "actin#branch_1",
            "actin#branch_ATP_1",
            "actin#branch_barbed_1",
            "actin#branch_barbed_ATP_1",
        ]

    @staticmethod
    def _filamentous_actin_types():
        """
        Get all the types for actins in a filament
        """
        return (
            ActinAnalyzer._pointed_actin_types()
            + ActinAnalyzer._middle_actin_types()
            + ActinAnalyzer._barbed_actin_types()
            + ActinAnalyzer._branch_actin_types()
        )

    @staticmethod
    def _ATP_actin_types():
        """
        Get all the types for actins with ATP bound
        """
        result = []
        for i in range(1, 4):
            result.append(f"actin#ATP_{i}")
            result.append(f"actin#pointed_ATP_{i}")
            result.append(f"actin#barbed_ATP_{i}")
        result.append("actin#branch_ATP_1")
        result.append("actin#branch_barbed_ATP_1")
        return result

    @staticmethod
    def _ADP_actin_types():
        """
        Get all the types for actins with ADP bound
        """
        result = []
        for i in range(1, 4):
            result.append(f"actin#{i}")
            result.append(f"actin#pointed_{i}")
            result.append(f"actin#barbed_{i}")
        result.append("actin#branch_1")
        result.append("actin#branch_barbed_1")
        return result

    @staticmethod
    def _get_frame_filaments_from_start_actins(start_actin_ids, frame_particle_data):
        """
        Get a list of filaments in the given frame of data
        starting from each of the start_actin_ids.
        Each filament is a list of the actin ids in the filament
        in order from pointed to barbed end
        """
        result = []
        non_start_actin_types = (
            ActinAnalyzer._middle_actin_types() + ActinAnalyzer._barbed_actin_types()
        )
        for start_actin_id in start_actin_ids:
            result.append(
                ReaddyUtil.analyze_frame_get_chain_of_types(
                    start_actin_id,
                    non_start_actin_types,
                    frame_particle_data,
                    result=[start_actin_id],
                )
            )
        return result

    @staticmethod
    def _frame_mother_filaments(frame_particle_data):
        """
        Get a list of mother filaments in the given frame of data,
        each filament is a list of the actin ids in the filament
        in order from pointed to barbed end
        """
        return ActinAnalyzer._get_frame_filaments_from_start_actins(
            ReaddyUtil.analyze_frame_get_ids_for_types(
                ActinAnalyzer._pointed_actin_types(), frame_particle_data
            ),
            frame_particle_data,
        )

    @staticmethod
    def _frame_daughter_filaments(frame_particle_data):
        """
        Get a list of daughter filaments in the given frame of data,
        each filament is a list of the actin ids in the filament
        in order from pointed to barbed end
        """
        return ActinAnalyzer._get_frame_filaments_from_start_actins(
            ReaddyUtil.analyze_frame_get_ids_for_types(
                ActinAnalyzer._branch_actin_types(), frame_particle_data
            ),
            frame_particle_data,
        )

    def analyze_ratio_of_filamentous_to_total_actin(self):
        """
        Get a list of the ratio of actin in filaments to total actin over time
        """
        result = []
        for t in range(len(self.particle_data)):
            free_actin = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ActinAnalyzer._free_actin_types(), self.particle_data[t]
                )
            )
            filamentous_actin = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ActinAnalyzer._filamentous_actin_types(), self.particle_data[t]
                )
            )
            if free_actin + filamentous_actin > 0:
                result.append(filamentous_actin / float(free_actin + filamentous_actin))
            else:
                result.append(0)
        return result

    def analyze_ratio_of_ATP_actin_to_total_actin(self):
        """
        Get a list of the ratio of ATP-actin to total actin in filaments over time
        """
        result = []
        for t in range(len(self.particle_data)):
            ATP_actin = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ActinAnalyzer._ATP_actin_types(), self.particle_data[t]
                )
            )
            ADP_actin = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    ActinAnalyzer._ADP_actin_types(), self.particle_data[t]
                )
            )
            if ADP_actin + ATP_actin > 0:
                result.append(ATP_actin / float(ADP_actin + ATP_actin))
            else:
                result.append(1.0)
        return result

    def analyze_ratio_of_daughter_filament_actin_to_total_filamentous_actin(self):
        """
        Get a list of the ratio
        [daughter filament actin] / [filamentous actin] over time
        """
        result = []
        for t in range(len(self.particle_data)):
            mother_actin = 0
            mother_filaments = ActinAnalyzer._frame_mother_filaments(
                self.particle_data[t]
            )
            for mother_filament in mother_filaments[t]:
                mother_actin += len(mother_filament)
            daughter_actin = 0
            daughter_filaments = ActinAnalyzer._frame_daughter_filaments(
                self.particle_data[t]
            )
            for daughter_filament in daughter_filaments[t]:
                daughter_actin += len(daughter_filament)
            if mother_actin + daughter_actin > 0:
                result.append(daughter_actin / float(mother_actin + daughter_actin))
            else:
                result.append(0)
        return result

    def analyze_mother_filament_lengths(self):
        """
        Get a list of the length of mother filaments in each frame of the trajectory
        """
        result = []
        for t in range(len(self.particle_data)):
            mother_filaments = ActinAnalyzer._frame_mother_filaments(
                self.particle_data[t]
            )
            result.append([])
            for filament in mother_filaments:
                result[t].append(len(filament))
        return result

    @staticmethod
    def analyze_average_over_time(data):
        """
        Get a list of the average per time frame of the given data
        """
        return np.mean(data, axis=1)

    def analyze_daughter_filament_lengths(self):
        """
        Get a list of the length of daughter filaments in each frame of the trajectory
        """
        result = []
        for t in range(len(self.particle_data)):
            daughter_filaments = ActinAnalyzer._frame_daughter_filaments(
                self.particle_data[t]
            )
            result.append([])
            for filament in daughter_filaments:
                result[t].append(len(filament))
        return result

    def analyze_ratio_of_bound_to_total_arp23(self):
        """
        Get a list of the ratio of bound to total arp2/3 complexes over time
        """
        result = []
        for t in range(len(self.particle_data)):
            bound_arp23 = 0
            free_arp23 = 0
            arp3_ids = ReaddyUtil.analyze_frame_get_ids_for_types(
                ["arp3", "arp3#branched"], self.particle_data[t]
            )
            for arp3_id in arp3_ids:
                if len(self.particle_data[t][arp3_id][1]) > 1:
                    bound_arp23 += 1
                if len(self.particle_data[t][arp3_id][1]) <= 1:
                    free_arp23 += 1
            if free_arp23 + bound_arp23 > 0:
                result.append(bound_arp23 / float(free_arp23 + bound_arp23))
            else:
                result.append(0)
        return result

    def analyze_ratio_of_capped_ends_to_total_ends(self):
        """
        Get a list of the ratio of barbed ends capped
        with capping protein to all barbed ends over time
        """
        capped_end_types = ["cap#bound"]
        growing_end_types = ActinAnalyzer._barbed_actin_types() + [
            "actin#branch_barbed_1",
            "actin#branch_barbed_ATP_1",
        ]
        result = []
        for t in range(len(self.particle_data)):
            capped_ends = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    capped_end_types, self.particle_data[t]
                )
            )
            growing_ends = len(
                ReaddyUtil.analyze_frame_get_ids_for_types(
                    growing_end_types, self.particle_data[t]
                )
            )
            if growing_ends + capped_ends > 0:
                result.append(capped_ends / float(growing_ends + capped_ends))
            else:
                result.append(0)
        return result

    @staticmethod
    def _get_axis_position_for_actin(frame_particle_data, actin_ids):
        """
        get the position on the filament axis closest to an actin
        actin_ids = [previous actin id, this actin id, next actin id]
        """
        positions = []
        for i in range(4):
            positions.append(frame_particle_data[actin_ids[i]][2])
        ActinUtil.get_actin_axis_position(positions)

    @staticmethod
    def _get_frame_branch_ids(frame_particle_data):
        """
        for each branch point at a time frame, get list of ids for (in order):
        - [0,1,2,3] 4 actins after branch on main filament
          (toward barbed end, ordered from pointed end toward barbed end)
        - [4,5,6,7] first 4 actins on branch
          (ordered from pointed end toward barbed end)
        """
        arp2_ids = ReaddyUtil.analyze_frame_get_ids_for_types(
            ["arp2#branched"], frame_particle_data
        )
        actin_types = (
            ActinAnalyzer._pointed_actin_types()
            + ActinAnalyzer._middle_actin_types()
            + ActinAnalyzer._barbed_actin_types()
        )
        result = []
        for arp2_id in arp2_ids:
            actin1_id = ReaddyUtil.analyze_frame_get_id_for_neighbor_of_types(
                arp2_id, ActinAnalyzer._branch_actin_types(), frame_particle_data
            )
            if actin1_id is None:
                raise Exception(
                    "couldn't parse branch point: failed to find branch actin"
                )
            branch_actins = ReaddyUtil.analyze_frame_get_chain_of_types(
                actin1_id, actin_types, frame_particle_data, 3, arp2_id, [actin1_id]
            )
            if len(branch_actins) < 4:
                print(
                    f"couldn't parse branch point: only found \
                        {len(branch_actins)} daughter actins"
                )
                continue
            actin_arp2_id = ReaddyUtil.analyze_frame_get_id_for_neighbor_of_types(
                arp2_id, actin_types, frame_particle_data, [actin1_id]
            )
            if actin_arp2_id is None:
                raise Exception(
                    "couldn't parse branch point: failed to find actin_arp2"
                )
            n = ReaddyUtil.calculate_polymer_number(
                int(frame_particle_data[actin_arp2_id][0][-1:]), -1
            )
            actin_arp3_types = [
                f"actin#{n}",
                f"actin#ATP_{n}",
                f"actin#pointed_{n}",
                f"actin#pointed_ATP_{n}",
                f"actin#branch_{n}",
                f"actin#branch_ATP_{n}",
            ]
            actin_arp3_id = ReaddyUtil.analyze_frame_get_id_for_neighbor_of_types(
                actin_arp2_id, actin_arp3_types, frame_particle_data
            )
            if actin_arp3_id is None:
                raise Exception(
                    "couldn't parse branch point: failed to find actin_arp3"
                )
            main_actins = ReaddyUtil.analyze_frame_get_chain_of_types(
                actin_arp3_id,
                actin_types,
                frame_particle_data,
                2,
                actin_arp2_id,
                [actin_arp2_id, actin_arp3_id],
            )
            if len(main_actins) < 4:
                print(
                    f"couldn't parse branch point: only found \
                    {len(main_actins)} mother actins"
                )
            result.append(main_actins + branch_actins)
        return result

    @staticmethod
    def _get_frame_branch_angles(frame_particle_data):
        """
        get the angle between mother and daughter filament
        at each branch point in the given frame of the trajectory
        """
        branch_ids = ActinAnalyzer._get_frame_branch_ids(frame_particle_data)
        result = []
        for branch in branch_ids:
            main_pos1 = ActinAnalyzer._get_axis_position_for_actin(
                frame_particle_data, [branch[0], branch[1], branch[2]]
            )
            if main_pos1 is None or ReaddyUtil.vector_is_invalid(main_pos1):
                raise Exception("Failed to axis position for mother actin 1")
            main_pos2 = ActinAnalyzer._get_axis_position_for_actin(
                frame_particle_data, [branch[1], branch[2], branch[3]]
            )
            if main_pos2 is None or ReaddyUtil.vector_is_invalid(main_pos2):
                raise Exception("Failed to axis position for mother actin 2")
            v_main = ReaddyUtil.normalize(main_pos2 - main_pos1)
            branch_pos1 = ActinAnalyzer._get_axis_position_for_actin(
                frame_particle_data, [branch[4], branch[5], branch[6]]
            )
            if branch_pos1 is None or ReaddyUtil.vector_is_invalid(branch_pos1):
                raise Exception("Failed to axis position for daughter actin 1")
            branch_pos2 = ActinAnalyzer._get_axis_position_for_actin(
                frame_particle_data, [branch[5], branch[6], branch[7]]
            )
            if branch_pos2 is None or ReaddyUtil.vector_is_invalid(branch_pos2):
                raise Exception("Failed to axis position for daughter actin 2")
            v_branch = ReaddyUtil.normalize(branch_pos2 - branch_pos1)
            result.append(ReaddyUtil.get_angle_between_vectors(v_main, v_branch, True))
        return result

    def analyze_branch_angles(self):
        """
        Get a list of the angles between mother and daughter filaments
        at each branch point in each frame of the trajectory
        """
        result = []
        for t in range(len(self.particle_data)):
            branch_angles = ActinAnalyzer._get_frame_branch_angles(
                self.particle_data[t]
            )
            result.append(branch_angles)
        return result

    @staticmethod
    def _calculate_pitch(frame_particle_data, actin1_ids, actin2_ids, box_size):
        """
        Calculate the pitch of the helix between two actins
        actin_ids = [previous actin id, this actin id, next actin id]
        for each of the two actins
        """
        actin1_pos = frame_particle_data[actin1_ids[1]][2]
        actin1_axis_pos = ActinAnalyzer._get_axis_position_for_actin(
            frame_particle_data, actin1_ids
        )
        if actin1_axis_pos is None or ReaddyUtil.vector_is_invalid(actin1_axis_pos):
            raise Exception("Failed to axis position for actin 1")
        v1 = ReaddyUtil.normalize(actin1_axis_pos - actin1_pos)
        actin2_pos = frame_particle_data[actin2_ids[1]][2]
        actin2_axis_pos = ActinAnalyzer._get_axis_position_for_actin(
            frame_particle_data, actin2_ids
        )
        if actin2_axis_pos is None or ReaddyUtil.vector_is_invalid(actin2_axis_pos):
            raise Exception("Failed to axis position for actin 2")
        v2 = ReaddyUtil.normalize(actin2_axis_pos - actin2_pos)
        length = np.linalg.norm(actin2_axis_pos - actin1_axis_pos)
        if length > box_size / 2:
            print("filament crossed periodic boundary, skipping pitch calculation")
            return None
        angle = ReaddyUtil.get_angle_between_vectors(v1, v2, True)
        return (360.0 / angle) * length

    @staticmethod
    def _get_frame_short_helix_pitches(frame_particle_data, box_size):
        """
        Get the pitch of the short helix between all actins on each filament
        for a given frame of data
        """
        result = []
        filaments = ActinAnalyzer.all_filaments(frame_particle_data)
        for filament in filaments:
            for i in range(1, len(filament) - 3):
                short_pitch = ActinAnalyzer._calculate_pitch(
                    frame_particle_data,
                    [filament[i - 1], filament[i], filament[i + 1]],
                    [filament[i], filament[i + 1], filament[i + 2]],
                    box_size,
                )
                if short_pitch is not None:
                    result.append(short_pitch)
        return result

    @staticmethod
    def _get_frame_long_helix_pitches(frame_particle_data, box_size):
        """
        Get the pitch of the long helix between all actins on each filament
        for a given frame of data
        """
        result = []
        filaments = ActinAnalyzer.all_filaments(frame_particle_data)
        for filament in filaments:
            for i in range(1, len(filament) - 3):
                long_pitch = ActinAnalyzer._calculate_pitch(
                    frame_particle_data,
                    [filament[i - 1], filament[i], filament[i + 1]],
                    [filament[i + 1], filament[i + 2], filament[i + 3]],
                    box_size,
                )
                if long_pitch is not None:
                    result.append(long_pitch)
        return result

    def analyze_short_helix_pitches(self):
        """
        Get a list of the pitch of short helices between all actins
        on each filament in each frame of the trajectory
        """
        result = []
        for t in range(len(self.particle_data)):
            helix_pitches = ActinAnalyzer._get_frame_short_helix_pitches(
                self.particle_data[t], self.box_size
            )
            result.append(helix_pitches)
        return result

    def analyze_long_helix_pitches(self):
        """
        Get a list of the pitch of long helices between all actins
        on each filament in each frame of the trajectory
        """
        result = []
        for t in range(len(self.particle_data)):
            helix_pitches = ActinAnalyzer._get_frame_long_helix_pitches(
                self.particle_data[t], self.box_size
            )
            result.append(helix_pitches)
        return result

    @staticmethod
    def _calculate_line(points, length):
        """
        Use singular value decomposition (first PCA component)
        to calculate a best fit vector along the 3D points
        """
        center = np.mean(points, axis=0)
        uu, dd, vv = np.linalg.svd(points - center)
        return np.array(
            [center - (length / 2.0) * vv[0], center + (length / 2.0) * vv[0]]
        )

    @staticmethod
    def _get_closest_point_on_line(line, point):
        """
        Get the point on the line closest to the given point
        """
        lineDir = ReaddyUtil.normalize(line[1] - line[0])
        v = point - line[0]
        d = np.dot(v, lineDir)
        return line[0] + d * lineDir

    @staticmethod
    def _get_frame_distance_from_straight(frame_particle_data, box_size):
        """
        Get the distance from each actin axis position to the ideal axis position
        if the filament axis was a straight line
        """
        result = []
        filaments = ActinAnalyzer.all_filaments(frame_particle_data)
        for filament in filaments:
            positions = []
            last_pos = frame_particle_data[filament[0]][2]
            for i in range(1, len(filament) - 1):
                axis_pos = ActinAnalyzer._get_axis_position_for_actin(
                    frame_particle_data, [filament[i - 1], filament[i], filament[i + 1]]
                )
                if axis_pos is None or ReaddyUtil.vector_is_invalid(axis_pos):
                    raise Exception(
                        f"Failed to axis position for actin[{i}] on filament {filament}"
                    )
                if np.linalg.norm(axis_pos - last_pos) < box_size / 2.0:
                    print(
                        "filament crossed periodic boundary, \
                        skipping straightness calculation"
                    )
                    continue
                positions.append(axis_pos)
                last_pos = axis_pos
            if len(positions) > 2:
                axis = ActinAnalyzer._calculate_line(
                    np.squeeze(np.array(positions)), box_size
                )
                for pos in positions:
                    line_pos = ActinAnalyzer._get_closest_point_on_line(axis, pos)
                    result.append(np.linalg.norm(line_pos - pos))
        return result

    def analyze_filament_straightness(self):
        """
        Get a list of the distances from each actin axis position
        to the ideal axis position on each filament in each frame of the trajectory
        """
        result = []
        for t in range(len(self.particle_data)):
            straightness = ActinAnalyzer._get_frame_distance_from_straight(
                self.particle_data[t], self.box_size
            )
            result.append(straightness)
        return result

    @staticmethod
    def _total_reactions():
        """
        [total reaction name] : (readdy reactions to add, readdy reactions to subtract)
        """
        return {
            "dimerize": (["Dimerize"], []),
            "rev dimerize": (["Reverse_Dimerize"], ["Fail_Reverse_Dimerize"]),
            "trimerize": (["Trimerize1", "Trimerize2", "Trimerize3"], []),
            "rev trimerize": (["Reverse_Trimerize"], ["Fail_Reverse_Trimerize"]),
            "pointed growth ATP": (
                [
                    "Pointed_Growth_ATP11",
                    "Pointed_Growth_ATP12",
                    "Pointed_Growth_ATP13",
                    "Pointed_Growth_ATP21",
                    "Pointed_Growth_ATP22",
                    "Pointed_Growth_ATP23",
                ],
                [],
            ),
            "pointed growth ADP": (
                [
                    "Pointed_Growth_ADP11",
                    "Pointed_Growth_ADP12",
                    "Pointed_Growth_ADP13",
                    "Pointed_Growth_ADP21",
                    "Pointed_Growth_ADP22",
                    "Pointed_Growth_ADP23",
                ],
                [],
            ),
            "pointed shrink ATP": (["Pointed_Shrink_ATP"], ["Fail_Pointed_Shrink_ATP"]),
            "pointed shrink ADP": (["Pointed_Shrink_ADP"], ["Fail_Pointed_Shrink_ADP"]),
            "barbed growth ATP": (
                [
                    "Barbed_Growth_ATP11",
                    "Barbed_Growth_ATP12",
                    "Barbed_Growth_ATP13",
                    "Barbed_Growth_ATP21",
                    "Barbed_Growth_ATP22",
                    "Barbed_Growth_ATP23",
                    "Barbed_Growth_Nucleate_ATP1",
                    "Barbed_Growth_Nucleate_ATP2",
                    "Barbed_Growth_Nucleate_ATP3",
                    "Barbed_Growth_Branch_ATP",
                ],
                [],
            ),
            "barbed growth ADP": (
                [
                    "Barbed_Growth_ADP11",
                    "Barbed_Growth_ADP12",
                    "Barbed_Growth_ADP13",
                    "Barbed_Growth_ADP21",
                    "Barbed_Growth_ADP22",
                    "Barbed_Growth_ADP23",
                    "Barbed_Growth_Nucleate_ADP1",
                    "Barbed_Growth_Nucleate_ADP2",
                    "Barbed_Growth_Nucleate_ADP3",
                    "Barbed_Growth_Branch_ADP",
                ],
                [],
            ),
            "barbed shrink ATP": (["Barbed_Shrink_ATP"], ["Fail_Barbed_Shrink_ATP"]),
            "barbed shrink ADP": (["Barbed_Shrink_ADP"], ["Fail_Barbed_Shrink_ADP"]),
            "hydrolyze actin": (["Hydrolysis_Actin"], ["Fail_Hydrolysis_Actin"]),
            "hydrolyze arp": (["Hydrolysis_Arp"], ["Fail_Hydrolysis_Arp"]),
            "nucleotide exchange actin": (["Nucleotide_Exchange_Actin"], []),
            "nucleotide exchange arp": (["Nucleotide_Exchange_Arp"], []),
            "arp2/3 bind ATP": (
                [
                    "Arp_Bind_ATP11",
                    "Arp_Bind_ATP12",
                    "Arp_Bind_ATP13",
                    "Arp_Bind_ATP21",
                    "Arp_Bind_ATP22",
                    "Arp_Bind_ATP23",
                ],
                ["Fail_Arp_Bind_ATP"],
            ),
            "arp2/3 bind ADP": (
                [
                    "Arp_Bind_ADP11",
                    "Arp_Bind_ADP12",
                    "Arp_Bind_ADP13",
                    "Arp_Bind_ADP21",
                    "Arp_Bind_ADP22",
                    "Arp_Bind_ADP23",
                ],
                ["Fail_Arp_Bind_ADP"],
            ),
            "debranch ATP": (["Debranch_ATP"], ["Fail_Debranch_ATP"]),
            "debranch ADP": (["Debranch_ADP"], ["Fail_Debranch_ADP"]),
            "arp2 unbind ATP": (["Arp_Unbind_ATP"], ["Fail_Arp_Unbind_ATP"]),
            "arp2 unbind ADP": (["Arp_Unbind_ADP"], ["Fail_Arp_Unbind_ADP"]),
            "cap bind": (
                [
                    "Cap_Bind11",
                    "Cap_Bind12",
                    "Cap_Bind13",
                    "Cap_Bind21",
                    "Cap_Bind22",
                    "Cap_Bind23",
                ],
                [],
            ),
            "cap unbind": (["Cap_Unbind"], ["Fail_Cap_Unbind"]),
        }

    @staticmethod
    def _get_reaction_type(readdy_reaction_name, reactions):
        """
        Get the type of ReaDDy reaction for a ReaDDy reaction name
        """
        if readdy_reaction_name in reactions["reactions"]:
            return "reactions"
        if readdy_reaction_name in reactions["structural_topology_reactions"]:
            return "structural_topology_reactions"
        if readdy_reaction_name in reactions["spatial_topology_reactions"]:
            return "spatial_topology_reactions"
        raise Exception(f"couldn't find reaction named {readdy_reaction_name}")

    @staticmethod
    def _get_readdy_reaction_events_over_time(
        readdy_reaction_name, multiplier, stride, result, reactions
    ):
        """
        Get a list of the number of times a ReaDDy reaction
        has happened by each time step
        """
        reaction_type = ActinAnalyzer._get_reaction_type(readdy_reaction_name)
        count = 0
        for t in range(len(reactions[reaction_type][readdy_reaction_name])):
            count += multiplier * reactions[reaction_type][readdy_reaction_name][t]
            if t % stride == 0:
                i = int(math.floor(t / float(stride)))
                if len(result) < i + 1:
                    result.append(0)
                result[i] += count
        return result

    def analyze_reaction_events_over_time(self, total_reaction_name):
        """
        Get a list of the number of times a set
        of ReaDDy reactions has happened by each time step
        """
        if self.reactions is None:
            reaction_times, self.reactions = self.traj.read_observable_reaction_counts()
        readdy_reactions = ActinAnalyzer._total_reactions()[total_reaction_name]
        result = []
        for reaction_name in readdy_reactions[0]:
            result = ActinAnalyzer._get_readdy_reaction_events_over_time(
                reaction_name, 1, 1, result, self.reactions
            )
        for reaction_name in readdy_reactions[1]:
            result = ActinAnalyzer._get_readdy_reaction_events_over_time(
                reaction_name, -1, 1, result, self.reactions
            )
        return result

    def analyze_all_reaction_events_over_time(self):
        """
        Get a dictionary of lists of the number of times
        a set of ReaDDy reactions has happened by each time step
        for each total reaction
        """
        result = {}
        i = 0
        total_reactions = ActinAnalyzer._total_reactions()
        for total_reaction_name in total_reactions:
            result[total_reaction_name] = self.analyze_reaction_events_over_time(
                total_reaction_name
            )
            sys.stdout.write("\r")
            p = 100.0 * (i + 1) / float(len(total_reactions))
            sys.stdout.write(
                "Analyzing reactions [{}{}] {}%".format(
                    "=" * int(round(p)),
                    " " * int(100.0 - round(p)),
                    round(10.0 * p) / 10.0,
                )
            )
            sys.stdout.flush()
            i += 1
        return result

    def analyze_free_actin_concentration_over_time(self):
        """
        Get a list of the concentration of free actin at each step
        """
        result = []
        for t in range(len(self.particle_data)):
            result.append(
                ReaddyUtil.calculate_concentration(
                    len(
                        ReaddyUtil.analyze_frame_get_ids_for_types(
                            ActinAnalyzer._free_actin_types(), self.particle_data[t]
                        )
                    ),
                    self.box_size,
                )
            )
        return result
