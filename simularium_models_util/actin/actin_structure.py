#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from ..common import ReaddyUtil
from .fiber_data import FiberData
from .curve_point_data import CurvePointData


class ActinStructure:
    """
    Measurements are from Chimera:
      - load PDB 7AQK as model #0
      - duplicate it as model #1
      - from command line run: `matchmaker #0:.k #1:.l`
        to align copied second mother actin to original pointed mother actin
      - run: `measure rotation #0 #1` to measure mother axis
      - duplicate model #0 as model #3
      - from command line run: `matchmaker #0:.h #3:.i`
        to align copied second daughter actin to original first branch daughter actin
      - run: `measure rotation #0 #3` to measure daughter axis
      - run: `measure center #0:.[chain ID]` for a = arp3, b = arp2,
        [h,i,j] = daughter actin, [k, l, m, n, o, p, q, r] = mother actin
      - find these values in the output from the measure commands
        (spatial values are multiplied by 0.1 from Angstroms to nm,
        angle values are in degrees)
    """

    arp3_position = np.array([29.275, 27.535, 23.944])
    arp2_position = np.array([28.087, 30.872, 26.657])
    mother_positions = np.array(
        [
            [19.126, 20.838, 27.757],
            [21.847, 24.171, 27.148],
            [24.738, 20.881, 26.671],
            [27.609, 24.061, 27.598],
            [30.382, 21.190, 25.725],
            [33.374, 23.553, 27.951],
            [36.075, 21.642, 25.060],
            [39.005, 22.861, 27.970],
        ]
    )
    daughter_positions = np.array(
        [
            [29.821, 33.088, 23.356],
            [30.476, 36.034, 26.528],
            [30.897, 38.584, 23.014],
        ]
    )
    mother_axis_direction = np.array([0.99873035, 0.01436705, -0.04828331])
    mother_axis_point = np.array([0.00000000, 22.186136822, 28.273905592])
    daughter_axis_direction = np.array([0.31037692, 0.94885799, -0.05774669])
    daughter_axis_point = np.array([18.640140530, 0.00000000, 27.032325657])
    actin_to_actin_angle_degrees = -(167.44857939 + 166.44837565) / 2.0
    actin_to_actin_axis_distance = (2.795019154 + 2.811178372) / 2.0

    @staticmethod
    def actin_to_actin_axis_angle():  # was -2.91 rad
        return np.deg2rad(ActinStructure.actin_to_actin_angle_degrees)

    @staticmethod
    def mother_fiber():
        position0 = ActinStructure.mother_positions[0]
        v0 = position0 - ActinStructure.mother_axis_point
        d0 = np.dot(v0, ActinStructure.mother_axis_direction)
        axis_position0 = (
            ActinStructure.mother_axis_point + ActinStructure.mother_axis_direction * d0
        )
        position1 = ActinStructure.mother_positions[7]
        v1 = position1 - ActinStructure.mother_axis_point
        d1 = np.dot(v1, ActinStructure.mother_axis_direction)
        axis_position1 = (
            ActinStructure.mother_axis_point + ActinStructure.mother_axis_direction * d1
        )
        arc_length = np.linalg.norm(axis_position1 - axis_position0)
        return FiberData(
            0,
            [
                CurvePointData(axis_position0, ActinStructure.mother_axis_direction, 0),
                CurvePointData(
                    axis_position1, ActinStructure.mother_axis_direction, arc_length
                ),
            ],
        )

    @staticmethod
    def vector_to_axis(): 
        mother_fiber = ActinStructure.mother_fiber()
        axis_position = mother_fiber.get_nearest_position(
            ActinStructure.mother_positions[3]
        )
        return axis_position - ActinStructure.mother_positions[3]

    @staticmethod
    def actin_distance_from_axis():  # was 1.59 nm
        return np.linalg.norm(ActinStructure.vector_to_axis())

    @staticmethod
    def branch_positions():
        """
        get the points on the mother and daughter axes that are closest to each other
        """
        # find unit direction vector for line perpendicular to mother and daughter axes
        perpendicular = np.cross(
            ActinStructure.daughter_axis_direction, ActinStructure.mother_axis_direction
        )
        perpendicular /= np.linalg.norm(perpendicular)
        # solve the system
        RHS = ActinStructure.daughter_axis_point - ActinStructure.mother_axis_point
        LHS = np.array(
            [
                ActinStructure.mother_axis_direction,
                -ActinStructure.daughter_axis_direction,
                perpendicular,
            ]
        ).T
        t = np.linalg.solve(LHS, RHS)
        return [
            ActinStructure.mother_axis_point
            + t[0] * ActinStructure.mother_axis_direction,
            ActinStructure.daughter_axis_point
            + t[1] * ActinStructure.daughter_axis_direction,
        ]

    @staticmethod
    def mother_branch_position():
        return ActinStructure.branch_positions()[0]

    @staticmethod
    def branch_shift():  # was -2.058923 nm
        branch_positions = ActinStructure.branch_positions()
        return np.linalg.norm(branch_positions[1] - branch_positions[0])

    @staticmethod
    def branch_monomer_position(monomer_type):
        """
        get the given monomer position
        """
        if monomer_type == "actin_arp2":
            return ActinStructure.mother_positions[3]
        elif monomer_type == "arp2":
            return ActinStructure.arp2_position
        elif monomer_type == "arp3":
            return ActinStructure.arp3_position
        else:  # actin1
            return ActinStructure.daughter_positions[0]

    @staticmethod
    def branch_angle():  # was 77 deg
        return ReaddyUtil.get_angle_between_vectors(
            ActinStructure.mother_axis_direction, ActinStructure.daughter_axis_direction
        )

    @staticmethod
    def bound_arp_orientation():
        actin_arp2_pos = ActinStructure.mother_positions[3]
        actin_arp2_axis_pos = ActinStructure.mother_fiber().get_nearest_position(
            actin_arp2_pos
        )
        v_actin_arp2 = ReaddyUtil.normalize(actin_arp2_pos - actin_arp2_axis_pos)
        return ReaddyUtil.get_orientation_from_vectors(
            ActinStructure.mother_axis_direction, v_actin_arp2
        )

    @staticmethod
    def nucleated_arp_orientation():
        return ReaddyUtil.get_orientation_from_vectors(
            ActinStructure.mother_axis_direction, ActinStructure.daughter_axis_direction
        )

    @staticmethod
    def actin_to_actin_distance():  # was 4.27 nm
        distances = []
        for i in range(len(ActinStructure.mother_positions) - 1):
            distances.append(
                np.linalg.norm(
                    ActinStructure.mother_positions[i + 1]
                    - ActinStructure.mother_positions[i]
                )
            )
        return np.mean(np.array(distances))

    @staticmethod
    def arp2_to_mother_distance():  # was 4.17 nm
        return np.linalg.norm(
            ActinStructure.mother_positions[3] - ActinStructure.arp2_position
        )

    @staticmethod
    def arp3_to_mother_distance():  # was 8.22 nm
        return np.linalg.norm(
            ActinStructure.mother_positions[4] - ActinStructure.arp3_position
        )

    @staticmethod
    def arp2_to_arp3_distance():  # was 4.18 nm
        return np.linalg.norm(
            ActinStructure.arp3_position - ActinStructure.arp2_position
        )

    @staticmethod
    def arp2_to_daughter_distance():  # arp3 to daughter was 4.19 nm
        return np.linalg.norm(
            ActinStructure.daughter_positions[0] - ActinStructure.arp2_position
        )

    @staticmethod
    def actin_to_actin_angle():  # was 1.48 rad
        angles = []
        for i in range(len(ActinStructure.mother_positions) - 2):
            v1 = (
                ActinStructure.mother_positions[i]
                - ActinStructure.mother_positions[i + 1]
            )
            v2 = (
                ActinStructure.mother_positions[i + 2]
                - ActinStructure.mother_positions[i + 1]
            )
            angles.append(ReaddyUtil.get_angle_between_vectors(v1, v2))
        return np.mean(np.array(angles))

    @staticmethod
    def actin_to_actin_dihedral_angle():  # was 2.80 rad
        angles = []
        for i in range(len(ActinStructure.mother_positions) - 3):
            v1 = (
                ActinStructure.mother_positions[i]
                - ActinStructure.mother_positions[i + 1]
            )
            v2 = (
                ActinStructure.mother_positions[i + 3]
                - ActinStructure.mother_positions[i + 2]
            )
            angles.append(ReaddyUtil.get_angle_between_vectors(v1, v2))
        return np.mean(np.array(angles))

    @staticmethod
    def arp3_arp2_daughter1_angle():  # was 1.43 rad
        v1 = ActinStructure.arp3_position - ActinStructure.arp2_position
        v2 = ActinStructure.daughter_positions[0] - ActinStructure.arp2_position
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def arp2_daughter1_daughter2_angle():  # was 1.46 rad
        v1 = ActinStructure.arp2_position - ActinStructure.daughter_positions[0]
        v2 = ActinStructure.daughter_positions[1] - ActinStructure.daughter_positions[0]
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def mother1_mother2_arp3_angle():  # was 2.15 rad
        v1 = ActinStructure.mother_positions[3] - ActinStructure.mother_positions[4]
        v2 = ActinStructure.arp3_position - ActinStructure.mother_positions[4]
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def mother3_mother2_arp3_angle():  # was 2.31 rad
        v1 = ActinStructure.mother_positions[5] - ActinStructure.mother_positions[4]
        v2 = ActinStructure.arp3_position - ActinStructure.mother_positions[4]
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def mother0_mother1_arp2_angle():  # was 0.91 rad
        v1 = ActinStructure.mother_positions[2] - ActinStructure.mother_positions[3]
        v2 = ActinStructure.arp2_position - ActinStructure.mother_positions[3]
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def actin_to_actin_repulsion_distance():  # was 4 nm
        return 0.95 * ActinStructure.actin_to_actin_distance()

    @staticmethod
    def arp3_arp2_daughter1_daughter2_dihedral_angle():  # was 2.92 rad
        v1 = ActinStructure.arp3_position - ActinStructure.arp2_position
        v2 = ActinStructure.daughter_positions[1] - ActinStructure.daughter_positions[0]
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def arp2_daughter1_daughter2_daughter3_dihedral_angle():  # was 2.82 rad
        v1 = ActinStructure.arp2_position - ActinStructure.daughter_positions[0]
        v2 = ActinStructure.daughter_positions[2] - ActinStructure.daughter_positions[1]
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def mother1_arp2_daughter1_daughter2_dihedral_angle():  # was 2.76 rad
        v1 = ActinStructure.mother_positions[3] - ActinStructure.arp2_position
        v2 = ActinStructure.daughter_positions[1] - ActinStructure.daughter_positions[0]
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def mother1_mother2_arp3_arp2_dihedral_angle():  # was 1.67 rad
        v1 = ActinStructure.mother_positions[3] - ActinStructure.mother_positions[4]
        v2 = ActinStructure.arp2_position - ActinStructure.arp3_position
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def mother0_mother1_arp2_daughter1_dihedral_angle():  # was nothing
        v1 = ActinStructure.mother_positions[2] - ActinStructure.mother_positions[3]
        v2 = ActinStructure.daughter_positions[0] - ActinStructure.arp2_position
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def mother3_mother2_arp3_arp2_dihedral_angle():  # was nothing
        v1 = ActinStructure.mother_positions[5] - ActinStructure.mother_positions[4]
        v2 = ActinStructure.arp2_position - ActinStructure.arp3_position
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def mother2_arp3_arp2_daughter1_dihedral_angle():  # was nothing
        v1 = ActinStructure.mother_positions[4] - ActinStructure.arp3_position
        v2 = ActinStructure.daughter_positions[0] - ActinStructure.arp2_position
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def mother_mother0_mother1_arp2_dihedral_angle():  # was nothing
        v1 = ActinStructure.mother_positions[1] - ActinStructure.mother_positions[2]
        v2 = ActinStructure.arp2_position - ActinStructure.mother_positions[3]
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def arp2_mother1_mother2_arp3_dihedral_angle():  # was nothing
        v1 = ActinStructure.arp2_position - ActinStructure.mother_positions[3]
        v2 = ActinStructure.arp3_position - ActinStructure.mother_positions[4]
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def mother4_mother3_mother2_arp3_dihedral_angle():  # was nothing
        v1 = ActinStructure.mother_positions[6] - ActinStructure.mother_positions[5]
        v2 = ActinStructure.arp3_position - ActinStructure.mother_positions[4]
        return ReaddyUtil.get_angle_between_vectors(v1, v2)

    @staticmethod
    def orientation():
        return ReaddyUtil.get_orientation_from_positions(
            [
                ActinStructure.mother_positions[2],
                ActinStructure.mother_positions[3],
                ActinStructure.mother_positions[4],
            ]
        )

    @staticmethod
    def mother1_to_branch_actin_vectors():
        return [
            ActinStructure.daughter_positions[0] - ActinStructure.mother_positions[3],
            ActinStructure.daughter_positions[1] - ActinStructure.mother_positions[3],
            ActinStructure.daughter_positions[2] - ActinStructure.mother_positions[3],
        ]

    @staticmethod
    def mother1_to_arp3_vector():
        return ActinStructure.arp3_position - ActinStructure.mother_positions[3]

    @staticmethod
    def mother1_to_arp2_vector():
        return ActinStructure.arp2_position - ActinStructure.mother_positions[3]

    @staticmethod
    def mother1_to_mother3_vector():
        return ActinStructure.mother_positions[5] - ActinStructure.mother_positions[3]

    @staticmethod
    def mother1_to_mother_vector():
        return ActinStructure.mother_positions[1] - ActinStructure.mother_positions[3]
