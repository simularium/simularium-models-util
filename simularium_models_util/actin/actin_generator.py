#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

from ..common import ReaddyUtil


class ActinGenerator:
    """
    Generates positions, types, and edges for monomers in actin networks
    """

    @staticmethod
    def get_actin_number(actin_number, offset):
        """
        get the type number for an actin plus the given offset in range [-1, 1]
        (i.e. return 3 for type = "actin#ATP_1" and offset = -1)
        """
        if offset < -1 or offset > 1:
            raise Exception("Offset for actin number is not in [-1, 1]")
        n = actin_number + offset
        if n > 3:
            n -= 3
        if n < 1:
            n += 3
        return n

    @staticmethod
    def check_shift_branch_actin_numbers(type_names):
        """
        if the first actin's number is not 2, 
        shift the branch's actin numbers so that it is
        """
        if "2" not in type_names[0]:
            result = []
            n = int(type_names[0][-1:])
            offset = n - 2
            for i in range(len(type_names)):
                new_n = n - offset
                if new_n > 3:
                    new_n -= 3
                result.append(f"{type_names[i][:-1]}{new_n}")
                n += 1
                if n > 3:
                    n = 1
            return result
        return type_names 

    @staticmethod
    def get_actins_for_linear_fiber(
        fiber, start_normal, start_axis_pos, direction, 
        offset_vector, pointed_actin_number
    ):  
        '''
        get actin monomer data pointed to barbed for a fiber with no daughter branches
        '''
        normal = np.copy(start_normal)
        axis_pos = fiber.get_nearest_position(np.copy(start_axis_pos))
        monomer_positions = []
        monomer_types = []
        monomer_edges = []
        # get actin positions
        fiber_points = fiber.reversed_points() if direction < 0 else fiber.points
        start_index = fiber.get_index_of_curve_start_point(axis_pos, direction < 0)
        actin_offset_from_axis = ActinStructure.actin_distance_from_axis()
        for i in range(start_index, len(fiber_points) - 1):
            while (np.linalg.norm(axis_pos - fiber_points[i + 1].position) >= 
                ActinStructure.actin_to_actin_axis_distance
            ):
                axis_pos += ActinStructure.actin_to_actin_axis_distance * direction * fiber_points[i].tangent
                normal = ReaddyUtil.rotate(normal, fiber_points[i].tangent, direction * ActinStructure.actin_to_actin_axis_angle())
                monomer_positions.append(axis_pos + actin_offset_from_axis * normal + offset_vector)
        # get actin types and edges
        actin_number = pointed_actin_number
        for i in range(len(monomer_positions)):
            monomer_types.append(f"actin#ATP_{actin_number}")
            actin_number = ActinGenerator.get_actin_number(actin_number, 1)
            if len(monomer_positions) > 1 and i < len(monomer_positions) - 1:
                monomer_edges.append((i, i + 1))
        if direction < 0:
            monomer_types.reverse()
        return monomer_positions, monomer_types, monomer_edges, actin_number

    @staticmethod
    def add_bound_arp_monomers(
        monomer_positions, monomer_types, monomer_edges, 
        fiber, arp, mother_barbed_index, branch_indices
    ):  
        """
        add positions, types, and edges for a bound arp2 and arp3
        """
        closest_actin_index = arp.get_closest_actin_index(
            monomer_positions, monomer_types, branch_indices, mother_barbed_index)
        if closest_actin_index < 0:
            return monomer_positions, monomer_types, monomer_edges
        # positions
        monomer_positions.append(
            arp.get_bound_monomer_position(monomer_positions[closest_actin_index], fiber, "arp2"))
        monomer_positions.append(
            arp.get_bound_monomer_position(monomer_positions[closest_actin_index], fiber, "arp3"))
        # types
        monomer_types.append("arp2") 
        monomer_types.append("arp3#ATP") 
        # edges
        arp2_index = len(monomer_positions) - 2
        arp3_index = len(monomer_positions) - 1
        monomer_edges.append((closest_actin_index, arp2_index))
        monomer_edges.append((closest_actin_index + 1, arp3_index))
        monomer_edges.append((arp2_index, arp3_index))
        return monomer_positions, monomer_types, monomer_edges

    @staticmethod
    def get_nucleated_arp_monomer_positions(mother_fiber, nucleated_arp):   
        """
        get actin positions pointed to barbed for a branch
        """
        # get ideal monomer positions near the arp
        monomer_positions = []
        arp_mother_pos = mother_fiber.get_nearest_position(nucleated_arp.position)
        v_mother = mother_fiber.get_nearest_segment_direction(nucleated_arp.position)
        v_daughter = nucleated_arp.daughter_fiber.get_nearest_segment_direction(nucleated_arp.position)
        monomer_positions.append(
            arp_mother_pos + nucleated_arp.get_local_nucleated_monomer_position(v_mother, v_daughter, "actin_arp2"))
        monomer_positions.append(
            arp_mother_pos + nucleated_arp.get_local_nucleated_monomer_position(v_mother, v_daughter, "arp2"))
        monomer_positions.append(
            arp_mother_pos + nucleated_arp.get_local_nucleated_monomer_position(v_mother, v_daughter, "arp3"))
        monomer_positions.append(
            arp_mother_pos + nucleated_arp.get_local_nucleated_monomer_position(v_mother, v_daughter, "actin1"))
        # # rotate them to match the actual branch angle
        # branch_angle = ReaddyUtil.get_angle_between_vectors(v_mother, v_daughter)
        # branch_normal = ReaddyUtil.normalize(monomer_positions[0] - arp_mother_pos)
        # for i in range(1,len(monomer_positions)):
        #     monomer_positions[i] = Arp.rotate_position_to_match_branch_angle(
        #         monomer_positions[i], branch_angle, arp_mother_pos, branch_normal)
        # # translate them 2nm since the mother and daughter axes don't actually intersect
        # v_branch_shift = ActinStructure.branch_shift() * ReaddyUtil.normalize(monomer_positions[0] - arp_mother_pos)
        # for i in range(len(monomer_positions)):
        #     monomer_positions[i] = monomer_positions[i] + v_branch_shift
        return monomer_positions, np.zeros(3)

    @staticmethod
    def get_monomers_for_branching_fiber(monomer_positions, monomer_types, monomer_edges, fiber, offset_vector, pointed_actin_number): 
        '''
        recursively get all the monomer data for the given branching fiber
        '''    
        daughter_positions = []
        daughter_types = []
        daughter_edges = []
        current_index = 0
        branch_indices = []
        actin_number = pointed_actin_number
        for a in range(len(fiber.nucleated_arps)):
            arp = fiber.nucleated_arps[a]
            # add arp2, arp3, and daughter actin bound to arp3
            fork_positions, v_branch_shift = ActinGenerator.get_nucleated_arp_monomer_positions(fiber, arp)
            daughter_positions.append(fork_positions[1:])
            daughter_types.append(["arp2#branched", "arp3#ATP", "actin#branch_ATP_1"])
            daughter_edges.append([(0, 1), (0, 2)])
            # add the rest of the daughter monomers on this branch
            axis_pos = arp.daughter_fiber.get_nearest_position(fork_positions[3])
            branch_positions, branch_types, branch_edges = ActinGenerator.get_monomers_for_fiber(
                arp.daughter_fiber, ReaddyUtil.normalize(fork_positions[3] - axis_pos), axis_pos, 
                offset_vector + v_branch_shift, 2)
            daughter_positions[a] += branch_positions
            daughter_types[a] += branch_types
            if len(branch_positions) > 0:
                daughter_edges[a].append((2, 3))
                for edge in branch_edges:
                    daughter_edges[a].append((edge[0] + 3, edge[1] + 3))
            # get mother monomer positions toward pointed end
            actin_arp2_axis_pos = fiber.get_nearest_position(fork_positions[0])
            actin_arp2_normal = ReaddyUtil.normalize(fork_positions[0] - actin_arp2_axis_pos)
            if a == 0: # if this is the first nucleated arp
                # get the rest of the fiber
                pointed_positions, pointed_types, pointed_edges, actin_number = ActinGenerator.get_actins_for_linear_fiber(
                    fiber, actin_arp2_normal, actin_arp2_axis_pos, -1, offset_vector, actin_number)
                pointed_positions.reverse()
                monomer_positions += pointed_positions
                pointed_types.reverse()
                # if fiber.mother_arp is not None:
                #     pointed_types = ActinGenerator.check_shift_branch_actin_numbers(pointed_types)
                #     actin_number = int(pointed_types[len(pointed_types) - 1][-1]) + 1
                pointed_types[0] = f"actin#branch_ATP_{pointed_types[0][-1:]}"
                monomer_types += pointed_types
                monomer_edges += pointed_edges
                current_index += len(pointed_positions)
                if len(pointed_positions) > 0:
                    monomer_edges.append((current_index - 1, current_index))
            # add the actin monomer attached to the branch's arp2
            monomer_positions.append(fork_positions[0])
            monomer_types.append(f"actin#ATP_{actin_number}")
            actin_number = ActinGenerator.get_actin_number(actin_number, 1)
            branch_indices.append(current_index)
            # get monomers pointed to barbed until next arp
            barbed_positions, barbed_types, barbed_edges, actin_number = ActinGenerator.get_actins_for_linear_fiber(
                fiber, actin_arp2_normal, actin_arp2_axis_pos, 1, offset_vector, actin_number)
            monomer_positions += barbed_positions
            barbed_types[-1] = f"actin#barbed_ATP_{barbed_types[-1][-1:]}"
            monomer_types += barbed_types
            if len(barbed_positions) > 0:
                monomer_edges.append((current_index, current_index + 1))
            current_index += 1
            for e in range(len(barbed_edges)):
                monomer_edges.append((barbed_edges[e][0] + current_index, barbed_edges[e][1] + current_index))
            current_index += len(barbed_positions)
        mother_barbed_index = current_index
        for d in range(len(daughter_positions)):
            monomer_positions += daughter_positions[d]
            monomer_types += daughter_types[d]
            monomer_edges.append((branch_indices[d], current_index)) # mother actin to arp2
            monomer_edges.append((branch_indices[d] + 1, current_index + 1)) # mother actin to arp3
            for edge in daughter_edges[d]:
                monomer_edges.append((edge[0] + current_index, edge[1] + current_index))
            current_index += branch_indices[d]
        for a in range(len(fiber.bound_arps)):
            monomer_positions, monomer_types, monomer_edges = ActinGenerator.add_bound_arp_monomers(
                monomer_positions, 
                monomer_types, 
                monomer_edges, 
                fiber, 
                fiber.bound_arps[a], 
                mother_barbed_index, 
                branch_indices
            )
        return monomer_positions, monomer_types, monomer_edges

    @staticmethod
    def get_monomers_for_fiber(fiber, start_normal, start_axis_pos, offset_vector, pointed_actin_number): 
        '''
        recursively get all the monomer data for the given fiber network
        '''    
        monomer_positions = []
        monomer_types = []
        monomer_edges = []
        actin_number = pointed_actin_number
        if len(fiber.nucleated_arps) < 1:
            # fiber has no branches
            new_positions, new_types, new_edges, b = ActinGenerator.get_actins_for_linear_fiber(
                fiber, start_normal, start_axis_pos, 1, offset_vector, actin_number)
            monomer_positions += new_positions
            # if fiber.mother_arp is not None:
            #     new_types = ActinGenerator.check_shift_branch_actin_numbers(new_types)
            new_types[-1] = f"actin#barbed_ATP_{new_types[-1][-1:]}"
            monomer_types += new_types
            monomer_edges += new_edges
            for a in range(len(fiber.bound_arps)):
                monomer_positions, monomer_types, monomer_edges = ActinGenerator.add_bound_arp_monomers(
                    monomer_positions, 
                    monomer_types, 
                    monomer_edges, 
                    fiber, 
                    fiber.bound_arps[a], 
                    len(monomer_positions)-1, 
                    [],
                )
        else:
            # fiber has branches
            monomer_positions, monomer_types, monomer_edges = ActinGenerator.get_monomers_for_branching_fiber(
                monomer_positions, monomer_types, monomer_edges, fiber, offset_vector, actin_number
            )
        return monomer_positions, monomer_types, monomer_edges

    @staticmethod
    def get_monomers(fibers_data):
        """
        get all the monomer data for the (branched) fibers in fibers_data 

        fibers_data: List[FiberData]
        (FiberData for mother fibers only, which should have 
        their daughters' FiberData attached to their nucleated arps)
        """
        result = []
        for fiber_data in fibers_data:
            monomer_positions, monomer_types, edges = ActinGenerator.get_monomers_for_fiber(
                fiber_data, 
                ReaddyUtil.get_random_perpendicular_vector(fiber_data.get_first_segment_direction()), 
                fiber_data.pointed_point().position, 
                np.zeros(3),
                1
            )
            monomer_types[0] = f"actin#pointed_ATP_{monomer_types[0][-1:]}"
            result.append((monomer_types, np.array(monomer_positions), edges))
        return result
