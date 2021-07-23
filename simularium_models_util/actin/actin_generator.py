#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
import math

import numpy as np

from ..common import ReaddyUtil, ParticleData
from .actin_structure import ActinStructure


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
    def check_shift_branch_actin_numbers(particles, particle_ids):
        """
        if the first actin's number is not 2,
        shift the branch's actin numbers so that it is
        """
        first_actin_type = particles[particle_ids[0]].type_name
        if "2" not in first_actin_type:
            n = int(first_actin_type[-1:])
            offset = n - 2
            for i in range(len(particle_ids)):
                new_n = n - offset
                if new_n > 3:
                    new_n -= 3
                type_name = particles[particle_ids[i]].type_name
                particles[particle_ids[i]].type_name = f"{type_name[:-1]}{new_n}"
                n += 1
                if n > 3:
                    n = 1
            return result
        return particles

    @staticmethod
    def get_actins_for_linear_fiber(
        fiber,
        start_normal,
        start_axis_pos,
        direction,
        offset_vector,
        pointed_actin_number,
        particles = {}
    ):
        """
        get actin monomer data pointed to barbed for a fiber with no daughter branches
        """
        normal = np.copy(start_normal)
        axis_pos = fiber.get_nearest_position(np.copy(start_axis_pos))
        particle_ids = []
        # get actin positions
        fiber_points = fiber.reversed_points() if direction < 0 else fiber.points
        fiber_tangents = fiber.reversed_tangents() if direction < 0 else fiber.tangents
        start_index = fiber.get_index_of_curve_start_point(axis_pos, direction < 0)
        actin_offset_from_axis = ActinStructure.actin_distance_from_axis()
        for i in range(start_index, len(fiber_points) - 1):
            while (
                np.linalg.norm(axis_pos - fiber_points[i + 1])
                >= ActinStructure.actin_to_actin_axis_distance
            ):
                axis_pos += (
                    ActinStructure.actin_to_actin_axis_distance
                    * direction
                    * fiber_tangents[i]
                )
                normal = ReaddyUtil.rotate(
                    normal,
                    fiber_tangents[i],
                    direction * ActinStructure.actin_to_actin_axis_angle(),
                )
                new_particle_id = str(uuid.uuid4())
                particle_ids.append(new_particle_id)
                particles[new_particle_id] = ParticleData(
                    unique_id=new_particle_id,
                    position=axis_pos + actin_offset_from_axis * normal + offset_vector,
                    neighbor_ids=[],
                )
        # get actin types and edges
        actin_number = pointed_actin_number
        for i in range(len(particle_ids)):
            particle_id = particle_ids[i]
            particles[particle_id].type_name = f"actin#ATP_{actin_number}"
            actin_number = ActinGenerator.get_actin_number(actin_number, 1)
            if i > 0:
                particles[particle_id].neighbor_ids.append(particle_ids[i - 1])
            if i < len(particle_ids) - 1:
                particles[particle_id].neighbor_ids.append(particle_ids[i + 1])
        if direction < 0:
            particle_ids.reverse()
        return particles, particle_ids, actin_number

    @staticmethod
    def add_bound_arp_monomers(
        particle_ids,
        fiber,
        arp,
        actin_arp_ids,
        particles = {},
    ):
        """
        add positions, types, and edges for a bound arp2 and arp3
        """
        closest_actin_index = arp.get_closest_actin_index(
            particle_ids, actin_arp_ids, particles
        )
        if closest_actin_index < 0:
            return particles, particle_ids
        arp2_id = str(uuid.uuid4())
        arp2_index = len(particle_ids)
        particle_ids.append(arp2_id)
        particles[arp2_id] = ParticleData(
            unique_id=arp2_id,
            type_name="arp2",
            position=arp.get_bound_monomer_position(
                particles[particle_ids[closest_actin_index]].position, fiber, "arp2"
            ),
            neighbor_ids=[particle_ids[closest_actin_index]],
        )
        particles[particle_ids[closest_actin_index]].neighbor_ids.append(arp2_id)
        arp3_id = str(uuid.uuid4())
        arp3_index = len(particle_ids)
        particle_ids.append(arp3_id)
        particles[arp3_id] = ParticleData(
            unique_id=arp3_id,
            type_name="arp3#ATP",
            position=arp.get_bound_monomer_position(
                particles[particle_ids[closest_actin_index]].position, fiber, "arp3"
            ),
            neighbor_ids=[particle_ids[closest_actin_index + 1], arp2_id],
        )
        particles[particle_ids[closest_actin_index + 1]].neighbor_ids.append(arp3_id)
        particles[arp2_id].neighbor_ids.append(arp3_id)
        return particles, particle_ids

    @staticmethod
    def get_nucleated_arp_monomer_positions(mother_fiber, nucleated_arp):
        """
        get actin positions pointed to barbed for a branch
        """
        # get ideal monomer positions near the arp
        monomer_positions = []
        arp_mother_pos = mother_fiber.get_nearest_position(nucleated_arp.position)
        v_mother = mother_fiber.get_nearest_segment_direction(nucleated_arp.position)
        v_daughter = nucleated_arp.daughter_fiber.get_nearest_segment_direction(
            nucleated_arp.position
        )
        monomer_positions.append(
            arp_mother_pos
            + nucleated_arp.get_local_nucleated_monomer_position(
                v_mother, v_daughter, "actin_arp2"
            )
        )
        monomer_positions.append(
            arp_mother_pos
            + nucleated_arp.get_local_nucleated_monomer_position(
                v_mother, v_daughter, "arp2"
            )
        )
        monomer_positions.append(
            arp_mother_pos
            + nucleated_arp.get_local_nucleated_monomer_position(
                v_mother, v_daughter, "arp3"
            )
        )
        monomer_positions.append(
            arp_mother_pos
            + nucleated_arp.get_local_nucleated_monomer_position(
                v_mother, v_daughter, "actin1"
            )
        )
        # # rotate them to match the actual branch angle
        # branch_angle = ReaddyUtil.get_angle_between_vectors(v_mother, v_daughter)
        # branch_normal = ReaddyUtil.normalize(monomer_positions[0] - arp_mother_pos)
        # for i in range(1,len(monomer_positions)):
        #     monomer_positions[i] = Arp.rotate_position_to_match_branch_angle(
        #         monomer_positions[i], branch_angle, arp_mother_pos, branch_normal)
        # # translate them 2nm since the mother and daughter axes don't intersect
        # v_branch_shift = ActinStructure.branch_shift() * ReaddyUtil.normalize(
        #     monomer_positions[0] - arp_mother_pos)
        # for i in range(len(monomer_positions)):
        #     monomer_positions[i] = monomer_positions[i] + v_branch_shift
        return monomer_positions, np.zeros(3)

    @staticmethod
    def get_monomers_for_branching_fiber(
        particle_ids,
        fiber,
        offset_vector,
        pointed_actin_number,
        particles = {},
    ):
        """
        recursively get all the monomer data for the given branching fiber
        """
        fiber_particle_ids = []
        daughter_particle_ids = []
        current_index = 0
        actin_arp_ids = []
        actin_number = pointed_actin_number
        for a in range(len(fiber.nucleated_arps)):
            # first get positions for arp2, arp3, and daughter actin bound to arp 
            # for the first nucleated arp, they constrain the helical position of everything else
            arp = fiber.nucleated_arps[a]
            fork_positions, v_branch_shift = ActinGenerator.get_nucleated_arp_monomer_positions(fiber, arp)
            # get mother monomers if this is the first nucleated arp
            if a == 0:
                actin_arp2_axis_pos = fiber.get_nearest_position(fork_positions[0])
                actin_arp2_normal = ReaddyUtil.normalize(
                    fork_positions[0] - actin_arp2_axis_pos
                )
                # get mother monomers from the pointed end
                (
                    particles, 
                    pointed_particle_ids,
                    actin_number,
                ) = ActinGenerator.get_actins_for_linear_fiber(
                    fiber,
                    actin_arp2_normal,
                    actin_arp2_axis_pos,
                    -1,
                    offset_vector,
                    actin_number,
                    particles,
                )
                pointed_particle_ids.reverse()
                raw_pointed_type = particles[pointed_particle_ids[0]].type_name
                pointed_state = "branch" if fiber.is_daughter else "pointed"
                particles[pointed_particle_ids[0]].type_name = f"actin#{pointed_state}_ATP_{raw_pointed_type[-1:]}"
                if fiber.is_daughter:
                    particles = ActinGenerator.check_shift_branch_actin_numbers(
                        particles, pointed_particle_ids)
                    actin_number = int(particles[pointed_particle_ids[len(pointed_particle_ids) - 1]].type_name[-1]) + 1
                # get mother actin bound to arp2
                actin_arp2_id = str(uuid.uuid4())
                last_pointed_id = pointed_particle_ids[len(pointed_particle_ids) - 1]
                particles[actin_arp2_id] = ParticleData(
                    unique_id=actin_arp2_id,
                    type_name=f"actin#ATP_{actin_number}",
                    position=fork_positions[0],
                    neighbor_ids=[last_pointed_id],
                )
                particles[last_pointed_id].neighbor_ids.append(actin_arp2_id)
                actin_number = ActinGenerator.get_actin_number(actin_number, 1)
                # get mother monomers toward the barbed end
                (
                    particles, 
                    barbed_particle_ids,
                    actin_number,
                ) = ActinGenerator.get_actins_for_linear_fiber(
                    fiber,
                    actin_arp2_normal,
                    actin_arp2_axis_pos,
                    1,
                    offset_vector,
                    actin_number,
                    particles,
                )
                particles[barbed_particle_ids[0]].neighbor_ids.append(actin_arp2_id)
                particles[actin_arp2_id].neighbor_ids.append(barbed_particle_ids[0])
                barbed_id = barbed_particle_ids[len(barbed_particle_ids) - 1]
                particles[barbed_id].type_name = f"actin#barbed_ATP_{particles[barbed_id].type_name[-1:]}"
                fiber_particle_ids += pointed_particle_ids + [actin_arp2_id] + barbed_particle_ids
            # get the daughter monomers on this branch after the first branch actin
            axis_pos = arp.daughter_fiber.get_nearest_position(fork_positions[3])
            (
                particles,
                branch_particle_ids,
            ) = ActinGenerator.get_monomers_for_fiber(
                arp.daughter_fiber,
                ReaddyUtil.normalize(fork_positions[3] - axis_pos),
                axis_pos,
                offset_vector + v_branch_shift,
                2,
                particles,
            )
            # get daughter arp2, arp3, and first branch actin monomers
            arp2_id = str(uuid.uuid4())
            particles[arp2_id] = ParticleData(
                unique_id=arp2_id,
                type_name="arp2#branched",
                position=fork_positions[1],
                neighbor_ids=[],
            )
            arp3_id = str(uuid.uuid4())
            particles[arp3_id] = ParticleData(
                unique_id=arp3_id,
                type_name="arp3#ATP",
                position=fork_positions[2],
                neighbor_ids=[arp2_id],
            )
            particles[arp2_id].neighbor_ids.append(arp3_id)
            branch_actin_id = str(uuid.uuid4())
            branch_state = "_barbed" if len(branch_particle_ids) == 0 else ""
            particles[branch_actin_id] = ParticleData(
                unique_id=branch_actin_id,
                type_name=f"actin#branch{branch_state}_ATP_1",
                position=fork_positions[3],
                neighbor_ids=[arp2_id],
            )
            # attach mother to arp
            particles[arp2_id].neighbor_ids.append(branch_actin_id)
            if a == 0:
                actin_arp3_id = barbed_particle_ids[0]
            else:
                actin_arp2_index = arp.get_closest_actin_index(fiber_particle_ids, actin_arp_ids, particles)
                if actin_arp2_index < 0:
                    raise Exception("Failed to find mother actins to bind to arp")
                actin_arp2_id = fiber_particle_ids[actin_arp2_index]
                actin_arp3_id = fiber_particle_ids[actin_arp2_index + 1]
            actin_arp_ids += [actin_arp2_id, actin_arp3_id]
            particles[arp2_id].neighbor_ids.append(actin_arp2_id)
            particles[actin_arp2_id].neighbor_ids.append(arp2_id)
            particles[arp3_id].neighbor_ids.append(actin_arp3_id)
            particles[actin_arp3_id].neighbor_ids.append(arp3_id)
            # attach daughter to arp
            if len(branch_particle_ids) > 0: 
                particles[branch_actin_id].neighbor_ids.append(branch_particle_ids[0])
                particles[branch_particle_ids[0]].neighbor_ids.append(branch_actin_id)
            daughter_particle_ids += [arp2_id, arp3_id, branch_actin_id] + branch_particle_ids
        # add non-nucleated bound arps
        for a in range(len(fiber.bound_arps)):
            particles, particle_ids = ActinGenerator.add_bound_arp_monomers(
                particle_ids,
                fiber,
                fiber.bound_arps[a],
                actin_arp_ids,
                particles,
            )
        particle_ids += fiber_particle_ids + daughter_particle_ids
        return particles, particle_ids

    @staticmethod
    def get_monomers_for_fiber(
        fiber, start_normal, start_axis_pos, offset_vector, pointed_actin_number, particles = {}
    ):
        """
        recursively get all the monomer data for the given fiber network
        """
        actin_number = pointed_actin_number
        if len(fiber.nucleated_arps) < 1:
            # fiber has no branches
            particles, particle_ids, _, = ActinGenerator.get_actins_for_linear_fiber(
                fiber, start_normal, start_axis_pos, 1, offset_vector, actin_number, particles
            )
            for a in range(len(fiber.bound_arps)):
                particles, particle_ids = ActinGenerator.add_bound_arp_monomers(
                    particle_ids,
                    fiber,
                    fiber.bound_arps[a],
                    [],
                    particles,
                )
        else:
            # fiber has branches
            particles, particle_ids = ActinGenerator.get_monomers_for_branching_fiber(
                [],
                fiber,
                offset_vector,
                actin_number,
                particles,
            )
        return particles, particle_ids

    @staticmethod
    def get_monomers(fibers_data):
        """
        get all the monomer data for the (branched) fibers in fibers_data

        fibers_data: List[FiberData]
        (FiberData for mother fibers only, which should have
        their daughters' FiberData attached to their nucleated arps)
        """
        result = {
            "topologies": {},
            "particles": {},
        }
        for fiber_data in fibers_data:
            particles, particle_ids = ActinGenerator.get_monomers_for_fiber(
                fiber_data,
                ReaddyUtil.get_random_perpendicular_vector(
                    fiber_data.get_first_segment_direction()
                ),
                fiber_data.pointed_point(),
                np.zeros(3),
                1,
            )
            result["topologies"][str(uuid.uuid4())] = {
                "type": "Actin-Polymer",
                "particle_ids": particle_ids,
            }
            result["particles"] = {**result["particles"], **particles}
        for particle_id in result["particles"]:
            result["particles"][particle_id] = dict(result["particles"][particle_id])
        return result

    @staticmethod
    def particles_to_string(particles):
        result = ""
        for particle_id in particles:
            result += str(dict(particles[particle_id])) + ", "
        return result
