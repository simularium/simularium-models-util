#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid

import numpy as np

from ..common import ReaddyUtil, ParticleData
from .actin_structure import ActinStructure, FiberData

next_id = 0


class ActinGenerator:
    """
    Generates positions, types, and edges for monomers in actin networks
    """

    @staticmethod
    def set_next_id(_id):
        global next_id
        next_id = _id

    @staticmethod
    def get_next_id():
        global next_id
        if next_id >= 0:
            result = next_id
            next_id += 1
        else:
            result = str(uuid.uuid4())
        return result

    @staticmethod
    def get_actin_number(actin_number, offset):
        """
        get the type number for an actin plus the given offset in range [-1, 1]
        (i.e. return 3 for type = "actin#ATP_1" and offset = -1)
        """
        assert offset >= -1 or offset <= 1, "Offset for actin number is not in [-1, 1]"
        n = actin_number + offset
        if n > 3:
            n -= 3
        if n < 1:
            n += 3
        return n

    @staticmethod
    def get_neighbor_actin_id(particle_id, direction, particles):
        """
        get the id for an actin's actin neighbor in the given direction
        """
        if particle_id is None:
            return None
        type_name = particles[particle_id].type_name
        assert "actin" in type_name, "Particle is not an actin"
        direction = direction / abs(direction)  # normalize
        actin_number = int(type_name[-1:])
        neighbor_ids = particles[particle_id].neighbor_ids
        for neighbor_id in neighbor_ids:
            neighbor_type_name = particles[neighbor_id].type_name
            if "actin" not in neighbor_type_name:
                continue
            neighbor_actin_number = int(neighbor_type_name[-1:])
            goal_actin_number = ActinGenerator.get_actin_number(actin_number, direction)
            if goal_actin_number == neighbor_actin_number:
                return neighbor_id
        return None

    @staticmethod
    def set_particle_type_name(type_prefix, particle_id, particles):
        """
        set the type name of the particle using it's same actin number
        with the new type_prefix
        """
        if particle_id is None:
            return particles
        actin_number = particles[particle_id].type_name[-1:]
        particles[particle_id].type_name = f"{type_prefix}{actin_number}"
        return particles

    @staticmethod
    def remove_mid_from_actin(particle_id, particles):
        """
        remove the "mid" flag in the actin type_name
        for the monomer with particle_id
        """
        if particle_id is None:
            return particles
        old_type_name = particles[particle_id].type_name
        if "mid" not in old_type_name:
            return particles
        particles = ActinGenerator.set_particle_type_name(
            "actin#ATP_", particle_id, particles
        )
        return particles

    @staticmethod
    def remove_mother_mid_at_junction(
        actin_arp_ids,
        particles={},
    ):
        """
        remove "mid" flags from actins near a branch junction,
        including the actin bound to arp2, the actin bound to arp3,
        as well as one actin before and two actins after the arps
        on the mother filament
        """
        actin_mother0_id = ActinGenerator.get_neighbor_actin_id(
            actin_arp_ids[0], -1, particles
        )
        actin_mother3_id = ActinGenerator.get_neighbor_actin_id(
            actin_arp_ids[1], 1, particles
        )
        actin_mother4_id = ActinGenerator.get_neighbor_actin_id(
            actin_mother3_id, 1, particles
        )
        particles = ActinGenerator.remove_mid_from_actin(actin_mother0_id, particles)
        particles = ActinGenerator.remove_mid_from_actin(actin_arp_ids[0], particles)
        particles = ActinGenerator.remove_mid_from_actin(actin_arp_ids[1], particles)
        particles = ActinGenerator.remove_mid_from_actin(actin_mother3_id, particles)
        particles = ActinGenerator.remove_mid_from_actin(actin_mother4_id, particles)
        return particles

    @staticmethod
    def check_shift_branch_actin_numbers(particles, particle_ids):
        """
        if the first actin's number is not 2,
        shift the branch's actin numbers so that it is
        """
        first_actin_type = particles[particle_ids[0]].type_name
        if "2" not in first_actin_type:
            actin_number = int(first_actin_type[-1:])
            offset = actin_number - 2
            for i in range(len(particle_ids)):
                new_actin_number = actin_number - offset
                if new_actin_number > 3:
                    new_actin_number -= 3
                type_name = particles[particle_ids[i]].type_name
                particles[
                    particle_ids[i]
                ].type_name = f"{type_name[:-1]}{new_actin_number}"
                actin_number += 1
                if actin_number > 3:
                    actin_number = 1
            return particles
        return particles

    @staticmethod
    def get_actins_for_linear_fiber(
        fiber,
        start_normal,
        start_axis_pos,
        direction,
        offset_vector,
        pointed_actin_number,
        particles={},
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
                new_particle_id = ActinGenerator.get_next_id()
                particle_ids.append(new_particle_id)
                particles[new_particle_id] = ParticleData(
                    unique_id=new_particle_id,
                    position=axis_pos + actin_offset_from_axis * normal + offset_vector,
                    neighbor_ids=[],
                )
        # get actin types and edges
        actin_number = pointed_actin_number
        for i in range(len(particle_ids)):
            index = i if direction > 0 else len(particle_ids) - 1 - i
            particle_id = particle_ids[index]
            particles[particle_id].type_name = f"actin#mid_ATP_{actin_number}"
            actin_number = ActinGenerator.get_actin_number(actin_number, 1)
            if index > 0:
                particles[particle_id].neighbor_ids.append(particle_ids[index - 1])
            if index < len(particle_ids) - 1:
                particles[particle_id].neighbor_ids.append(particle_ids[index + 1])
        if direction < 0:
            particle_ids.reverse()
        return particles, particle_ids, actin_number

    @staticmethod
    def add_bound_arp_monomers(
        particle_ids,
        fiber,
        actin_arp_ids,
        particles={},
    ):
        """
        add positions, types, and edges for a bound arp2 and arp3
        """
        for a in range(len(fiber.bound_arps)):
            arp = fiber.bound_arps[a]
            closest_actin_index = arp.get_closest_actin_index(
                particle_ids, actin_arp_ids, None, particles
            )
            if closest_actin_index < 0:
                return particles, particle_ids
            # create arp2 and arp3
            arp2_id = ActinGenerator.get_next_id()
            arp3_id = ActinGenerator.get_next_id()
            actin_arp2_id = particle_ids[closest_actin_index]
            actin_arp3_id = particle_ids[closest_actin_index + 1]
            actin_arp_ids += [actin_arp2_id, actin_arp3_id]
            particle_ids.append(arp2_id)
            particles[arp2_id] = ParticleData(
                unique_id=arp2_id,
                type_name="arp2",
                position=arp.get_bound_monomer_position(
                    particles[actin_arp2_id].position, fiber, "arp2"
                ),
                neighbor_ids=[actin_arp2_id, arp3_id],
            )
            particle_ids.append(arp3_id)
            particles[arp3_id] = ParticleData(
                unique_id=arp3_id,
                type_name="arp3#ATP",
                position=arp.get_bound_monomer_position(
                    particles[actin_arp2_id].position, fiber, "arp3"
                ),
                neighbor_ids=[actin_arp3_id, arp2_id],
            )
            # update actin_arp2 and actin_arp3 (actins bound to the arps)
            particles[actin_arp2_id].neighbor_ids.append(arp2_id)
            particles[actin_arp3_id].neighbor_ids.append(arp3_id)
            particles = ActinGenerator.remove_mother_mid_at_junction(
                [actin_arp2_id, actin_arp3_id], particles
            )
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
    def get_monomers_for_daughter_fiber(
        mother_fiber,
        nucleated_arp,
        offset_vector,
        particles={},
    ):
        """
        get any bound arps and any daughter fibers attached to this fiber
        """
        (
            fork_positions,
            v_branch_shift,
        ) = ActinGenerator.get_nucleated_arp_monomer_positions(
            mother_fiber, nucleated_arp
        )
        # create daughter monomers on this branch after the first branch actin
        axis_pos = nucleated_arp.daughter_fiber.get_nearest_position(fork_positions[3])
        particles, daughter_particle_ids = ActinGenerator.get_monomers_for_fiber(
            nucleated_arp.daughter_fiber,
            ReaddyUtil.normalize(fork_positions[3] - axis_pos),
            axis_pos,
            offset_vector + v_branch_shift,
            2,
            particles,
        )
        # create branch junction monomers (arp2, arp3, first daughter actin)
        arp2_id = ActinGenerator.get_next_id()
        arp3_id = ActinGenerator.get_next_id()
        branch_actin_id = ActinGenerator.get_next_id()
        branch_state = "_barbed" if len(daughter_particle_ids) == 0 else ""
        particles[branch_actin_id] = ParticleData(
            unique_id=branch_actin_id,
            type_name=f"actin#branch{branch_state}_ATP_1",
            position=fork_positions[3],
            neighbor_ids=[arp2_id],
        )
        particles[arp2_id] = ParticleData(
            unique_id=arp2_id,
            type_name="arp2#branched",
            position=fork_positions[1],
            neighbor_ids=[arp3_id, branch_actin_id],
        )
        particles[arp3_id] = ParticleData(
            unique_id=arp3_id,
            type_name="arp3#ATP",
            position=fork_positions[2],
            neighbor_ids=[arp2_id],
        )
        return particles, daughter_particle_ids, [arp2_id, arp3_id, branch_actin_id]

    @staticmethod
    def attach_daughter_fiber_to_mother_fiber(
        nucleated_arp,
        junction_ids,
        these_actin_arp_ids,
        mother_particle_ids,
        all_actin_arp_ids,
        second_daughter_actin_id,
        mother_fiber,
        particles={},
    ):
        """
        attach daughter fiber monomers to their mother fiber monomers
        """
        # choose mother actins to attach to arps if not already determined
        if these_actin_arp_ids is not None:
            actin_arp2_id = these_actin_arp_ids[0]
            actin_arp3_id = these_actin_arp_ids[1]
        else:
            actin_arp2_index = nucleated_arp.get_closest_actin_index(
                mother_particle_ids, all_actin_arp_ids, mother_fiber, particles
            )
            if actin_arp2_index < 0:
                raise Exception("Failed to find mother actins to bind to arp")
            actin_arp2_id = mother_particle_ids[actin_arp2_index]
            actin_arp3_id = mother_particle_ids[actin_arp2_index + 1]
        # attach mother actins to arps
        particles[junction_ids[0]].neighbor_ids.append(actin_arp2_id)
        particles[actin_arp2_id].neighbor_ids.append(junction_ids[0])
        particles[junction_ids[1]].neighbor_ids.append(actin_arp3_id)
        particles[actin_arp3_id].neighbor_ids.append(junction_ids[1])
        particles = ActinGenerator.remove_mother_mid_at_junction(
            [actin_arp2_id, actin_arp3_id], particles
        )
        # attach daughter to arp
        if second_daughter_actin_id is not None:
            particles[junction_ids[2]].neighbor_ids.append(second_daughter_actin_id)
            particles[second_daughter_actin_id].neighbor_ids.append(junction_ids[2])
            particles = ActinGenerator.remove_mid_from_actin(
                second_daughter_actin_id, particles
            )
        return particles, [actin_arp2_id, actin_arp3_id]

    @staticmethod
    def get_main_monomers_for_fiber(
        fiber,
        start_normal,
        start_axis_pos,
        offset_vector,
        pointed_actin_number,
        particles={},
    ):
        """
        get the main actins for a fiber (i.e. no branches or arps)
        """
        actin_number = pointed_actin_number
        if not fiber.is_daughter and len(fiber.nucleated_arps) > 0:
            # if this is a mother filament with daughters,
            # the pointed end is constrained by the first branch junction
            (
                fork_positions,
                v_branch_shift,
            ) = ActinGenerator.get_nucleated_arp_monomer_positions(
                fiber, fiber.nucleated_arps[0]
            )
            actin_arp2_axis_pos = fiber.get_nearest_position(fork_positions[0])
            actin_arp2_normal = ReaddyUtil.normalize(
                fork_positions[0] - actin_arp2_axis_pos
            )
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
            if len(pointed_particle_ids) > 0:
                # add "pointed" flag to first actin
                particles = ActinGenerator.set_particle_type_name(
                    "actin#pointed_ATP_", pointed_particle_ids[0], particles
                )
                if len(pointed_particle_ids) > 1:
                    # remove "mid" from second actin
                    ActinGenerator.remove_mid_from_actin(
                        pointed_particle_ids[1], particles
                    )
            # get mother actin bound to arp2
            actin_arp2_id = ActinGenerator.get_next_id()
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
                _,
            ) = ActinGenerator.get_actins_for_linear_fiber(
                fiber,
                actin_arp2_normal,
                actin_arp2_axis_pos,
                1,
                offset_vector,
                actin_number,
                particles,
            )
            if len(barbed_particle_ids) == 0:
                raise Exception("No actin was generated to attach arp3 to")
            actin_arp3_id = barbed_particle_ids[0]
            particles[actin_arp3_id].neighbor_ids.append(actin_arp2_id)
            particles[actin_arp2_id].neighbor_ids.append(actin_arp3_id)
            particle_ids = pointed_particle_ids + [actin_arp2_id] + barbed_particle_ids
            actin_arp_ids = [actin_arp2_id, actin_arp3_id]
        else:
            (particles, particle_ids, _,) = ActinGenerator.get_actins_for_linear_fiber(
                fiber,
                start_normal,
                start_axis_pos,
                1,
                offset_vector,
                actin_number,
                particles,
            )
            if fiber.is_daughter:
                particles = ActinGenerator.check_shift_branch_actin_numbers(
                    particles, particle_ids
                )
            elif len(particle_ids) > 0:
                particles = ActinGenerator.set_particle_type_name(
                    "actin#pointed_ATP_", particle_ids[0], particles
                )
                if len(particle_ids) > 1:
                    # remove "mid" from second actin
                    ActinGenerator.remove_mid_from_actin(particle_ids[1], particles)
            actin_arp_ids = None
        particles = ActinGenerator.set_particle_type_name(
            "actin#barbed_ATP_",
            particle_ids[len(particle_ids) - 1],
            particles,
        )
        return particles, particle_ids, actin_arp_ids

    @staticmethod
    def get_monomers_for_fiber(
        fiber,
        start_normal,
        start_axis_pos,
        offset_vector,
        pointed_actin_number,
        particles={},
    ):
        """
        get the main actins for a fiber as well as any bound arps and daughter fibers
        """
        (
            particles,
            main_particle_ids,
            first_actin_arp_ids,
        ) = ActinGenerator.get_main_monomers_for_fiber(
            fiber,
            start_normal,
            start_axis_pos,
            offset_vector,
            pointed_actin_number,
            particles,
        )
        daughter_particle_ids = []
        all_actin_arp_ids = []
        for a in range(len(fiber.nucleated_arps)):
            nucleated_arp = fiber.nucleated_arps[a]
            (
                particles,
                particle_ids,
                junction_ids,
            ) = ActinGenerator.get_monomers_for_daughter_fiber(
                fiber, nucleated_arp, offset_vector, particles
            )
            (
                particles,
                actin_arp_ids,
            ) = ActinGenerator.attach_daughter_fiber_to_mother_fiber(
                nucleated_arp,
                junction_ids,
                first_actin_arp_ids if a == 0 else None,
                main_particle_ids,
                all_actin_arp_ids,
                particle_ids[0] if len(particle_ids) > 0 else None,
                fiber,
                particles,
            )
            daughter_particle_ids += junction_ids + particle_ids
            all_actin_arp_ids += actin_arp_ids
        particles, main_particle_ids = ActinGenerator.add_bound_arp_monomers(
            main_particle_ids,
            fiber,
            all_actin_arp_ids,
            particles,
        )
        return particles, main_particle_ids + daughter_particle_ids

    @staticmethod
    def get_extents(child_box_center, child_box_size):
        """
        get the min and max extents
        within the coordinates defined by a parent box
        of a child box defined by center and size
        """
        return (
            child_box_center - child_box_size / 2.0,
            child_box_center + child_box_size / 2.0,
        )

    @staticmethod
    def position_is_in_bounds(position, min_extent, max_extent):
        """
        check if a position is within the given extents
        """
        for dim in range(3):
            if position[dim] < min_extent[dim] or position[dim] > max_extent[dim]:
                return False
        return True

    @staticmethod
    def get_point_on_plane_of_intersecting_extent(
        point1, point2, min_extent, max_extent, direction
    ):
        """
        get a point (which is also the normal) of the extent plane intersected
        by the line segment between the given points,
        assume bounds are a rectangular prism orthogonal to cartesian grid
        """
        result = np.zeros(3)
        for dim in range(3):
            if (point1[dim] < min_extent[dim] and point2[dim] > min_extent[dim]) or (
                point2[dim] < min_extent[dim] and point1[dim] > min_extent[dim]
            ):
                # points straddle both extents
                if direction > 0:
                    result[dim] = min_extent[dim]
                else:
                    result[dim] = max_extent[dim]
                return result
            if (point1[dim] < min_extent[dim] and point2[dim] > min_extent[dim]) or (
                point2[dim] < min_extent[dim] and point1[dim] > min_extent[dim]
            ):
                # points straddle min extent
                result[dim] = min_extent[dim]
                return result
            elif (point1[dim] < max_extent[dim] and point2[dim] > max_extent[dim]) or (
                point2[dim] < max_extent[dim] and point1[dim] > max_extent[dim]
            ):
                # points straddle max extent
                result[dim] = max_extent[dim]
                return result
        return None

    @staticmethod
    def get_intersection_point_with_extents(
        point1, point2, min_extent, max_extent, direction=1
    ):
        """
        get the point where the line segment between the given positions
        intersects the bounds volume, assume bounds are
        a rectangular prism orthogonal to cartesian grid
        """
        plane = ActinGenerator.get_point_on_plane_of_intersecting_extent(
            point1, point2, min_extent, max_extent, direction
        )
        if plane is None:
            return None
        v_direction = direction * ReaddyUtil.normalize(point2 - point1)
        t = (np.dot(plane, plane) - np.dot(plane, point1)) / np.dot(plane, v_direction)
        intersection = point1 + t * v_direction
        if not ActinGenerator.position_is_in_bounds(
            intersection, min_extent, max_extent
        ):
            return None
        return intersection

    @staticmethod
    def _get_max_fiber_id(fibers_data):
        """
        get the largest fiber id
        """
        max_id = 0
        for fiber in fibers_data:
            if fiber.fiber_id > max_id:
                max_id = fiber.fiber_id
        return max_id

    @staticmethod
    def _create_fiber(current_chunk, source_fiber, found_chunk, max_fiber_id):
        """
        create a FiberData for a cropped chunk of a source fiber
        """
        if not found_chunk:
            fiber_id = source_fiber.fiber_id
        else:
            max_fiber_id += 1
            fiber_id = max_fiber_id
        return FiberData(fiber_id, current_chunk, source_fiber.type_name), max_fiber_id
            
    @staticmethod
    def get_cropped_fibers(fibers_data, min_extent, max_extent, position_offset=None):
        """
        crop the fiber data to a cube volume
        defined by min_extent and max_extent
        and apply the position_offset

        fibers_data: List[FiberData]
        (FiberData for mother fibers only, which should have
        their daughters' FiberData attached to their nucleated arps)

        # TODO handle daughter fiber connections
        """
        if min_extent is None or max_extent is None:
            return fibers_data
        if position_offset is None:
            position_offset = np.zeros(3)
        max_fiber_id = ActinGenerator._get_max_fiber_id(fibers_data)
        found_chunk = False
        result = []
        s = ""
        for fiber in fibers_data:
            s += f"fiber id = {fiber.fiber_id}\n"
            current_chunk = []
            s += f"d\n"
            tracing = False
            for i in range(len(fiber.points)):
                position_is_in_bounds = ActinGenerator.position_is_in_bounds(
                    fiber.points[i], min_extent, max_extent
                )
                s += f"in bounds? {position_is_in_bounds}\n"
                s += f"tracing? {tracing}\n"
                if not tracing:
                    if i == 0 and position_is_in_bounds:
                        s += "start at the first point\n"
                        # start at the first point if it's in bounds
                        current_chunk = [fiber.points[i] + position_offset]
                        s += f"a\n"
                        tracing = True
                    elif i < len(fiber.points) - 1 or position_is_in_bounds:
                        s += "start at bounds intersect\n"
                        # start at the intersection with the bounds
                        # between the current and either prev or next points
                        # depending on whether the current point is in bounds
                        point1_index = i - 1 if position_is_in_bounds else i
                        point2_index = i if position_is_in_bounds else i + 1
                        s += f"{point1_index} {point2_index}\n"
                        intersection = (
                            ActinGenerator.get_intersection_point_with_extents(
                                fiber.points[point1_index],
                                fiber.points[point2_index],
                                min_extent,
                                max_extent,
                            )
                        )
                        if intersection is not None:
                            current_chunk = [intersection + position_offset]
                            s += f"b\n"
                            tracing = True
                        else:
                            s += f"no intersect\n"
                            
                else:
                    if position_is_in_bounds:
                        s += "add point\n"
                        # continue adding points within the volume
                        current_chunk.append(fiber.points[i] + position_offset)
                        if i == len(fiber.points) - 1 and len(current_chunk) > 0:
                            s += "end at point\n"
                            # end if this is the last point
                            new_fiber, max_fiber_id = ActinGenerator._create_fiber(
                                current_chunk, fiber, found_chunk, max_fiber_id
                            )
                            result.append(new_fiber)
                            found_chunk = True
                    else:
                        s += "end at bounds intersect\n"
                        # end at the intersection with the bounds
                        # between the prev and current points
                        s += f"c\n"
                        tracing = False
                        intersection = (
                            ActinGenerator.get_intersection_point_with_extents(
                                fiber.points[i - 1],
                                fiber.points[i],
                                min_extent,
                                max_extent,
                                -1,
                            )
                        )
                        if intersection is not None and len(current_chunk) > 0:
                            current_chunk.append(intersection + position_offset)
                            new_fiber, max_fiber_id = ActinGenerator._create_fiber(
                                current_chunk, fiber, found_chunk, max_fiber_id
                            )
                            result.append(new_fiber)
                            found_chunk = True
        # raise Exception(s)
        return result

    @staticmethod
    def get_monomers(
        fibers_data, child_box_center=None, child_box_size=None, use_uuids=True
    ):
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
        ActinGenerator.set_next_id(-1 if use_uuids else 0)
        if child_box_center is not None and child_box_size is not None:
            min_extent, max_extent = ActinGenerator.get_extents(
                child_box_center, child_box_size
            )
            cropped_fiber_data = ActinGenerator.get_cropped_fibers(
                fibers_data, min_extent, max_extent, -1 * child_box_center
            )
        else:
            cropped_fiber_data = fibers_data
        for fiber in cropped_fiber_data:
            particles, particle_ids = ActinGenerator.get_monomers_for_fiber(
                fiber,
                ReaddyUtil.get_random_perpendicular_vector(
                    fiber.get_first_segment_direction()
                ),
                fiber.pointed_point(),
                np.zeros(3),
                1,
            )
            result["topologies"][ActinGenerator.get_next_id()] = {
                "type_name": "Actin-Polymer",
                "particle_ids": particle_ids,
            }
            particles = {
                p_id: dict(particle_data) for p_id, particle_data in particles.items()
            }
            result["particles"] = {**result["particles"], **particles}
        return result

    @staticmethod
    def particles_to_string(particle_ids, particles, info=""):
        result = ""
        for particle_id in particle_ids:
            if len(info) < 1:
                result += str(dict(particles[particle_id])) + ", "
            elif "id" in info:
                result += str(particles[particle_id].unique_id) + ", "
            elif "type" in info:
                result += str(particles[particle_id].type_name) + ", "
            elif "pos" in info:
                result += str(particles[particle_id].position) + ", "
        return result
