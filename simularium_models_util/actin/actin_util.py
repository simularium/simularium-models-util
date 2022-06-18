#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import readdy
import random

from simularium_models_util import actin

from ..common import ReaddyUtil
from .actin_generator import ActinGenerator
from .actin_structure import ActinStructure
from .fiber_data import FiberData


parameters = {}


def set_parameters(p):
    global parameters
    parameters = p
    return p


displacements = {}


def set_displacements(d):
    global displacements
    displacements = d
    return d


time_index = 0
init_monomer_positions = {}
pointed_monomer_positions = []


class ActinUtil:
    def __init__(self, parameters, displacements=None):
        """
        Utility functions for ReaDDy branched actin models

        Parameters need to be accessible in ReaDDy callbacks
        which can't be instance methods, so parameters are global
        """
        set_parameters(parameters)
        if displacements is not None:
            set_displacements(displacements)

    @staticmethod
    def get_new_vertex(topology):
        """
        Get the vertex tagged "new"
        """
        results = ReaddyUtil.get_vertices_of_type(
            topology, "new", exact_match=False, error_msg="Failed to find new vertex"
        )
        if len(results) > 1:
            raise Exception(
                f"Found more than one new vertex\n"
                f"{ReaddyUtil.topology_to_string(topology)}"
            )
        return results[0]

    @staticmethod
    def get_new_arp23(topology):
        """
        get a new arp3 and its unbranched arp2#free neighbor,
        meaning the arp2/3 dimer has just bound
        """
        for vertex in topology.graph.get_vertices():
            pt = topology.particle_type_of_vertex(vertex)
            if "arp3#new" in pt:
                for neighbor in vertex:
                    if topology.particle_type_of_vertex(neighbor.get()) == "arp2#free":
                        return neighbor.get(), vertex
        return None, None

    @staticmethod
    def get_actin_number(topology, vertex, offset):
        """
        get the type number for an actin plus the given offset in range [-1, 1]
        (i.e. return 3 for type = "actin#ATP_1" and offset = -1)
        """
        pt = topology.particle_type_of_vertex(vertex)
        if "actin" not in pt:
            raise Exception(
                f"Failed to get actin number: {pt} is not actin\n"
                f"{ReaddyUtil.topology_to_string(topology)}"
            )
        return ReaddyUtil.calculate_polymer_number(int(pt[-1]), offset)

    @staticmethod
    def get_all_polymer_actin_types(vertex_type):
        """
        get a list of all numbered versions of a type
        (e.g. for "actin#ATP" return
        ["actin#ATP_1", "actin#ATP_2", "actin#ATP_3"])
        """
        spacer = "_"
        if "#" not in vertex_type:
            spacer = "#"
        return [
            f"{vertex_type}{spacer}1",
            f"{vertex_type}{spacer}2",
            f"{vertex_type}{spacer}3",
            f"{vertex_type}{spacer}4",
            f"{vertex_type}{spacer}5"
        ]

    @staticmethod
    def get_actin_rotation(positions, box_size, periodic_boundary=True):
        """
        get the difference in the actin's current orientation
        compared to the initial orientation as a rotation matrix
        positions = [prev actin position, middle actin position, next actin position]
        """
        if periodic_boundary:
            positions[0] = ReaddyUtil.get_non_periodic_boundary_position(
                positions[1], positions[0], box_size
            )
            positions[2] = ReaddyUtil.get_non_periodic_boundary_position(
                positions[1], positions[2], box_size
            )
        current_orientation = ReaddyUtil.get_orientation_from_positions(positions)
        return np.matmul(
            current_orientation, np.linalg.inv(ActinStructure.orientation())
        )

    @staticmethod
    def get_actin_axis_position(positions, box_size, periodic_boundary=True):
        """
        get the position on the filament axis closest to an actin
        positions = [
            previous actin position,
            middle actin position,
            next actin position
        ]
        """
        rotation = ActinUtil.get_actin_rotation(positions, box_size, periodic_boundary)
        if rotation is None:
            return None
        vector_to_axis_local = np.squeeze(
            np.array(np.dot(rotation, ActinStructure.vector_to_axis()))
        )
        return positions[1] + vector_to_axis_local

    @staticmethod
    def get_position_for_new_vertex(positions, offset_vector):
        """
        get the offset vector in the local space for the actin at positions[1]
        positions = [
            previous actin position,
            middle actin position,
            next actin position
        ]
        """
        rotation = ActinUtil.get_actin_rotation(
            positions, parameters["box_size"], bool(parameters["periodic_boundary"])
        )
        if rotation is None:
            return None
        vector_to_new_pos = np.squeeze(np.array(np.dot(rotation, offset_vector)))
        return (positions[1] + vector_to_new_pos).tolist()

    @staticmethod
    def get_next_actin(topology, v_actin, direction, error_if_not_found=False):
        """
        get the next actin toward the pointed or barbed direction
        """
        n = ActinUtil.get_actin_number(topology, v_actin, direction)
        end_type = "barbed" if direction > 0 else "pointed"
        actin_types = [
            f"actin#ATP_{n}",
            f"actin#{n}",
            f"actin#mid_ATP_{n}",
            f"actin#mid_{n}",
            f"actin#{end_type}_ATP_{n}",
            f"actin#{end_type}_{n}",
        ]
        if direction < 0 and n == 1:
            actin_types += ["actin#branch_1", "actin#branch_ATP_1"]
        error_msg = f"Couldn't find next actin with number {n}"
        v_actin_neighbor = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_actin,
            actin_types,
            [],
            parameters["verbose"],
            error_msg if not error_if_not_found else "",
            error_msg if error_if_not_found else "",
        )
        return v_actin_neighbor

    @staticmethod
    def get_prev_branch_actin(topology, vertex, last_vertex_id, max_edges):
        """
        recurse up the chain until first branch actin is found or max_edges is reached
        """
        for neighbor in vertex:
            n_id = topology.particle_id_of_vertex(neighbor)
            if n_id == last_vertex_id:
                continue
            pt = topology.particle_type_of_vertex(neighbor)
            if "branch_" in pt:
                return neighbor.get(), max_edges
            else:
                if max_edges <= 1:
                    return None, max_edges
                return ActinUtil.get_prev_branch_actin(
                    topology, neighbor.get(), n_id, max_edges - 1
                )
        return None, max_edges

    @staticmethod
    def get_branch_orientation_vertices_and_offset(topology, vertex):
        """
        get orientation vertices [actin, actin_arp2, actin_arp3]
        for a new actin within 3 actins of a branch,
        as well as the offset vector
        """
        v_arp2 = ReaddyUtil.get_neighbor_of_types(
            topology, vertex, ["arp2", "arp2#branched", "arp2#free"], []
        )
        offset_index = 0
        if v_arp2 is None:
            v_branch, edges = ActinUtil.get_prev_branch_actin(topology, vertex, None, 3)
            if v_branch is None:
                raise Exception(
                    "Failed to set position: couldn't find arp2 "
                    f"or first branch actin\n{ReaddyUtil.topology_to_string(topology)}"
                )
            offset_index = 3 - edges
            v_arp2 = ReaddyUtil.get_neighbor_of_types(
                topology,
                v_branch,
                ["arp2", "arp2#branched", "arp2#free"],
                [],
                error_msg="Failed to set position: couldn't find arp2",
            )
        v_arp3 = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_arp2,
            ["arp3", "arp3#ATP", "arp3#new", "arp3#new_ATP"],
            [],
            error_msg="Failed to set position: couldn't find arp3",
        )
        actin_types = (
            ActinUtil.get_all_polymer_actin_types("actin")
            + ActinUtil.get_all_polymer_actin_types("actin#ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#mid")
            + ActinUtil.get_all_polymer_actin_types("actin#mid_ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#barbed")
            + ActinUtil.get_all_polymer_actin_types("actin#barbed_ATP")
        )
        v_actin_arp3 = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_arp3,
            actin_types,
            [],
            error_msg="Failed to set position: couldn't find actin_arp3",
        )
        n_pointed = ActinUtil.get_actin_number(topology, v_actin_arp3, -1)
        actin_types = [f"actin#ATP_{n_pointed}", f"actin#{n_pointed}"]
        v_actin_arp2 = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_actin_arp3,
            actin_types,
            [],
            error_msg="Failed to set position: couldn't find actin_arp2",
        )
        n_pointed = ActinUtil.get_actin_number(topology, v_actin_arp2, -1)
        actin_types = [
            f"actin#ATP_{n_pointed}",
            f"actin#{n_pointed}",
            f"actin#pointed_ATP_{n_pointed}",
            f"actin#pointed_{n_pointed}",
        ]
        if n_pointed == 1:
            actin_types += ["actin#branch_1", "actin#branch_ATP_1"]
        v_prev = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_actin_arp2,
            actin_types,
            [v_actin_arp3],
            error_msg="Failed to set position: couldn't find v_prev",
        )
        return (
            [v_prev, v_actin_arp2, v_actin_arp3],
            ActinStructure.mother1_to_branch_actin_vectors()[offset_index],
        )

    @staticmethod
    def set_end_vertex_position(topology, recipe, v_new, barbed):
        """
        set the position of a new pointed or barbed vertex
        """
        vertices = []
        offset_vector = (
            ActinStructure.mother1_to_mother3_vector()
            if barbed
            else ActinStructure.mother1_to_mother_vector()
        )
        at_branch = False
        vertices.append(
            ReaddyUtil.get_neighbor_of_type(topology, v_new, "actin", False)
        )
        if vertices[0] is None:
            (
                vertices,
                offset_vector,
            ) = ActinUtil.get_branch_orientation_vertices_and_offset(topology, v_new)
            at_branch = True
        else:
            vertices.append(
                ReaddyUtil.get_neighbor_of_type(
                    topology, vertices[0], "actin", False, [v_new]
                )
            )
            if vertices[1] is None:
                (
                    vertices,
                    offset_vector,
                ) = ActinUtil.get_branch_orientation_vertices_and_offset(
                    topology, v_new
                )
                at_branch = True
            else:
                vertices.append(
                    ReaddyUtil.get_neighbor_of_type(
                        topology, vertices[1], "actin", False, [vertices[0]]
                    )
                )
                if vertices[2] is None:
                    (
                        vertices,
                        offset_vector,
                    ) = ActinUtil.get_branch_orientation_vertices_and_offset(
                        topology, v_new
                    )
                    at_branch = True
        positions = []
        for v in vertices:
            positions.append(ReaddyUtil.get_vertex_position(topology, v))
        if barbed and not at_branch:
            positions = positions[::-1]
        pos = ActinUtil.get_position_for_new_vertex(positions, offset_vector)
        if pos is None:
            raise Exception(
                f"Failed to set position: couldn't calculate position\n"
                f"{ReaddyUtil.topology_to_string(topology)}"
            )
        recipe.change_particle_position(v_new, pos)

    @staticmethod
    def set_new_trimer_vertex_position(topology, recipe, v_new, v_pointed, v_barbed):
        """
        set the position of an actin monomer just added to a dimer to create a trimer
        """
        pos_new = ReaddyUtil.get_vertex_position(topology, v_new)
        pos_pointed = ReaddyUtil.get_vertex_position(topology, v_pointed)
        pos_barbed = ReaddyUtil.get_vertex_position(topology, v_barbed)
        v_barbed_to_pointed = pos_pointed - pos_barbed
        v_barbed_to_new = pos_new - pos_barbed
        current_angle = ReaddyUtil.get_angle_between_vectors(
            v_barbed_to_pointed, v_barbed_to_new
        )
        angle = ActinStructure.actin_to_actin_angle() - current_angle
        axis = np.cross(v_barbed_to_pointed, v_barbed_to_new)
        pos = pos_barbed + ReaddyUtil.rotate(
            ActinStructure.actin_to_actin_distance()
            * ReaddyUtil.normalize(v_barbed_to_new),
            axis,
            angle,
        )
        recipe.change_particle_position(v_new, pos)

    @staticmethod
    def set_arp23_vertex_position(
        topology, recipe, v_arp2, v_arp3, v_actin_arp2, v_actin_arp3
    ):
        """
        set the position of new arp2/3 vertices
        """
        actin_types = (
            ActinUtil.get_all_polymer_actin_types("actin")
            + ActinUtil.get_all_polymer_actin_types("actin#ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#mid")
            + ActinUtil.get_all_polymer_actin_types("actin#mid_ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#pointed")
            + ActinUtil.get_all_polymer_actin_types("actin#pointed_ATP")
            + ["actin#branch_1", "actin#branch_ATP_1"]
        )
        v1 = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_actin_arp2,
            actin_types,
            [v_actin_arp3],
            error_msg="Failed to set position: couldn't find v1",
        )
        pos1 = ReaddyUtil.get_vertex_position(topology, v1)
        pos2 = ReaddyUtil.get_vertex_position(topology, v_actin_arp2)
        pos3 = ReaddyUtil.get_vertex_position(topology, v_actin_arp3)
        pos_arp2 = ActinUtil.get_position_for_new_vertex(
            [pos1, pos2, pos3], ActinStructure.mother1_to_arp2_vector()
        )
        if pos_arp2 is None:
            raise Exception(
                f"Failed to set position of arp2: couldn't calculate position\n"
                f"{ReaddyUtil.topology_to_string(topology)}"
            )
        recipe.change_particle_position(v_arp2, pos_arp2)
        pos_arp3 = ActinUtil.get_position_for_new_vertex(
            [pos1, pos2, pos3], ActinStructure.mother1_to_arp3_vector()
        )
        if pos_arp3 is None:
            raise Exception(
                f"Failed to set position of arp3: couldn't calculate position\n"
                f"{ReaddyUtil.topology_to_string(topology)}"
            )
        recipe.change_particle_position(v_arp3, pos_arp3)

    @staticmethod
    def get_random_arp2(topology, with_ATP, with_branch):
        """
        get a random bound arp2 with the given arp3 nucleotide state
        and with or without a branch attached to the arp2
        """
        v_arp3s = ReaddyUtil.get_vertices_of_type(
            topology,
            "arp3#ATP" if with_ATP else "arp3",
            True,
            parameters["verbose"],
            f"Couldn't find arp3 (ATP={with_ATP})",
        )
        if len(v_arp3s) < 1:
            return None
        v_arp2s = []
        for v_arp3 in v_arp3s:
            v_arp2 = ReaddyUtil.get_neighbor_of_types(
                topology, v_arp3, ["arp2#branched" if with_branch else "arp2"], []
            )
            if v_arp2 is not None:
                v_arp2s.append(v_arp2)
        if len(v_arp2s) < 1:
            if parameters["verbose"]:
                print(f"Couldn't find arp2 (branch={with_branch})")
            return None
        return random.choice(v_arp2s)

    @staticmethod
    def check_arp3_attached_to_neighbors(
        topology, vertex, max_edges, exclude_id=None, last_vertex_id=None
    ):
        """
        Check if an arp3 is attached to the vertex's neighbors within max_edges
        """
        for neighbor in vertex:
            n_id = topology.particle_id_of_vertex(neighbor)
            if n_id == last_vertex_id or n_id == exclude_id:
                continue
            pt = topology.particle_type_of_vertex(neighbor)
            if "arp3" in pt:
                return True
            if "actin" not in pt or max_edges <= 1:
                continue
            arp3_is_attached_to_neighbors = ActinUtil.check_arp3_attached_to_neighbors(
                topology,
                neighbor.get(),
                max_edges - 1,
                exclude_id,
                topology.particle_id_of_vertex(vertex),
            )
            if arp3_is_attached_to_neighbors:
                return True
        return False

    @staticmethod
    def set_actin_mid_flag(topology, recipe, vertex, exclude_id=None):
        """
        if an actin near a reaction is a "mid" actin,
        add the "mid" flag, otherwise remove it

        actin is "mid" unless:
        - it is pointed, barbed, or branch
        - it is within 2 neighbors from an actin bound to arp3
        - it is neighbor of a pointed or branch actin
        """
        pt = topology.particle_type_of_vertex(vertex)
        if "pointed" in pt or "branch" in pt or "barbed" in pt:
            return
        if (
            not ActinUtil.check_arp3_attached_to_neighbors(
                topology, vertex, 3, exclude_id
            )
            and ReaddyUtil.get_neighbor_of_type(topology, vertex, "actin#branch", False)
            is None
            and ReaddyUtil.get_neighbor_of_type(
                topology, vertex, "actin#pointed", False
            )
            is None
        ):
            ReaddyUtil.set_flags(topology, recipe, vertex, ["mid"], [""], True)
        else:
            ReaddyUtil.set_flags(topology, recipe, vertex, [""], ["mid"], True)

    @staticmethod
    def get_actins_near_branch(topology, recipe, v_actin_arp2, v_actin_arp3):
        """
        get the 5 mother actins near a branch
        """
        n_pointed = ActinUtil.get_actin_number(topology, v_actin_arp2, -1)
        pointed_types = [
            f"actin#ATP_{n_pointed}",
            f"actin#{n_pointed}",
            f"actin#mid_ATP_{n_pointed}",
            f"actin#mid_{n_pointed}",
            f"actin#pointed_ATP_{n_pointed}",
            f"actin#pointed_{n_pointed}",
        ]
        if n_pointed == 1:
            pointed_types += ["actin#branch_1", "actin#branch_ATP_1"]
        v_actin_pointed = ReaddyUtil.get_neighbor_of_types(
            topology, v_actin_arp2, pointed_types, []
        )
        n_barbed = ActinUtil.get_actin_number(topology, v_actin_arp3, 1)
        barbed_types = [
            f"actin#ATP_{n_barbed}",
            f"actin#{n_barbed}",
            f"actin#mid_ATP_{n_barbed}",
            f"actin#mid_{n_barbed}",
            f"actin#barbed_ATP_{n_barbed}",
            f"actin#barbed_{n_barbed}",
        ]
        v_actin_barbed1 = ReaddyUtil.get_neighbor_of_types(
            topology, v_actin_arp3, barbed_types, [v_actin_pointed]
        )
        v_actin_barbed2 = None
        if v_actin_barbed1 is not None:
            n_barbed = ActinUtil.get_actin_number(topology, v_actin_barbed1, 1)
            barbed_types = [
                f"actin#ATP_{n_barbed}",
                f"actin#{n_barbed}",
                f"actin#mid_ATP_{n_barbed}",
                f"actin#mid_{n_barbed}",
                f"actin#barbed_ATP_{n_barbed}",
                f"actin#barbed_{n_barbed}",
            ]
            v_actin_barbed2 = ReaddyUtil.get_neighbor_of_types(
                topology, v_actin_barbed1, barbed_types, [v_actin_arp3]
            )
        return [
            v_actin_pointed,
            v_actin_arp2,
            v_actin_arp3,
            v_actin_barbed1,
            v_actin_barbed2,
        ]

    @staticmethod
    def set_actin_mid_flags_at_new_branch(topology, recipe, v_actin_arp2, v_actin_arp3):
        """
        Remove the "mid" flag on all the mother actins near a branch nucleation reaction
        """
        v_branch_actins = ActinUtil.get_actins_near_branch(
            topology, recipe, v_actin_arp2, v_actin_arp3
        )
        for v_actin in v_branch_actins:
            if v_actin is not None:
                ReaddyUtil.set_flags(topology, recipe, v_actin, [""], ["mid"], True)

    @staticmethod
    def set_actin_mid_flags_at_removed_branch(
        topology, recipe, v_actin_arp2, v_actin_arp3, v_arp3
    ):
        """
        set the "mid" state on all the actins near a branch dissociation reaction
        """
        v_branch_actins = ActinUtil.get_actins_near_branch(
            topology, recipe, v_actin_arp2, v_actin_arp3
        )
        arp3_id = topology.particle_id_of_vertex(v_arp3)
        for v_actin in v_branch_actins:
            if v_actin is not None:
                ActinUtil.set_actin_mid_flag(topology, recipe, v_actin, arp3_id)

    @staticmethod
    def add_random_linear_fibers(simulation, n_fibers, length=20, use_uuids=True):
        """
        add linear actin fibers of the given length
        """
        positions = (
            np.random.uniform(size=(n_fibers, 3)) * parameters["box_size"]
            - parameters["box_size"] * 0.5
        )
        print("Adding random fibers at \n" + str(positions))
        for fiber in range(n_fibers):
            direction = ReaddyUtil.get_random_unit_vector()
            monomers = ActinGenerator.get_monomers(
                [
                    FiberData(
                        0,
                        [
                            positions[fiber],
                            positions[fiber] + length * direction,
                        ],
                    ),
                ],
                use_uuids,
            )
            ActinUtil.add_monomers_from_data(simulation, monomers)

    @staticmethod
    def add_fibers_from_data(simulation, fibers_data, use_uuids=True):
        """
        add (branched) actin fiber(s)

        fibers_data : List[FiberData]
        """
        fiber_monomers = ActinGenerator.get_monomers(fibers_data, use_uuids)
        ActinUtil.add_monomers_from_data(simulation, fiber_monomers)

    @staticmethod
    def add_monomers_from_data(simulation, monomer_data):
        """
        add actin and other monomers

        monomer_data : {
            "topologies": {
                "[topology ID]" : {
                    "type_name": "[topology type]",
                    "particle_ids": []
                },
            "particles": {
                "[particle ID]" : {
                    "type_name": "[particle type]",
                    "position": np.zeros(3),
                    "neighbor_ids": [],
                },
            },
        }
        * IDs are uuid strings or ints
        """
        topologies = []
        for topology_id in monomer_data["topologies"]:
            topology = monomer_data["topologies"][topology_id]
            types = []
            positions = []
            for particle_id in topology["particle_ids"]:
                particle = monomer_data["particles"][particle_id]
                types.append(particle["type_name"])
                positions.append(particle["position"])
            top = simulation.add_topology(
                topology["type_name"], types, np.array(positions)
            )
            added_edges = []
            for index, particle_id in enumerate(topology["particle_ids"]):
                for neighbor_id in monomer_data["particles"][particle_id][
                    "neighbor_ids"
                ]:
                    neighbor_index = topology["particle_ids"].index(neighbor_id)
                    if (index, neighbor_index) not in added_edges and (
                        neighbor_index,
                        index,
                    ) not in added_edges:
                        top.get_graph().add_edge(index, neighbor_index)
                        added_edges.append((index, neighbor_index))
                        added_edges.append((neighbor_index, index))
            topologies.append(top)
        return topologies

    @staticmethod
    def add_actin_dimer(position, simulation):
        """
        add an actin dimer fiber
        """
        positions = np.array(
            [
                [0, 0, 0],
                ActinStructure.actin_to_actin_distance()
                * ReaddyUtil.get_random_unit_vector(),
            ]
        )
        types = ["actin#pointed_ATP_1", "actin#barbed_ATP_2"]
        top = simulation.add_topology("Actin-Dimer", types, position + positions)
        top.get_graph().add_edge(0, 1)

    @staticmethod
    def add_actin_dimers(n, simulation):
        """
        add actin dimers
        """
        positions = (
            np.random.uniform(size=(n, 3)) * parameters["box_size"]
            - parameters["box_size"] * 0.5
        )
        for p in range(len(positions)):
            ActinUtil.add_actin_dimer(positions[p], simulation)

    @staticmethod
    def get_box_positions(n_particles, particle_type):
        """
        Get random positions for n particles of the given type
        either filling the simulation volume box
        or confined to a sub volume box
        """
        if parameters[f"use_box_{particle_type}"]:
            center = np.array(
                [
                    parameters[f"{particle_type}_box_center_x"],
                    parameters[f"{particle_type}_box_center_y"],
                    parameters[f"{particle_type}_box_center_z"],
                ]
            )
            size = np.array(
                [
                    parameters[f"{particle_type}_box_size_x"],
                    parameters[f"{particle_type}_box_size_y"],
                    parameters[f"{particle_type}_box_size_z"],
                ]
            )
            result = center + (np.random.uniform(size=(n_particles, 3)) - 0.5) * size
        else:
            result = (np.random.uniform(size=(n_particles, 3)) - 0.5) * parameters[
                "box_size"
            ]
        return result

    @staticmethod
    def add_actin_monomers(n, simulation):
        """
        add free actin
        """
        positions = ActinUtil.get_box_positions(n, "actin")
        for p in range(len(positions)):
            simulation.add_topology(
                "Actin-Monomer-ATP", ["actin#free_ATP"], np.array([positions[p]])
            )

    @staticmethod
    def add_arp23_dimers(n, simulation):
        """
        add arp2/3 dimers
        """
        positions = ActinUtil.get_box_positions(n, "arp")
        for p in range(len(positions)):
            top = simulation.add_topology(
                "Arp23-Dimer-ATP",
                ["arp2#free", "arp3#ATP"],
                np.array(
                    [
                        positions[p],
                        positions[p] + 4.0 * ReaddyUtil.get_random_unit_vector(),
                    ]
                ),
            )
            top.get_graph().add_edge(0, 1)

    @staticmethod
    def add_capping_protein(n, simulation):
        """
        add free capping protein
        """
        positions = ActinUtil.get_box_positions(n, "cap")
        for p in range(len(positions)):
            simulation.add_topology("Cap", ["cap"], np.array([positions[p]]))

    @staticmethod
    def reaction_function_reverse_dimerize(topology):
        """
        reaction function for a dimer falling apart
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Reverse Dimerize")
        v_barbed = ReaddyUtil.get_first_vertex_of_types(
            topology,
            [
                "actin#barbed_ATP_1",
                "actin#barbed_ATP_2",
                "actin#barbed_ATP_3",
                "actin#barbed_1",
                "actin#barbed_2",
                "actin#barbed_3",
            ],
            error_msg="Failed to find barbed end of dimer",
        )
        v_pointed = ReaddyUtil.get_first_neighbor(
            topology, v_barbed, [], error_msg="Failed to find pointed end of dimer"
        )
        pt_barbed = topology.particle_type_of_vertex(v_barbed)
        pt_pointed = topology.particle_type_of_vertex(v_pointed)
        recipe.remove_edge(v_barbed, v_pointed)
        recipe.change_particle_type(
            v_barbed, "actin#free" + ("_ATP" if "ATP" in pt_barbed else "")
        )
        recipe.change_particle_type(
            v_pointed, "actin#free" + ("_ATP" if "ATP" in pt_pointed else "")
        )
        recipe.change_topology_type("Actin-Monomer-ATP")
        return recipe

    @staticmethod
    def reaction_function_finish_trimerize(topology):
        """
        reaction function for a trimer forming
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Trimerize")
        v_new = ActinUtil.get_new_vertex(topology)
        v_neighbor1 = ReaddyUtil.get_first_neighbor(
            topology,
            v_new,
            [],
            error_msg="Failed to find first neighbor of new vertex in trimer",
        )
        v_neighbor2 = ReaddyUtil.get_first_neighbor(
            topology,
            v_neighbor1,
            [v_new],
            error_msg="Failed to find second neighbor of new vertex in trimer",
        )
        ReaddyUtil.set_flags(
            topology,
            recipe,
            v_new,
            ["barbed", str(ActinUtil.get_actin_number(topology, v_neighbor1, 1))],
            ["new"],
            True,
        )
        ActinUtil.set_new_trimer_vertex_position(
            topology, recipe, v_new, v_neighbor2, v_neighbor1
        )
        recipe.change_topology_type("Actin-Trimer")
        return recipe

    @staticmethod
    def reaction_function_reverse_trimerize(topology):
        """
        reaction function for removing ATP-actin from a trimer
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Reverse Trimerize")
        v_barbed = ReaddyUtil.get_first_vertex_of_types(
            topology,
            [
                "actin#barbed_ATP_1",
                "actin#barbed_ATP_2",
                "actin#barbed_ATP_3",
                "actin#barbed_1",
                "actin#barbed_2",
                "actin#barbed_3",
            ],
            error_msg="Failed to find barbed end in trimer",
        )
        v_neighbor = ReaddyUtil.get_first_neighbor(
            topology,
            v_barbed,
            [],
            error_msg="Failed to find neighbor of barbed end in trimer",
        )
        recipe.remove_edge(v_barbed, v_neighbor)
        recipe.change_particle_type(v_barbed, "actin#free_ATP")
        ReaddyUtil.set_flags(topology, recipe, v_neighbor, ["barbed"], [], True)
        recipe.change_topology_type("Actin-Polymer#Shrinking")
        return recipe

    @staticmethod
    def do_finish_grow(topology, barbed):
        """
        reaction function for the pointed or barbed end growing
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        end_type = "barbed" if barbed else "pointed"
        if parameters["verbose"]:
            print("Grow " + end_type)
        v_new = ActinUtil.get_new_vertex(topology)
        v_neighbor = ReaddyUtil.get_first_neighbor(
            topology,
            v_new,
            [],
            error_msg=f"Failed to find neighbor of new {end_type} end",
        )
        if not barbed:
            v_neighbor_neighbor = ActinUtil.get_next_actin(topology, v_neighbor, 1)
            if v_neighbor_neighbor is not None:
                # previous neighbor of pointed end probably needs "mid" added
                ActinUtil.set_actin_mid_flag(topology, recipe, v_neighbor_neighbor)
        ReaddyUtil.set_flags(
            topology,
            recipe,
            v_new,
            [
                end_type,
                str(
                    ActinUtil.get_actin_number(
                        topology, v_neighbor, 1 if barbed else -1
                    )
                ),
            ],
            ["new"],
            True,
        )
        if not barbed:
            # neighbor of pointed end should never be "mid"
            ReaddyUtil.set_flags(topology, recipe, v_neighbor, [""], ["mid"], True)
        else:
            # neighbor of barbed end could be "mid"
            ActinUtil.set_actin_mid_flag(topology, recipe, v_neighbor)
        ActinUtil.set_end_vertex_position(topology, recipe, v_new, barbed)
        recipe.change_topology_type("Actin-Polymer")
        return recipe

    @staticmethod
    def reaction_function_finish_pointed_grow(topology):
        """
        reaction function for the pointed end growing
        """
        return ActinUtil.do_finish_grow(topology, False)

    @staticmethod
    def reaction_function_finish_barbed_grow(topology):
        """
        reaction function for the barbed end growing
        """
        return ActinUtil.do_finish_grow(topology, True)

    @staticmethod
    def reaction_function_finish_arp_bind(topology):
        """
        reaction function to finish a branching reaction
        (triggered by a spatial reaction)
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Bind Arp2/3")
        v_arp2, v_arp3 = ActinUtil.get_new_arp23(topology)
        if v_arp2 is None or v_arp3 is None:
            raise Exception(
                f"Failed to find new arp2 and arp3\n"
                f"{ReaddyUtil.topology_to_string(topology)}"
            )
        v_actin_arp3 = ReaddyUtil.get_first_neighbor(
            topology, v_arp3, [v_arp2], error_msg="Failed to find new actin_arp3"
        )
        # make sure arp2 binds to the pointed end neighbor of the actin bound to arp3
        v_actin_arp2 = ActinUtil.get_next_actin(
            topology, v_actin_arp3, -1, error_if_not_found=True
        )
        actin_arp2_type = topology.particle_type_of_vertex(v_actin_arp2)
        if "pointed" in actin_arp2_type or "branch" in actin_arp2_type:
            raise Exception(
                "Branch is starting exactly at a pointed end or start of a branch"
            )
        ReaddyUtil.set_flags(topology, recipe, v_arp2, [], ["free"], True)
        ReaddyUtil.set_flags(topology, recipe, v_arp3, [], ["new"], True)
        ActinUtil.set_actin_mid_flags_at_new_branch(
            topology, recipe, v_actin_arp2, v_actin_arp3
        )
        recipe.add_edge(v_actin_arp2, v_arp2)
        recipe.change_topology_type("Actin-Polymer")
        ActinUtil.set_arp23_vertex_position(
            topology, recipe, v_arp2, v_arp3, v_actin_arp2, v_actin_arp3
        )
        return recipe

    @staticmethod
    def reaction_function_finish_start_branch(topology):
        """
        reaction function for adding the first actin to an arp2/3 to start a branch
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Start Branch")
        v_new = ActinUtil.get_new_vertex(topology)
        ReaddyUtil.set_flags(
            topology, recipe, v_new, ["barbed", "1", "branch"], ["new"], True
        )
        recipe.change_topology_type("Actin-Polymer")
        ActinUtil.set_end_vertex_position(topology, recipe, v_new, True)
        return recipe

    @staticmethod
    def do_shrink(topology, barbed, atp):
        """
        remove an (ATP or ADP)-actin from the (barbed or pointed) end
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        end_state = "Barbed" if barbed else "Pointed"
        atp_state = "ATP" if atp else "ADP"
        if parameters["verbose"]:
            print(f"Shrink {end_state} {atp_state}")
        end_flag = end_state.lower()
        atp_flag = "_ATP" if atp else ""
        end_type = f"actin#{end_flag}{atp_flag}"
        v_end = ReaddyUtil.get_random_vertex_of_types(
            topology,
            ActinUtil.get_all_polymer_actin_types(end_type),
            parameters["verbose"],
            "Couldn't find end actin to remove",
        )
        if v_end is None:
            return recipe
        v_arp = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_end,
            ["arp3", "arp3#ATP", "arp2", "arp2#branched"],
            [],
            parameters["verbose"],
            "Couldn't remove actin because a branch was attached",
        )
        if v_arp is not None:
            return recipe
        v_neighbor = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_end,
            ActinUtil.get_all_polymer_actin_types("actin")
            + ActinUtil.get_all_polymer_actin_types("actin#ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#mid")
            + ActinUtil.get_all_polymer_actin_types("actin#mid_ATP")
            + ["actin#branch_1", "actin#branch_ATP_1"],
            [],
            parameters["verbose"],
            "Couldn't find plain actin neighbor of actin to remove",
        )
        if v_neighbor is None:
            return recipe
        if not barbed:
            v_arp2 = ReaddyUtil.get_neighbor_of_types(
                topology,
                v_neighbor,
                ["arp2", "arp2#branched"],
                [],
                parameters["verbose"],
                "Couldn't remove actin because a branch "
                "was attached to its barbed neighbor",
            )
            if v_arp2 is not None:
                return recipe
            v_neighbor_neighbor = ActinUtil.get_next_actin(topology, v_neighbor, 1)
            if v_neighbor_neighbor is not None:
                ReaddyUtil.set_flags(
                    topology,
                    recipe,
                    v_neighbor_neighbor,
                    [],
                    ["mid"],
                    True,
                )
        recipe.remove_edge(v_end, v_neighbor)
        recipe.change_particle_type(
            v_end, "actin#free" if not atp else "actin#free_ATP"
        )
        ReaddyUtil.set_flags(
            topology,
            recipe,
            v_neighbor,
            ["barbed"] if barbed else ["pointed"],
            ["mid"],
            True,
        )
        recipe.change_topology_type("Actin-Polymer#Shrinking")
        return recipe

    @staticmethod
    def reaction_function_pointed_shrink_ATP(topology):
        """
        reaction function to remove an ATP-actin from the pointed end
        """
        return ActinUtil.do_shrink(topology, False, True)

    @staticmethod
    def reaction_function_pointed_shrink_ADP(topology):
        """
        reaction function to remove an ADP-actin from the pointed end
        """
        return ActinUtil.do_shrink(topology, False, False)

    @staticmethod
    def reaction_function_barbed_shrink_ATP(topology):
        """
        reaction function to remove an ATP-actin from the barbed end
        """
        return ActinUtil.do_shrink(topology, True, True)

    @staticmethod
    def reaction_function_barbed_shrink_ADP(topology):
        """
        reaction function to remove an ADP-actin from the barbed end
        """
        return ActinUtil.do_shrink(topology, True, False)

    @staticmethod
    def reaction_function_cleanup_shrink(topology):
        """
        reaction function for finishing a reverse polymerization reaction
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Cleanup Shrink")
        new_type = ""
        if len(topology.graph.get_vertices()) < 2:
            v_cap = ReaddyUtil.get_vertex_of_type(topology, "cap", True)
            if v_cap is not None:
                new_type = "Cap"
            else:
                v_actin = ReaddyUtil.get_vertex_of_type(
                    topology,
                    "actin",
                    False,
                    error_msg="Failed to find actin to set monomer's ATP state",
                )
                pt_actin = topology.particle_type_of_vertex(v_actin)
                new_type = "Actin-Monomer" + ("-ATP" if "ATP" in pt_actin else "")
        elif len(topology.graph.get_vertices()) < 3:
            v_arp2 = ReaddyUtil.get_vertex_of_type(topology, "arp2#free", True)
            if v_arp2 is not None:
                v_arp3 = ReaddyUtil.get_vertex_of_type(
                    topology,
                    "arp3",
                    False,
                    error_msg="Failed to find arp3 to set arp2/3's ATP state",
                )
                pt_arp3 = topology.particle_type_of_vertex(v_arp3)
                new_type = "Arp23-Dimer" + ("-ATP" if "ATP" in pt_arp3 else "")
            else:
                new_type = "Actin-Dimer"
        elif len(topology.graph.get_vertices()) < 4:
            new_type = "Actin-Trimer"
        else:
            new_type = "Actin-Polymer"
        if parameters["verbose"]:
            print(f"cleaned up {new_type}")
        recipe.change_topology_type(new_type)
        return recipe

    @staticmethod
    def reaction_function_hydrolyze_actin(topology):
        """
        reaction function to hydrolyze a filamentous ATP-actin to ADP-actin
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Hydrolyze Actin")
        v_actin = ReaddyUtil.get_random_vertex_of_types(
            topology,
            ActinUtil.get_all_polymer_actin_types("actin#ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#pointed_ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#mid_ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#barbed_ATP")
            + ["actin#branch_barbed_ATP_1", "actin#branch_ATP_1"],
            parameters["verbose"],
            "Couldn't find ATP-actin",
        )
        if v_actin is None:
            return recipe
        ReaddyUtil.set_flags(topology, recipe, v_actin, [], ["ATP"], True)
        return recipe

    @staticmethod
    def reaction_function_hydrolyze_arp(topology):
        """
        reaction function to hydrolyze a arp2/3
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Hydrolyze Arp2/3")
        v_arp3 = ReaddyUtil.get_random_vertex_of_types(
            topology, ["arp3#ATP"], parameters["verbose"], "Couldn't find ATP-arp3"
        )
        if v_arp3 is None:
            return recipe
        ReaddyUtil.set_flags(topology, recipe, v_arp3, [], ["ATP"], True)
        return recipe

    @staticmethod
    def reaction_function_nucleotide_exchange_actin(topology):
        """
        reaction function to exchange ATP for ADP in free actin
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Nucleotide Exchange Actin")
        v_actin = ReaddyUtil.get_vertex_of_type(
            topology,
            "actin#free",
            True,
            parameters["verbose"],
            "Couldn't find ADP-actin",
        )
        if v_actin is None:
            return recipe
        ReaddyUtil.set_flags(topology, recipe, v_actin, ["ATP"], [], True)
        return recipe

    @staticmethod
    def reaction_function_nucleotide_exchange_arp(topology):
        """
        reaction function to exchange ATP for ADP in free Arp2/3
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Nucleotide Exchange Arp2/3")
        v_arp3 = ReaddyUtil.get_vertex_of_type(
            topology, "arp3", True, parameters["verbose"], "Couldn't find ADP-arp3"
        )
        if v_arp3 is None:
            return recipe
        ReaddyUtil.set_flags(topology, recipe, v_arp3, ["ATP"], [], True)
        return recipe

    @staticmethod
    def do_arp23_unbind(topology, with_ATP):
        """
        dissociate an arp2/3 from a mother filament
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        state = "ATP" if with_ATP else "ADP"
        if parameters["verbose"]:
            print(f"Remove Arp2/3 {state}")
        v_arp2 = ActinUtil.get_random_arp2(topology, with_ATP, False)
        if v_arp2 is None:
            return recipe
        actin_types = (
            ActinUtil.get_all_polymer_actin_types("actin")
            + ActinUtil.get_all_polymer_actin_types("actin#ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#pointed")
            + ActinUtil.get_all_polymer_actin_types("actin#pointed_ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#mid")
            + ActinUtil.get_all_polymer_actin_types("actin#mid_ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#barbed")
            + ActinUtil.get_all_polymer_actin_types("actin#barbed_ATP")
            + ["actin#branch_1", "actin#branch_ATP_1"]
        )
        v_actin_arp2 = ReaddyUtil.get_neighbor_of_types(
            topology, v_arp2, actin_types, [], error_msg="Failed to find actin_arp2"
        )
        v_arp3 = ReaddyUtil.get_neighbor_of_types(
            topology, v_arp2, ["arp3", "arp3#ATP"], [], error_msg="Failed to find arp3"
        )
        v_actin_arp3 = ReaddyUtil.get_neighbor_of_types(
            topology, v_arp3, actin_types, [], error_msg="Failed to find actin_arp3"
        )
        recipe.remove_edge(v_arp2, v_actin_arp2)
        recipe.remove_edge(v_arp3, v_actin_arp3)
        ActinUtil.set_actin_mid_flags_at_removed_branch(
            topology, recipe, v_actin_arp2, v_actin_arp3, v_arp3
        )
        ReaddyUtil.set_flags(topology, recipe, v_arp2, ["free"], [])
        recipe.change_topology_type("Actin-Polymer#Shrinking")
        return recipe

    @staticmethod
    def reaction_function_arp23_unbind_ATP(topology):
        """
        reaction function to dissociate an arp2/3 with ATP from a mother filament
        """
        return ActinUtil.do_arp23_unbind(topology, True)

    @staticmethod
    def reaction_function_arp23_unbind_ADP(topology):
        """
        reaction function to dissociate an arp2/3 with ADP from a mother filament
        """
        return ActinUtil.do_arp23_unbind(topology, False)

    @staticmethod
    def do_debranching(topology, with_ATP):
        """
        reaction function to detach a branch filament from arp2/3
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        state = "ATP" if with_ATP else "ADP"
        if parameters["verbose"]:
            print(f"Debranching {state}")
        v_arp2 = ActinUtil.get_random_arp2(topology, with_ATP, True)
        if v_arp2 is None:
            return recipe
        actin_types = [
            "actin#branch_1",
            "actin#branch_ATP_1",
            "actin#branch_barbed_1",
            "actin#branch_barbed_ATP_1",
        ]
        v_actin1 = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_arp2,
            actin_types,
            [],
            error_msg="Failed to find first branch actin",
        )
        recipe.remove_edge(v_arp2, v_actin1)
        ReaddyUtil.set_flags(topology, recipe, v_arp2, [], ["branched"], True)
        pt_actin1 = topology.particle_type_of_vertex(v_actin1)
        if "barbed" in pt_actin1:
            # branch is a monomer
            state = "_ATP" if "ATP" in pt_actin1 else ""
            recipe.change_particle_type(v_actin1, f"actin#free{state}")
        else:
            # branch is a filament
            ReaddyUtil.set_flags(
                topology, recipe, v_actin1, ["pointed"], ["branch"], True
            )
        recipe.change_topology_type("Actin-Polymer#Shrinking")
        return recipe

    @staticmethod
    def reaction_function_debranching_ATP(topology):
        """
        reaction function to detach a branch filament from arp2/3 with ATP
        """
        return ActinUtil.do_debranching(topology, True)

    @staticmethod
    def reaction_function_debranching_ADP(topology):
        """
        reaction function to detach a branch filament from arp2/3 with ADP
        """
        return ActinUtil.do_debranching(topology, False)

    @staticmethod
    def reaction_function_finish_cap_bind(topology):
        """
        reaction function for adding a capping protein
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Finish Cap Bind")
        v_new = ActinUtil.get_new_vertex(topology)
        ReaddyUtil.set_flags(topology, recipe, v_new, ["bound"], ["new"], True)
        recipe.change_topology_type("Actin-Polymer")
        return recipe

    @staticmethod
    def reaction_function_cap_unbind(topology):
        """
        reaction function to detach capping protein from a barbed end
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Remove Cap")
        v_cap = ReaddyUtil.get_random_vertex_of_types(
            topology, ["cap#bound"], parameters["verbose"], "Couldn't find cap"
        )
        if v_cap is None:
            return recipe
        v_actin = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_cap,
            ActinUtil.get_all_polymer_actin_types("actin")
            + ActinUtil.get_all_polymer_actin_types("actin#ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#mid")
            + ActinUtil.get_all_polymer_actin_types("actin#mid_ATP")
            + ["actin#branch_1", "actin#branch_ATP_1"],
            [],
            error_msg="Failed to find actin bound to cap",
        )
        recipe.remove_edge(v_cap, v_actin)
        ReaddyUtil.set_flags(topology, recipe, v_cap, [], ["bound"], True)
        ReaddyUtil.set_flags(topology, recipe, v_actin, ["barbed"], [], True)
        recipe.change_topology_type("Actin-Polymer#Shrinking")
        return recipe

    @staticmethod
    def reaction_function_translate(topology):
        """
        reaction function to translate particles by the displacements
        """
        global time_index
        recipe = readdy.StructuralReactionRecipe(topology)
        for vertex_id in displacements:
            v = ReaddyUtil.get_vertex_with_id(
                topology,
                vertex_id,
                error_msg=f"Couldn't find particle {vertex_id} to displace",
            )
            vertex_pos = ReaddyUtil.get_vertex_position(topology, v)
            new_pos = displacements[vertex_id]["get_translation"](
                time_index,
                vertex_id,
                vertex_pos,
                displacements[vertex_id]["parameters"],
            )
            recipe.change_particle_position(v, new_pos)
        time_index += 1
        return recipe

    @staticmethod
    def get_all_actin_particle_types(actin_number_types):
        """
        get particle types for actin

        Actin filaments are polymers and to encode polarity,there are 3 polymer types. 
        These are represented as "actin#N" where N is in [1,3]. At branch points, 
        2 particles arp2 and arp3 join the pointed end of a branch to the side 
        of its mother filament. Spatially, the types are mapped like so:

        - end                                                                    + end

        actin#pointed_1     actin#3      actin#2       actin#1    actin#barbed_3     
                    \\      //   \\      // || \\      //   \\      //
                    \\    //     \\    //  ||  \\    //     \\    //
                    actin#2      actin#1   ||   actin#3      actin#2
                                    ||     ||
                                    ||    arp3
                                    ||   //     
                                    ||  //   
                                arp2#branched
                                        \\
                                        \\
                                        actin#branch_1
                                        //
                                        //
                                    actin#2
                                        \\
                                        \\
                                        actin#barbed_3

                                            + end
        """
        result = [
            "actin#free",
            "actin#free_ATP",
            "actin#new",
            "actin#new_ATP",
            "actin#branch_1",
            "actin#branch_ATP_1",
            "actin#branch_barbed_1",
            "actin#branch_barbed_ATP_1",
        ]
        for i in ActinUtil.actin_number_range(actin_number_types):
            result += [
                f"actin#{i}",
                f"actin#ATP_{i}",
                f"actin#mid_{i}",
                f"actin#mid_ATP_{i}",
                f"actin#pointed_{i}",
                f"actin#pointed_ATP_{i}",
                f"actin#barbed_{i}",
                f"actin#barbed_ATP_{i}",
            ]
        return result

    @staticmethod
    def get_all_fixed_actin_particle_types(actin_number_types):
        """
        get particle types for actins that don't diffuse
        """
        result = []
        for i in ActinUtil.actin_number_range(actin_number_types):
            result += [
                f"actin#fixed_{i}",
                f"actin#fixed_ATP_{i}",
                f"actin#mid_fixed_{i}",
                f"actin#mid_fixed_ATP_{i}",
                f"actin#pointed_fixed_{i}",
                f"actin#pointed_fixed_ATP_{i}",
                f"actin#fixed_barbed_{i}",
                f"actin#fixed_barbed_ATP_{i}",
            ]
        return result

    @staticmethod
    def get_all_arp23_particle_types():
        """
        get particle types for Arp2/3 dimer
        """
        return [
            "arp2",
            "arp2#branched",
            "arp2#free",
            "arp3",
            "arp3#ATP",
            "arp3#new",
            "arp3#new_ATP",
        ]

    @staticmethod
    def get_all_cap_particle_types():
        """
        get particle types for capping protein
        """
        return [
            "cap",
            "cap#new",
            "cap#bound",
        ]

    @staticmethod
    def get_all_particle_types(actin_number_types):
        """
        add the given particle_types to the system
        """
        return (
            ActinUtil.get_all_actin_particle_types(actin_number_types)
            + ActinUtil.get_all_fixed_actin_particle_types(actin_number_types)
            + ActinUtil.get_all_arp23_particle_types()
            + ActinUtil.get_all_cap_particle_types()
            + ["obstacle"]
        )

    @staticmethod
    def add_particle_types(particle_types, system, diffCoeff):
        """
        add the given particle_types to the system
        """
        for particle_type in particle_types:
            system.add_topology_species(particle_type, diffCoeff)

    @staticmethod
    def add_actin_types(system, diffCoeff, actin_number_types):
        """
        add particle and topology types for actin
        """
        system.topologies.add_type("Actin-Monomer-ATP")
        system.topologies.add_type("Actin-Monomer")
        system.topologies.add_type("Actin-Dimer")
        system.topologies.add_type("Actin-Trimer")
        system.topologies.add_type("Actin-Trimer#Growing")
        system.topologies.add_type("Actin-Trimer#Shrinking")
        system.topologies.add_type("Actin-Polymer")
        system.topologies.add_type("Actin-Polymer#GrowingPointed")
        system.topologies.add_type("Actin-Polymer#GrowingBarbed")
        system.topologies.add_type("Actin-Polymer#Shrinking")
        system.topologies.add_type("Actin-Polymer#Branching")
        system.topologies.add_type("Actin-Polymer#Branch-Nucleating")
        system.topologies.add_type("Actin-Polymer#Capping")
        ActinUtil.add_particle_types(
            ActinUtil.get_all_actin_particle_types(actin_number_types), system, diffCoeff
        )
        ActinUtil.add_particle_types(
            ActinUtil.get_all_fixed_actin_particle_types(actin_number_types), system, 0.0
        )

    @staticmethod
    def add_arp23_types(system, diffCoeff):
        """
        add particle and topology types for Arp2/3 dimer
        """
        system.topologies.add_type("Arp23-Dimer-ATP")
        system.topologies.add_type("Arp23-Dimer")
        ActinUtil.add_particle_types(
            ActinUtil.get_all_arp23_particle_types(), system, diffCoeff
        )

    @staticmethod
    def add_cap_types(system, diffCoeff):
        """
        add particle and topology types for capping protein
        """
        system.topologies.add_type("Cap")
        ActinUtil.add_particle_types(
            ActinUtil.get_all_cap_particle_types(), system, diffCoeff
        )

    @staticmethod
    def add_bonds_between_actins(force_constant, system, util):
        """
        add bonds between actins
        """
        bond_length = ActinStructure.actin_to_actin_distance()
        util.add_polymer_bond_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            0,
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
                "actin#fixed_barbed_",
                "actin#fixed_barbed_ATP_",
            ],
            1,
            force_constant,
            bond_length,
            system,
        )
        util.add_bond(
            [
                "actin#branch_1",
                "actin#branch_ATP_1",
            ],
            [
                "actin#2",
                "actin#ATP_2",
                "actin#mid_2",
                "actin#mid_ATP_2",
                "actin#barbed_2",
                "actin#barbed_ATP_2",
                "actin#fixed_2",
                "actin#fixed_ATP_2",
                "actin#mid_fixed_2",
                "actin#mid_fixed_ATP_2",
                "actin#fixed_barbed_2",
                "actin#fixed_barbed_ATP_2",
            ],
            force_constant,
            bond_length,
            system,
        )
        util.add_polymer_bond_1D(  # temporary during growth reactions
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
                "actin#fixed_barbed_",
                "actin#fixed_barbed_ATP_",
            ],
            0,
            [
                "actin#new",
                "actin#new_ATP",
            ],
            None,
            force_constant,
            bond_length,
            system,
        )
        util.add_bond(  # temporary during growth reactions
            [
                "actin#branch_1",
                "actin#branch_ATP_1",
                "actin#branch_barbed_1",
                "actin#branch_barbed_ATP_1",
                "arp3#ATP",
            ],
            [
                "actin#new",
                "actin#new_ATP",
            ],
            force_constant,
            bond_length,
            system,
        )

    @staticmethod
    def add_filament_twist_angles(force_constant, system, util):
        """
        add angles for filament twist and cohesiveness
        """
        angle = ActinStructure.actin_to_actin_angle()
        util.add_polymer_angle_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            -1,
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
            ],
            0,
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
                "actin#fixed_barbed_",
                "actin#fixed_barbed_ATP_",
            ],
            1,
            force_constant,
            angle,
            system,
        )
        util.add_angle(
            [
                "actin#branch_1",
                "actin#branch_ATP_1",
            ],
            [
                "actin#2",
                "actin#ATP_2",
                "actin#fixed_2",
                "actin#fixed_ATP_2",
            ],
            [
                "actin#3",
                "actin#ATP_3",
                "actin#mid_3",
                "actin#mid_ATP_3",
                "actin#barbed_3",
                "actin#barbed_ATP_3",
                "actin#fixed_3",
                "actin#fixed_ATP_3",
                "actin#mid_fixed_3",
                "actin#mid_fixed_ATP_3",
                "actin#fixed_barbed_3",
                "actin#fixed_barbed_ATP_3",
            ],
            force_constant,
            angle,
            system,
        )

    @staticmethod
    def add_filament_twist_dihedrals(force_constant, system, util):
        """
        add dihedrals for filament twist and cohesiveness
        """
        angle = ActinStructure.actin_to_actin_dihedral_angle()
        util.add_polymer_dihedral_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            -1,
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
            ],
            0,
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
            ],
            1,
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
                "actin#fixed_barbed_",
                "actin#fixed_barbed_ATP_",
            ],
            2,
            force_constant,
            angle,
            system,
        )
        util.add_dihedral(
            [
                "actin#branch_1",
                "actin#branch_ATP_1",
            ],
            [
                "actin#2",
                "actin#ATP_2",
                "actin#mid_2",
                "actin#mid_ATP_2",
                "actin#fixed_2",
                "actin#fixed_ATP_2",
                "actin#mid_fixed_2",
                "actin#mid_fixed_ATP_2",
            ],
            [
                "actin#3",
                "actin#ATP_3",
                "actin#mid_3",
                "actin#mid_ATP_3",
                "actin#fixed_3",
                "actin#fixed_ATP_3",
                "actin#mid_fixed_3",
                "actin#mid_fixed_ATP_3",
            ],
            [
                "actin#1",
                "actin#ATP_1",
                "actin#mid_1",
                "actin#mid_ATP_1",
                "actin#barbed_1",
                "actin#barbed_ATP_1",
                "actin#fixed_1",
                "actin#fixed_ATP_1",
                "actin#mid_fixed_1",
                "actin#mid_fixed_ATP_1",
                "actin#fixed_barbed_1",
                "actin#fixed_barbed_ATP_1",
            ],
            force_constant,
            angle,
            system,
        )

    @staticmethod
    def add_branch_bonds(force_constant, system, util):
        """
        add bonds between arp2, arp3, and actins
        """
        util.add_polymer_bond_1D(  # mother filament actin to arp2 bonds
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            0,
            [
                "arp2",
                "arp2#branched",
                "arp2#free",
            ],
            None,
            force_constant,
            ActinStructure.arp2_to_mother_distance(),
            system,
        )
        util.add_polymer_bond_1D(  # mother filament actin to arp3 bonds
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
                "actin#fixed_barbed_",
                "actin#fixed_barbed_ATP_",
            ],
            0,
            [
                "arp3",
                "arp3#ATP",
                "arp3#new",
                "arp3#new_ATP",
            ],
            None,
            force_constant,
            ActinStructure.arp3_to_mother_distance(),
            system,
        )
        util.add_bond(  # arp2 to arp3 bonds
            [
                "arp2",
                "arp2#branched",
                "arp2#free",
            ],
            [
                "arp3",
                "arp3#ATP",
                "arp3#new",
                "arp3#new_ATP",
            ],
            force_constant,
            ActinStructure.arp2_to_arp3_distance(),
            system,
        )
        util.add_bond(  # arp2 to daughter filament actin bonds
            ["arp2#branched"],
            [
                "actin#branch_1",
                "actin#branch_ATP_1",
                "actin#branch_barbed_1",
                "actin#branch_barbed_ATP_1",
                "actin#new",
                "actin#new_ATP",
            ],
            force_constant,
            ActinStructure.arp2_to_daughter_distance(),
            system,
        )

    @staticmethod
    def add_branch_angles(force_constant, system, util):
        """
        add angles for branching
        """
        util.add_angle(
            ["arp3", "arp3#ATP"],
            ["arp2#branched"],
            [
                "actin#branch_1",
                "actin#branch_ATP_1",
                "actin#branch_barbed_1",
                "actin#branch_barbed_ATP_1",
            ],
            force_constant,
            ActinStructure.arp3_arp2_daughter1_angle(),
            system,
        )
        util.add_angle(
            ["arp2", "arp2#branched"],
            [
                "actin#branch_1",
                "actin#branch_ATP_1",
                "actin#branch_barbed_1",
                "actin#branch_barbed_ATP_1",
            ],
            [
                "actin#2",
                "actin#ATP_2",
                "actin#barbed_2",
                "actin#barbed_ATP_2",
                "actin#fixed_2",
                "actin#fixed_ATP_2",
                "actin#fixed_barbed_2",
                "actin#fixed_barbed_ATP_2",
            ],
            force_constant,
            ActinStructure.arp2_daughter1_daughter2_angle(),
            system,
        )
        util.add_polymer_angle_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            0,
            [
                "actin#",
                "actin#ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
            ],
            1,
            ["arp3", "arp3#ATP"],
            None,
            force_constant,
            ActinStructure.mother1_mother2_arp3_angle(),
            system,
        )
        util.add_polymer_angle_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#fixed_barbed_",
                "actin#fixed_barbed_ATP_",
            ],
            1,
            [
                "actin#",
                "actin#ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            0,
            ["arp3", "arp3#ATP"],
            None,
            force_constant,
            ActinStructure.mother3_mother2_arp3_angle(),
            system,
        )
        angle = ActinStructure.mother0_mother1_arp2_angle()
        util.add_polymer_angle_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            0,
            [
                "actin#",
                "actin#ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
            ],
            1,
            ["arp2", "arp2#branched"],
            None,
            force_constant,
            angle,
            system,
        )
        util.add_angle(
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2", "actin#fixed_2", "actin#fixed_ATP_2"],
            ["arp2", "arp2#branched"],
            force_constant,
            angle,
            system,
        )

    @staticmethod
    def add_branch_dihedrals(force_constant, system, util):
        """
        add dihedrals for branching
        """
        # mother to arp
        angle = ActinStructure.mother4_mother3_mother2_arp3_dihedral_angle()
        util.add_polymer_dihedral_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#fixed_barbed_",
                "actin#fixed_barbed_ATP_",
            ],
            1,
            ["actin#", "actin#ATP_", "actin#fixed_", "actin#fixed_ATP_"],
            0,
            ["actin#", "actin#ATP_", "actin#fixed_", "actin#fixed_ATP_"],
            -1,
            ["arp3", "arp3#ATP"],
            None,
            force_constant,
            angle,
            system,
        )
        angle = ActinStructure.mother_mother0_mother1_arp2_dihedral_angle()
        util.add_polymer_dihedral_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            -1,
            ["actin#", "actin#ATP_", "actin#fixed_", "actin#fixed_ATP_"],
            0,
            ["actin#", "actin#ATP_", "actin#fixed_", "actin#fixed_ATP_"],
            1,
            ["arp2", "arp2#branched"],
            None,
            force_constant,
            angle,
            system,
        )
        util.add_dihedral(
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2", "actin#fixed_2", "actin#fixed_ATP_2"],
            ["actin#3", "actin#ATP_3", "actin#fixed_3", "actin#fixed_ATP_3"],
            ["arp2", "arp2#branched"],
            force_constant,
            angle,
            system,
        )
        util.add_polymer_dihedral_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#fixed_barbed_",
                "actin#fixed_barbed_ATP_",
            ],
            1,
            ["actin#", "actin#ATP_", "actin#fixed_", "actin#fixed_ATP_"],
            0,
            ["arp3", "arp3#ATP"],
            None,
            ["arp2#branched", "arp2"],
            None,
            force_constant,
            ActinStructure.mother3_mother2_arp3_arp2_dihedral_angle(),
            system,
        )
        # arp ring
        angle = ActinStructure.mother1_mother2_arp3_arp2_dihedral_angle()
        util.add_polymer_dihedral_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            0,
            ["actin#", "actin#ATP_", "actin#fixed_", "actin#fixed_ATP_"],
            1,
            ["arp3", "arp3#ATP"],
            None,
            ["arp2", "arp2#branched"],
            None,
            force_constant,
            angle,
            system,
        )
        util.add_dihedral(
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2", "actin#fixed_2", "actin#fixed_ATP_2"],
            ["arp3", "arp3#ATP"],
            ["arp2", "arp2#branched"],
            force_constant,
            angle,
            system,
        )
        angle = ActinStructure.arp2_mother1_mother2_arp3_dihedral_angle()
        util.add_polymer_dihedral_1D(
            ["arp2", "arp2#branched"],
            None,
            [
                "actin#",
                "actin#ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            0,
            [
                "actin#",
                "actin#ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#fixed_barbed_",
                "actin#fixed_barbed_ATP_",
            ],
            1,
            ["arp3", "arp3#ATP"],
            None,
            force_constant,
            angle,
            system,
        )
        util.add_dihedral(
            ["arp2", "arp2#branched"],
            ["actin#branch_1", "actin#branch_ATP_1"],
            [
                "actin#2",
                "actin#ATP_2",
                "actin#barbed_2",
                "actin#barbed_ATP_2",
                "actin#fixed_2",
                "actin#fixed_ATP_2",
                "actin#fixed_barbed_2",
                "actin#fixed_barbed_ATP_2",
            ],
            ["arp3", "arp3#ATP"],
            force_constant,
            angle,
            system,
        )
        # arp to daughter
        util.add_dihedral(
            ["arp3", "arp3#ATP"],
            ["arp2#branched"],
            ["actin#branch_1", "actin#branch_ATP_1"],
            [
                "actin#2",
                "actin#ATP_2",
                "actin#barbed_2",
                "actin#barbed_ATP_2",
                "actin#fixed_2",
                "actin#fixed_ATP_2",
                "actin#fixed_barbed_2",
                "actin#fixed_barbed_ATP_2",
            ],
            force_constant,
            ActinStructure.arp3_arp2_daughter1_daughter2_dihedral_angle(),
            system,
        )
        util.add_dihedral(
            ["arp2#branched"],
            ["actin#branch_1", "actin#branch_ATP_1"],
            [
                "actin#2",
                "actin#ATP_2",
                "actin#barbed_2",
                "actin#barbed_ATP_2",
                "actin#fixed_2",
                "actin#fixed_ATP_2",
                "actin#fixed_barbed_2",
                "actin#fixed_barbed_ATP_2",
            ],
            [
                "actin#3",
                "actin#ATP_3",
                "actin#mid_3",
                "actin#mid_ATP_3",
                "actin#barbed_3",
                "actin#barbed_ATP_3",
                "actin#fixed_3",
                "actin#fixed_ATP_3",
                "actin#mid_fixed_3",
                "actin#mid_fixed_ATP_3",
                "actin#fixed_barbed_3",
                "actin#fixed_barbed_ATP_3",
            ],
            force_constant,
            ActinStructure.arp2_daughter1_daughter2_daughter3_dihedral_angle(),
            system,
        )
        # mother to daughter
        angle = ActinStructure.mother0_mother1_arp2_daughter1_dihedral_angle()
        util.add_polymer_dihedral_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            -1,
            ["actin#", "actin#ATP_", "actin#fixed_", "actin#fixed_ATP_"],
            0,
            ["arp2#branched"],
            None,
            [
                "actin#branch_1",
                "actin#branch_ATP_1",
                "actin#branch_barbed_1",
                "actin#branch_barbed_ATP_1",
            ],
            None,
            force_constant,
            angle,
            system,
        )
        util.add_dihedral(
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2", "actin#fixed_2", "actin#fixed_ATP_2"],
            ["arp2#branched"],
            [
                "actin#branch_1",
                "actin#branch_ATP_1",
                "actin#branch_barbed_1",
                "actin#branch_barbed_ATP_1",
            ],
            force_constant,
            angle,
            system,
        )
        util.add_polymer_dihedral_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#fixed_barbed_",
                "actin#fixed_barbed_ATP_",
            ],
            0,
            ["arp3", "arp3#ATP"],
            None,
            ["arp2#branched"],
            None,
            [
                "actin#branch_1",
                "actin#branch_ATP_1",
                "actin#branch_barbed_1",
                "actin#branch_barbed_ATP_1",
            ],
            None,
            force_constant,
            ActinStructure.mother2_arp3_arp2_daughter1_dihedral_angle(),
            system,
        )
        util.add_dihedral(
            [
                "actin#1",
                "actin#ATP_1",
                "actin#pointed_1",
                "actin#pointed_ATP_1",
                "actin#fixed_1",
                "actin#fixed_ATP_1",
                "actin#pointed_fixed_1",
                "actin#pointed_fixed_ATP_1",
                "actin#branch_1",
                "actin#branch_ATP_1",
                "actin#2",
                "actin#ATP_2",
                "actin#pointed_2",
                "actin#pointed_ATP_2",
                "actin#fixed_2",
                "actin#fixed_ATP_2",
                "actin#pointed_fixed_2",
                "actin#pointed_fixed_ATP_2",
                "actin#3",
                "actin#ATP_3",
                "actin#pointed_3",
                "actin#pointed_ATP_3",
                "actin#fixed_3",
                "actin#fixed_ATP_3",
                "actin#pointed_fixed_3",
                "actin#pointed_fixed_ATP_3",
            ],
            ["arp2#branched"],
            ["actin#branch_1", "actin#branch_ATP_1"],
            [
                "actin#2",
                "actin#ATP_2",
                "actin#barbed_2",
                "actin#barbed_ATP_2",
                "actin#fixed_2",
                "actin#fixed_ATP_2",
                "actin#fixed_barbed_2",
                "actin#fixed_barbed_ATP_2",
            ],
            force_constant,
            ActinStructure.mother1_arp2_daughter1_daughter2_dihedral_angle(),
            system,
        )

    @staticmethod
    def add_cap_bonds(force_constant, system, util):
        """
        add capping protein to actin bonds
        """
        util.add_polymer_bond_1D(
            ["actin#", "actin#ATP_", "actin#fixed_", "actin#fixed_ATP_"],
            0,
            ["cap#bound", "cap#new"],
            None,
            force_constant,
            ActinStructure.actin_to_actin_distance() + 1.0,
            system,
        )

    @staticmethod
    def add_cap_angles(force_constant, system, util):
        """
        add angles for capping protein
        """
        angle = ActinStructure.actin_to_actin_angle()
        util.add_polymer_angle_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            0,
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
            ],
            1,
            ["cap#bound"],
            None,
            force_constant,
            angle,
            system,
        )
        util.add_angle(
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2", "actin#fixed_2", "actin#fixed_ATP_2"],
            ["cap#bound"],
            force_constant,
            angle,
            system,
        )

    @staticmethod
    def add_cap_dihedrals(force_constant, system, util):
        """
        add dihedrals for capping protein
        """
        angle = ActinStructure.actin_to_actin_dihedral_angle()
        util.add_polymer_dihedral_1D(
            [
                "actin#",
                "actin#ATP_",
                "actin#pointed_",
                "actin#pointed_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#pointed_fixed_",
                "actin#pointed_fixed_ATP_",
            ],
            -1,
            ["actin#", "actin#ATP_", "actin#fixed_", "actin#fixed_ATP_"],
            0,
            ["actin#", "actin#ATP_", "actin#fixed_", "actin#fixed_ATP_"],
            1,
            ["cap#bound"],
            None,
            force_constant,
            angle,
            system,
        )
        util.add_dihedral(
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2", "actin#fixed_2", "actin#fixed_ATP_2"],
            ["actin#3", "actin#ATP_3", "actin#fixed_3", "actin#fixed_ATP_3"],
            ["cap#bound"],
            force_constant,
            angle,
            system,
        )
        util.add_dihedral(
            ["arp3", "arp3#ATP"],
            ["arp2#branched"],
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["cap#bound"],
            force_constant,
            ActinStructure.arp3_arp2_daughter1_daughter2_dihedral_angle(),
            system,
        )

    @staticmethod
    def add_repulsions(
        actin_radius,
        arp23_radius,
        cap_radius,
        obstacle_radius,
        force_constant,
        system,
        util,
        actin_number_types
    ):
        """
        add repulsions
        """
        actin_types = (
            ActinUtil.get_all_actin_particle_types(actin_number_types)
            + ActinUtil.get_all_fixed_actin_particle_types(actin_number_types)
        )
        arp_types = ActinUtil.get_all_arp23_particle_types()
        cap_types = ActinUtil.get_all_cap_particle_types()
        # actin
        util.add_repulsion(
            actin_types,
            actin_types,
            force_constant,
            ActinStructure.actin_to_actin_repulsion_distance(),
            system,
        )
        util.add_repulsion(
            actin_types,
            ["obstacle"],
            force_constant,
            actin_radius + obstacle_radius,
            system,
        )
        # arp2/3
        util.add_repulsion(
            arp_types,
            arp_types,
            force_constant,
            2.0 * arp23_radius,
            system,
        )
        util.add_repulsion(
            arp_types,
            actin_types,
            force_constant,
            arp23_radius + actin_radius,
            system,
        )
        util.add_repulsion(
            arp_types,
            ["obstacle"],
            force_constant,
            arp23_radius + obstacle_radius,
            system,
        )
        # capping protein
        util.add_repulsion(
            cap_types,
            cap_types,
            force_constant,
            2.0 * cap_radius,
            system,
        )
        util.add_repulsion(
            cap_types,
            actin_types,
            force_constant,
            cap_radius + actin_radius,
            system,
        )
        util.add_repulsion(
            cap_types,
            arp_types,
            force_constant,
            cap_radius + arp23_radius,
            system,
        )
        util.add_repulsion(
            cap_types,
            ["obstacle"],
            force_constant,
            cap_radius + obstacle_radius,
            system,
        )

    @staticmethod
    def add_box_potential(particle_types, origin, extent, force_constant, system):
        """
        add a box potential to keep the given particle types
        inside a box centered at origin with extent
        """
        for particle_type in particle_types:
            system.potentials.add_box(
                particle_type=particle_type,
                force_constant=force_constant,
                origin=origin,
                extent=extent,
            )

    @staticmethod
    def check_add_global_box_potential(system, actin_number_types):
        """
        If the boundaries are not periodic,
        all particles need a box potential to keep them in the box volume
        """
        if bool(parameters["periodic_boundary"]):
            return
        # 1.0 margin on each side
        box_potential_size = np.array([parameters["box_size"] - 2.0] * 3)
        ActinUtil.add_box_potential(
            particle_types=ActinUtil.get_all_particle_types(actin_number_types),
            origin=-0.5 * box_potential_size,
            extent=box_potential_size,
            force_constant=parameters["force_constant"],
            system=system,
        )

    @staticmethod
    def add_monomer_box_potentials(system):
        """
        Confine free monomers to boxes centered at origin with extent
        """
        particle_types = {
            "actin": ["actin#free", "actin#free_ATP"],
            "arp": ["arp2#free"],
            "cap": ["cap"],
        }
        for particle_type in particle_types:
            if not parameters[f"use_box_{particle_type}"]:
                continue
            print(f"Adding box for {particle_type}")
            center = np.array(
                [
                    parameters[f"{particle_type}_box_center_x"],
                    parameters[f"{particle_type}_box_center_y"],
                    parameters[f"{particle_type}_box_center_z"],
                ]
            )
            size = np.array(
                [
                    parameters[f"{particle_type}_box_size_x"],
                    parameters[f"{particle_type}_box_size_y"],
                    parameters[f"{particle_type}_box_size_z"],
                ]
            )
            ActinUtil.add_box_potential(
                particle_types[particle_type],
                center - 0.5 * size,
                size,
                parameters["force_constant"],
                system,
            )

    @staticmethod
    def add_dimerize_reaction(system):
        """
        attach two monomers
        """
        system.topologies.add_spatial_reaction(
            "Dimerize: "
            "Actin-Monomer-ATP(actin#free_ATP) + Actin-Monomer-ATP(actin#free_ATP) -> "
            "Actin-Dimer(actin#pointed_ATP_1--actin#barbed_ATP_2)",
            rate=parameters["dimerize_rate"],
            radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
        )

    @staticmethod
    def add_dimerize_reverse_reaction(system):
        """
        detach two monomers
        """
        system.topologies.add_structural_reaction(
            "Reverse_Dimerize",
            topology_type="Actin-Dimer",
            reaction_function=ActinUtil.reaction_function_reverse_dimerize,
            rate_function=lambda x: parameters["dimerize_reverse_rate"],
        )

    @staticmethod
    def add_trimerize_reaction(system, actin_number_types):
        """
        attach a monomer to a dimer
        """
        for i in ActinUtil.actin_number_range(actin_number_types):
            system.topologies.add_spatial_reaction(
                f"Trimerize{i}: "
                f"Actin-Dimer(actin#barbed_ATP_{i}) + "
                "Actin-Monomer-ATP(actin#free_ATP) -> "
                f"Actin-Trimer#Growing(actin#ATP_{i}--actin#new_ATP)",
                rate=parameters["trimerize_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )
        system.topologies.add_structural_reaction(
            "Finish_Trimerize",
            topology_type="Actin-Trimer#Growing",
            reaction_function=ActinUtil.reaction_function_finish_trimerize,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_trimerize_reverse_reaction(system):
        """
        detach a monomer from a dimer
        """
        system.topologies.add_structural_reaction(
            "Reverse_Trimerize",
            topology_type="Actin-Trimer",
            reaction_function=ActinUtil.reaction_function_reverse_trimerize,
            rate_function=lambda x: parameters["trimerize_reverse_rate"],
        )

    @staticmethod
    def add_nucleate_reaction(system, actin_number_types):
        """
        reversibly attach a monomer to a trimer
        """
        for i in ActinUtil.actin_number_range(actin_number_types):
            system.topologies.add_spatial_reaction(
                f"Barbed_Growth_Nucleate_ATP{i}: "
                f"Actin-Trimer(actin#barbed_ATP_{i}) + "
                "Actin-Monomer-ATP(actin#free_ATP) "
                f"-> Actin-Polymer#GrowingBarbed(actin#ATP_{i}--actin#new_ATP)",
                rate=parameters["nucleate_ATP_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Barbed_Growth_Nucleate_ADP{i}: "
                f"Actin-Trimer(actin#barbed_ATP_{i}) + Actin-Monomer(actin#free) -> "
                f"Actin-Polymer#GrowingBarbed(actin#ATP_{i}--actin#new)",
                rate=parameters["nucleate_ADP_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )

    @staticmethod
    def add_pointed_growth_reaction(system, actin_number_types):
        """
        attach a monomer to the pointed (-) end of a filament
        """
        for i in ActinUtil.actin_number_range(actin_number_types):
            system.topologies.add_spatial_reaction(
                f"Pointed_Growth_ATP1{i}: Actin-Polymer(actin#pointed_{i}) + "
                "Actin-Monomer-ATP(actin#free_ATP) -> "
                f"Actin-Polymer#GrowingPointed(actin#{i}--actin#new_ATP)",
                rate=parameters["pointed_growth_ATP_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Pointed_Growth_ATP2{i}: Actin-Polymer(actin#pointed_ATP_{i}) + "
                "Actin-Monomer-ATP(actin#free_ATP) -> "
                f"Actin-Polymer#GrowingPointed(actin#ATP_{i}--actin#new_ATP)",
                rate=parameters["pointed_growth_ATP_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Pointed_Growth_ADP1{i}: Actin-Polymer(actin#pointed_{i}) + "
                "Actin-Monomer(actin#free) -> "
                f"Actin-Polymer#GrowingPointed(actin#{i}--actin#new)",
                rate=parameters["pointed_growth_ADP_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Pointed_Growth_ADP2{i}: Actin-Polymer(actin#pointed_ATP_{i}) + "
                "Actin-Monomer(actin#free) -> "
                f"Actin-Polymer#GrowingPointed(actin#ATP_{i}--actin#new)",
                rate=parameters["pointed_growth_ADP_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )
        system.topologies.add_structural_reaction(
            "Finish_Pointed_Growth",
            topology_type="Actin-Polymer#GrowingPointed",
            reaction_function=ActinUtil.reaction_function_finish_pointed_grow,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_pointed_shrink_reaction(system):
        """
        remove a monomer from the pointed (-) end of a filament
        """
        system.topologies.add_structural_reaction(
            "Pointed_Shrink_ATP",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_pointed_shrink_ATP,
            rate_function=lambda x: parameters["pointed_shrink_ATP_rate"],
        )
        system.topologies.add_structural_reaction(
            "Pointed_Shrink_ADP",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_pointed_shrink_ADP,
            rate_function=lambda x: parameters["pointed_shrink_ADP_rate"],
        )
        system.topologies.add_structural_reaction(
            "Cleanup_Shrink",
            topology_type="Actin-Polymer#Shrinking",
            reaction_function=ActinUtil.reaction_function_cleanup_shrink,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_barbed_growth_reaction(system, actin_number_types):
        """
        attach a monomer to the barbed (+) end of a filament
        """
        for i in ActinUtil.actin_number_range(actin_number_types):
            system.topologies.add_spatial_reaction(
                f"Barbed_Growth_ATP1{i}: Actin-Polymer(actin#barbed_{i}) + "
                "Actin-Monomer-ATP(actin#free_ATP) -> "
                f"Actin-Polymer#GrowingBarbed(actin#{i}--actin#new_ATP)",
                rate=parameters["barbed_growth_ATP_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Barbed_Growth_ATP2{i}: Actin-Polymer(actin#barbed_ATP_{i}) + "
                "Actin-Monomer-ATP(actin#free_ATP) -> "
                f"Actin-Polymer#GrowingBarbed(actin#ATP_{i}--actin#new_ATP)",
                rate=parameters["barbed_growth_ATP_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Barbed_Growth_ADP1{i}: Actin-Polymer(actin#barbed_{i}) + "
                "Actin-Monomer(actin#free) -> "
                f"Actin-Polymer#GrowingBarbed(actin#{i}--actin#new)",
                rate=parameters["barbed_growth_ADP_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Barbed_Growth_ADP2{i}: Actin-Polymer(actin#barbed_ATP_{i}) + "
                "Actin-Monomer(actin#free) -> "
                f"Actin-Polymer#GrowingBarbed(actin#ATP_{i}--actin#new)",
                rate=parameters["barbed_growth_ADP_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )
        system.topologies.add_spatial_reaction(
            "Branch_Barbed_Growth_ATP1: Actin-Polymer(actin#branch_barbed_1) + "
            "Actin-Monomer-ATP(actin#free_ATP) -> "
            "Actin-Polymer#GrowingBarbed(actin#branch_1--actin#new_ATP)",
            rate=parameters["barbed_growth_ATP_rate"],
            radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
        )
        system.topologies.add_spatial_reaction(
            "Branch_Barbed_Growth_ATP2: Actin-Polymer(actin#branch_barbed_ATP_1) + "
            "Actin-Monomer-ATP(actin#free_ATP) -> "
            "Actin-Polymer#GrowingBarbed(actin#branch_ATP_1--actin#new_ATP)",
            rate=parameters["barbed_growth_ATP_rate"],
            radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
        )
        system.topologies.add_spatial_reaction(
            "Branch_Barbed_Growth_ADP1: Actin-Polymer(actin#branch_barbed_1) + "
            "Actin-Monomer(actin#free) -> "
            "Actin-Polymer#GrowingBarbed(actin#branch_1--actin#new)",
            rate=parameters["barbed_growth_ADP_rate"],
            radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
        )
        system.topologies.add_spatial_reaction(
            "Branch_Barbed_Growth_ADP2: Actin-Polymer(actin#branch_barbed_ATP_1) + "
            "Actin-Monomer(actin#free) -> "
            "Actin-Polymer#GrowingBarbed(actin#branch_ATP_1--actin#new)",
            rate=parameters["barbed_growth_ADP_rate"],
            radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
        )
        system.topologies.add_structural_reaction(
            "Finish_Barbed_growth",
            topology_type="Actin-Polymer#GrowingBarbed",
            reaction_function=ActinUtil.reaction_function_finish_barbed_grow,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_barbed_shrink_reaction(system):
        """
        remove a monomer from the barbed (+) end of a filament
        """
        system.topologies.add_structural_reaction(
            "Barbed_Shrink_ATP",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_barbed_shrink_ATP,
            rate_function=lambda x: parameters["barbed_shrink_ATP_rate"],
        )
        system.topologies.add_structural_reaction(
            "Barbed_Shrink_ADP",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_barbed_shrink_ADP,
            rate_function=lambda x: parameters["barbed_shrink_ADP_rate"],
        )

    @staticmethod
    def add_hydrolyze_reaction(system):
        """
        hydrolyze ATP
        """
        system.topologies.add_structural_reaction(
            "Hydrolysis_Actin",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_hydrolyze_actin,
            rate_function=lambda x: parameters["hydrolysis_actin_rate"],
        )
        system.topologies.add_structural_reaction(
            "Hydrolysis_Arp",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_hydrolyze_arp,
            rate_function=lambda x: parameters["hydrolysis_arp_rate"],
        )

    @staticmethod
    def add_actin_nucleotide_exchange_reaction(system):
        """
        exchange ATP for ADP in free actin monomers
        """
        system.topologies.add_structural_reaction(
            "Nucleotide_Exchange_Actin",
            topology_type="Actin-Monomer",
            reaction_function=ActinUtil.reaction_function_nucleotide_exchange_actin,
            rate_function=lambda x: parameters["nucleotide_exchange_actin_rate"],
        )

    @staticmethod
    def add_arp23_nucleotide_exchange_reaction(system):
        """
        exchange ATP for ADP in free Arp2/3 dimers
        """
        system.topologies.add_structural_reaction(
            "Nucleotide_Exchange_Arp",
            topology_type="Arp23-Dimer",
            reaction_function=ActinUtil.reaction_function_nucleotide_exchange_arp,
            rate_function=lambda x: parameters["nucleotide_exchange_arp_rate"],
        )

    @staticmethod
    def add_arp23_bind_reaction(system):
        """
        add arp2/3 along filament to start a branch
        """
        for i in range(1, 4):
            system.topologies.add_spatial_reaction(
                f"Arp_Bind_ATP1{i}: "
                f"Actin-Polymer(actin#mid_ATP_{i}) + Arp23-Dimer(arp3) -> "
                f"Actin-Polymer#Branching(actin#ATP_{i}--arp3#new)",
                rate=parameters["arp_bind_ATP_rate"],
                radius=parameters["actin_radius"]
                + parameters["arp23_radius"]
                + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Arp_Bind_ATP2{i}: "
                f"Actin-Polymer(actin#mid_ATP_{i}) + Arp23-Dimer-ATP(arp3#ATP) -> "
                f"Actin-Polymer#Branching(actin#ATP_{i}--arp3#new_ATP)",
                rate=parameters["arp_bind_ATP_rate"],
                radius=parameters["actin_radius"]
                + parameters["arp23_radius"]
                + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Arp_Bind_ADP1{i}: "
                f"Actin-Polymer(actin#mid_{i}) + Arp23-Dimer(arp3) -> "
                f"Actin-Polymer#Branching(actin#{i}--arp3#new)",
                rate=parameters["arp_bind_ADP_rate"],
                radius=parameters["actin_radius"]
                + parameters["arp23_radius"]
                + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Arp_Bind_ADP2{i}: "
                f"Actin-Polymer(actin#mid_{i}) + Arp23-Dimer-ATP(arp3#ATP) -> "
                f"Actin-Polymer#Branching(actin#{i}--arp3#new_ATP)",
                rate=parameters["arp_bind_ADP_rate"],
                radius=parameters["actin_radius"]
                + parameters["arp23_radius"]
                + parameters["reaction_distance"],
            )
        system.topologies.add_structural_reaction(
            "Finish_Arp_Bind",
            topology_type="Actin-Polymer#Branching",
            reaction_function=ActinUtil.reaction_function_finish_arp_bind,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_arp23_unbind_reaction(system):
        """
        remove an arp2/3 that is not nucleated
        """
        system.topologies.add_structural_reaction(
            "Arp_Unbind_ATP",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_arp23_unbind_ATP,
            rate_function=lambda x: parameters["arp_unbind_ATP_rate"],
        )
        system.topologies.add_structural_reaction(
            "Arp_Unbind_ADP",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_arp23_unbind_ADP,
            rate_function=lambda x: parameters["arp_unbind_ADP_rate"],
        )

    @staticmethod
    def add_nucleate_branch_reaction(system):
        """
        add actin to arp2/3 to begin a branch
        """
        system.topologies.add_spatial_reaction(
            "Barbed_Growth_Branch_ATP: "
            "Actin-Polymer(arp2) + Actin-Monomer-ATP(actin#free_ATP) -> "
            "Actin-Polymer#Branch-Nucleating(arp2#branched--actin#new_ATP)",
            rate=parameters["barbed_growth_branch_ATP_rate"],
            radius=parameters["actin_radius"]
            + parameters["arp23_radius"]
            + parameters["reaction_distance"],
        )
        system.topologies.add_spatial_reaction(
            "Barbed_Growth_Branch_ADP: "
            "Actin-Polymer(arp2) + Actin-Monomer(actin#free) -> "
            "Actin-Polymer#Branch-Nucleating(arp2#branched--actin#new)",
            rate=parameters["barbed_growth_branch_ADP_rate"],
            radius=parameters["actin_radius"]
            + parameters["arp23_radius"]
            + parameters["reaction_distance"],
        )
        system.topologies.add_structural_reaction(
            "Nucleate_Branch",
            topology_type="Actin-Polymer#Branch-Nucleating",
            reaction_function=ActinUtil.reaction_function_finish_start_branch,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_debranch_reaction(system):
        """
        remove a branch
        """
        system.topologies.add_structural_reaction(
            "Debranch_ATP",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_debranching_ATP,
            rate_function=lambda x: parameters["debranching_ATP_rate"],
        )
        system.topologies.add_structural_reaction(
            "Debranch_ADP",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_debranching_ADP,
            rate_function=lambda x: parameters["debranching_ADP_rate"],
        )

    @staticmethod
    def add_cap_bind_reaction(system):
        """
        add capping protein to a barbed end to stop growth
        """
        for i in range(1, 4):
            system.topologies.add_spatial_reaction(
                f"Cap_Bind1{i}: Actin-Polymer(actin#barbed_{i}) + Cap(cap) -> "
                f"Actin-Polymer#Capping(actin#{i}--cap#new)",
                rate=parameters["cap_bind_rate"],
                radius=parameters["actin_radius"]
                + parameters["cap_radius"]
                + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Cap_Bind2{i}: Actin-Polymer(actin#barbed_ATP_{i}) + Cap(cap) -> "
                f"Actin-Polymer#Capping(actin#ATP_{i}--cap#new)",
                rate=parameters["cap_bind_rate"],
                radius=parameters["actin_radius"]
                + parameters["cap_radius"]
                + parameters["reaction_distance"],
            )
        system.topologies.add_structural_reaction(
            "Finish_Cap_Bind",
            topology_type="Actin-Polymer#Capping",
            reaction_function=ActinUtil.reaction_function_finish_cap_bind,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_cap_unbind_reaction(system):
        """
        remove capping protein
        """
        system.topologies.add_structural_reaction(
            "Cap_Unbind",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_cap_unbind,
            rate_function=lambda x: parameters["cap_unbind_rate"],
        )

    @staticmethod
    def add_translate_reaction(system):
        """
        translate particles by the displacements each timestep
        """
        system.topologies.add_structural_reaction(
            "Translate",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_translate,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def get_position_for_tangent_translation(
        time_index, monomer_id, monomer_pos, displacement_parameters
    ):
        return monomer_pos + (
            displacement_parameters["total_displacement_nm"]
            / displacement_parameters["total_steps"]
        )

    @staticmethod
    def get_position_for_radial_translation(
        time_index, monomer_id, monomer_pos, displacement_parameters
    ):
        global init_monomer_positions, pointed_monomer_positions
        if time_index == 0 and monomer_id > 0:
            init_monomer_positions[monomer_id] = monomer_pos
        if monomer_id == 0:
            pointed_monomer_positions.append(monomer_pos)
        radius = displacement_parameters["radius_nm"]
        theta_init = displacement_parameters["theta_init_radians"]
        theta_final = displacement_parameters["theta_final_radians"]
        d_theta = (theta_final - theta_init) / displacement_parameters["total_steps"]
        theta_1 = theta_init + time_index * d_theta
        theta_2 = theta_init + (time_index + 1) * d_theta
        dx = radius * (np.cos(theta_2) - np.cos(theta_1))
        dy = radius * (np.sin(theta_2) - np.sin(theta_1))
        d_pos_0 = np.array([dx, dy, 0])
        if monomer_id == 0:
            return monomer_pos + d_pos_0
        v_pos_init = init_monomer_positions[monomer_id] - pointed_monomer_positions[0]
        pos_magnitude = np.linalg.norm(v_pos_init)
        v_pos_init = ReaddyUtil.normalize(v_pos_init)
        v_pos_1 = ReaddyUtil.rotate(
            v=v_pos_init,
            axis=np.array([0, 0, 1]),
            angle=time_index * d_theta,
        )
        v_pos_2 = ReaddyUtil.rotate(
            v=v_pos_init,
            axis=np.array([0, 0, 1]),
            angle=(time_index + 1) * d_theta,
        )
        return (monomer_pos - pos_magnitude * v_pos_1) + (
            d_pos_0 + pos_magnitude * v_pos_2
        )

    @staticmethod
    def actin_number_range(actin_number_types):
        if (actin_number_types < 3 or actin_number_types > 5) or actin_number_types == 4:
            raise Exception("Only polymer number values of 3 and 5 are supported.")
        return range(1, actin_number_types+1)