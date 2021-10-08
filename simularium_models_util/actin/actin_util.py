#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import readdy
import random

from ..common import ReaddyUtil
from .actin_generator import ActinGenerator
from .actin_structure import ActinStructure
from .fiber_data import FiberData


parameters = {}


def set_parameters(p):
    global parameters
    parameters = p
    return p


class ActinUtil:
    def __init__(self, parameters):
        """
        Utility functions for ReaDDy branched actin models

        Parameters need to be accessible in ReaDDy callbacks
        which can't be instance methods, so parameters are global
        """
        set_parameters(parameters)

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
    def cancel_branch_reaction(topology, recipe, actin_arp3, arp3):
        """
        Undo the branching spatial reaction if the structural reaction fails
        """
        if parameters["verbose"]:
            print("Canceling branch reaction")
        pt = topology.particle_type_of_vertex(actin_arp3)
        recipe.remove_edge(actin_arp3, arp3)
        ReaddyUtil.set_flags(topology, recipe, arp3, [], ["new"], True)
        state = "ATP" if "ATP" in pt else "ADP"
        recipe.change_topology_type(f"Actin-Polymer#Fail-Branch-{state}")

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
        ]

    @staticmethod
    def get_actin_rotation(positions, box_size):
        """
        get the difference in the actin's current orientation
        compared to the initial orientation as a rotation matrix
        positions = [prev actin position, middle actin position, next actin position]
        """
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
    def get_actin_axis_position(positions, box_size):
        """
        get the position on the filament axis closest to an actin
        positions = [
            previous actin position,
            middle actin position,
            next actin position
        ]
        """
        rotation = ActinUtil.get_actin_rotation(positions, box_size)
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
        rotation = ActinUtil.get_actin_rotation(positions, parameters["box_size"])
        if rotation is None:
            return None
        vector_to_new_pos = np.squeeze(np.array(np.dot(rotation, offset_vector)))
        return (positions[1] + vector_to_new_pos).tolist()

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
    def add_random_linear_fibers(simulation, n_fibers, length=20, use_uuids=True):
        """
        add linear actin fibers of the given length
        """
        positions = (
            np.random.uniform(size=(n_fibers, 3)) * parameters["box_size"]
            - parameters["box_size"] * 0.5
        )
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
            test = []
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
                        test.append((index, neighbor_index))

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
                "Actin-Monomer", ["actin#free_ATP"], np.array([positions[p]])
            )

    @staticmethod
    def add_arp23_dimers(n, simulation):
        """
        add arp2/3 dimers
        """
        positions = ActinUtil.get_box_positions(n, "arp")
        for p in range(len(positions)):
            top = simulation.add_topology(
                "Arp23-Dimer",
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
        recipe.remove_edge(v_barbed, v_pointed)
        recipe.change_particle_type(v_barbed, "actin#free_ATP")
        recipe.change_particle_type(v_pointed, "actin#free_ATP")
        recipe.change_topology_type("Actin-Monomer")
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
    def do_nonspatial_growth(
        recipe, topology, end_type, with_ATP, exact_end_type=False
    ):
        """
        add an implicit monomer to an arp2 or to a barbed or pointed end

        "Actin-Polymer(arp2) + Actin-Monomer(actin#free) -> "
            "Actin-Polymer#Branch-Nucleating(arp2#branched--actin#new)",
        """
        v_end = ReaddyUtil.get_random_vertex_of_type(
            topology,
            end_type,
            exact_end_type,
            parameters["verbose"],
            "Couldn't find end monomer",
        )
        if v_end is None:
            return False
        v_neighbor = ReaddyUtil.get_first_neighbor(
            topology, v_end, [], error_msg="Failed to find neighbor of end"
        )
        pos_end = ReaddyUtil.get_vertex_position(topology, v_end)
        pos_neighbor = ReaddyUtil.get_vertex_position(topology, v_neighbor)
        v_neighbor_to_end = pos_end - pos_neighbor
        recipe.append_particle(
            [v_end],
            "actin#new_ATP" if with_ATP else "actin#new",
            pos_end + v_neighbor_to_end,
        )
        if end_type == "arp2":
            ReaddyUtil.set_flags(topology, recipe, v_end, ["branched"], [], True)
        else:
            ReaddyUtil.set_flags(topology, recipe, v_end, [], [end_type], True)
        return True

    @staticmethod
    def reaction_function_nonspatial_trimerize(topology):
        """
        reaction function for adding an implicit monomer to a dimer
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if ActinUtil.do_nonspatial_growth(recipe, topology, "barbed", True):
            recipe.change_topology_type("Actin-Trimer#Growing")
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
    def reaction_function_finish_pointed_grow(topology):
        """
        reaction function for the pointed end growing
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Grow Pointed")
        v_new = ActinUtil.get_new_vertex(topology)
        v_neighbor = ReaddyUtil.get_first_neighbor(
            topology, v_new, [], error_msg="Failed to find neighbor of new pointed end"
        )
        ReaddyUtil.set_flags(
            topology,
            recipe,
            v_new,
            ["pointed", str(ActinUtil.get_actin_number(topology, v_neighbor, -1))],
            ["new"],
            True,
        )
        recipe.change_topology_type("Actin-Polymer")
        ActinUtil.set_end_vertex_position(topology, recipe, v_new, False)
        return recipe

    @staticmethod
    def reaction_function_nonspatial_pointed_grow_ATP(topology):
        """
        reaction function for adding an implicit ATP-monomer to a pointed end
        """
        if parameters["verbose"]:
            print("(nonspatial) Pointed Grow ATP")
        recipe = readdy.StructuralReactionRecipe(topology)
        if ActinUtil.do_nonspatial_growth(recipe, topology, "pointed", True):
            recipe.change_topology_type("Actin-Polymer#GrowingPointed")
        return recipe

    @staticmethod
    def reaction_function_nonspatial_pointed_grow_ADP(topology):
        """
        reaction function for adding an implicit ADP-monomer to a pointed end
        """
        if parameters["verbose"]:
            print("(nonspatial) Pointed Grow ADP")
        recipe = readdy.StructuralReactionRecipe(topology)
        if ActinUtil.do_nonspatial_growth(recipe, topology, "pointed", False):
            recipe.change_topology_type("Actin-Polymer#GrowingPointed")
        return recipe

    @staticmethod
    def reaction_function_finish_barbed_grow(topology):
        """
        reaction function for the barbed end growing
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Grow Barbed")
        v_new = ActinUtil.get_new_vertex(topology)
        v_neighbor = ReaddyUtil.get_first_neighbor(
            topology, v_new, [], error_msg="Failed to find neighbor of new barbed end"
        )
        ReaddyUtil.set_flags(
            topology,
            recipe,
            v_new,
            ["barbed", str(ActinUtil.get_actin_number(topology, v_neighbor, 1))],
            ["new"],
            True,
        )
        ActinUtil.set_end_vertex_position(topology, recipe, v_new, True)
        recipe.change_topology_type("Actin-Polymer")
        return recipe

    @staticmethod
    def reaction_function_nonspatial_barbed_grow_ATP(topology):
        """
        reaction function for adding an implicit ATP-monomer to a barbed end
        """
        if parameters["verbose"]:
            print("(nonspatial) Barbed Grow ATP")
        recipe = readdy.StructuralReactionRecipe(topology)
        if ActinUtil.do_nonspatial_growth(recipe, topology, "barbed", True):
            recipe.change_topology_type("Actin-Polymer#GrowingBarbed")
        return recipe

    @staticmethod
    def reaction_function_nonspatial_barbed_grow_ADP(topology):
        """
        reaction function for adding an implicit ADP-monomer to a barbed end
        """
        if parameters["verbose"]:
            print("(nonspatial) Barbed Grow ADP")
        recipe = readdy.StructuralReactionRecipe(topology)
        if ActinUtil.do_nonspatial_growth(recipe, topology, "barbed", False):
            recipe.change_topology_type("Actin-Polymer#GrowingBarbed")
        return recipe

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
        v_actin_barbed = ReaddyUtil.get_first_neighbor(
            topology, v_arp3, [v_arp2], error_msg="Failed to find new actin_arp3"
        )
        # make sure arp3 binds to the barbed end neighbor of the actin bound to arp2
        n_pointed = ActinUtil.get_actin_number(topology, v_actin_barbed, -1)
        actin_types = [
            f"actin#ATP_{n_pointed}",
            f"actin#{n_pointed}",
            f"actin#pointed_ATP_{n_pointed}",
            f"actin#pointed_{n_pointed}",
        ]
        if n_pointed == 1:
            actin_types += ["actin#branch_1", "actin#branch_ATP_1"]
        v_actin_pointed = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_actin_barbed,
            actin_types,
            [],
            parameters["verbose"],
            f"Couldn't find actin_arp2 with number {n_pointed}",
        )
        if v_actin_pointed is not None:
            pointed_type = topology.particle_type_of_vertex(v_actin_pointed)
            if "pointed" in pointed_type or "branch" in pointed_type:
                if parameters["verbose"]:
                    print("Branch is starting exactly at a pointed end")
                ActinUtil.cancel_branch_reaction(
                    topology, recipe, v_actin_barbed, v_arp3
                )
                return recipe
        else:
            ActinUtil.cancel_branch_reaction(topology, recipe, v_actin_barbed, v_arp3)
            return recipe
        ReaddyUtil.set_flags(topology, recipe, v_arp2, [], ["free"], True)
        ReaddyUtil.set_flags(topology, recipe, v_arp3, [], ["new"], True)
        recipe.add_edge(v_actin_pointed, v_arp2)
        recipe.change_topology_type("Actin-Polymer")
        ActinUtil.set_arp23_vertex_position(
            topology, recipe, v_arp2, v_arp3, v_actin_pointed, v_actin_barbed
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
    def reaction_function_nonspatial_nucleate_branch_ATP(topology):
        """
        reaction function for adding an implicit ATP-monomer to arp2/3 to begin a branch
        """
        if parameters["verbose"]:
            print("(nonspatial) Nucleate Branch ATP")
        recipe = readdy.StructuralReactionRecipe(topology)
        if ActinUtil.do_nonspatial_growth(recipe, topology, "arp2", True, True):
            recipe.change_topology_type("Actin-Polymer#Branch-Nucleating")
        return recipe

    @staticmethod
    def reaction_function_nonspatial_nucleate_branch_ADP(topology):
        """
        reaction function for adding an implicit ADP-monomer to arp2/3 to begin a branch
        """
        if parameters["verbose"]:
            print("(nonspatial) Nucleate Branch ADP")
        recipe = readdy.StructuralReactionRecipe(topology)
        if ActinUtil.do_nonspatial_growth(recipe, topology, "arp2", False, True):
            recipe.change_topology_type("Actin-Polymer#Branch-Nucleating")
        return recipe

    @staticmethod
    def do_shrink(topology, recipe, barbed, ATP):
        """
        remove an (ATP or ADP)-actin from the (barbed or pointed) end
        """
        end_state = "barbed" if barbed else "pointed"
        atp_state = "_ATP" if ATP else ""
        end_type = f"actin#{end_state}{atp_state}"
        v_end = ReaddyUtil.get_random_vertex_of_types(
            topology,
            ActinUtil.get_all_polymer_actin_types(end_type),
            parameters["verbose"],
            "Couldn't find end actin to remove",
        )
        if v_end is None:
            return False
        v_arp = ReaddyUtil.get_neighbor_of_types(
            topology, v_end, ["arp3", "arp3#ATP", "arp2", "arp2#branched"], []
        )
        if v_arp is not None:
            if parameters["verbose"]:
                print("Couldn't remove actin because a branch was attached")
            return False
        v_neighbor = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_end,
            ActinUtil.get_all_polymer_actin_types("actin")
            + ActinUtil.get_all_polymer_actin_types("actin#ATP")
            + ["actin#branch_1", "actin#branch_ATP_1"],
            [],
            parameters["verbose"],
            "Couldn't find plain actin neighbor of actin to remove",
        )
        if v_neighbor is None:
            return False
        if not barbed:
            v_arp2 = ReaddyUtil.get_neighbor_of_types(
                topology, v_neighbor, ["arp2", "arp2#branched"], []
            )
            if v_arp2 is not None:
                if parameters["verbose"]:
                    print(
                        "Couldn't remove actin because a branch was attached "
                        "to its barbed neighbor"
                    )
                return False
        recipe.remove_edge(v_end, v_neighbor)
        recipe.change_particle_type(
            v_end, "actin#free" if not ATP else "actin#free_ATP"
        )
        ReaddyUtil.set_flags(
            topology,
            recipe,
            v_neighbor,
            ["barbed"] if barbed else ["pointed"],
            [],
            True,
        )
        recipe.change_topology_type("Actin-Polymer#Shrinking")
        return True

    @staticmethod
    def reaction_function_pointed_shrink_ATP(topology):
        """
        reaction function to remove an ATP-actin from the pointed end
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Shrink pointed ATP")
        if not ActinUtil.do_shrink(topology, recipe, False, True):
            recipe.change_topology_type("Actin-Polymer#Fail-Pointed-Shrink-ATP")
        return recipe

    @staticmethod
    def reaction_function_pointed_shrink_ADP(topology):
        """
        reaction function to remove an ADP-actin from the pointed end
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Shrink pointed ADP")
        if not ActinUtil.do_shrink(topology, recipe, False, False):
            recipe.change_topology_type("Actin-Polymer#Fail-Pointed-Shrink-ADP")
        return recipe

    @staticmethod
    def reaction_function_barbed_shrink_ATP(topology):
        """
        reaction function to remove an ATP-actin from the barbed end
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Shrink barbed ATP")
        if not ActinUtil.do_shrink(topology, recipe, True, True):
            recipe.change_topology_type("Actin-Polymer#Fail-Barbed-Shrink-ATP")
        return recipe

    @staticmethod
    def reaction_function_barbed_shrink_ADP(topology):
        """
        reaction function to remove an ADP-actin from the barbed end
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Shrink barbed ADP")
        if not ActinUtil.do_shrink(topology, recipe, True, False):
            recipe.change_topology_type("Actin-Polymer#Fail-Barbed-Shrink-ADP")
        return recipe

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
                new_type = "Actin-Monomer"
        elif len(topology.graph.get_vertices()) < 3:
            v_arp2 = ReaddyUtil.get_vertex_of_type(topology, "arp2#free", True)
            if v_arp2 is not None:
                new_type = "Arp23-Dimer"
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
        v = ReaddyUtil.get_random_vertex_of_types(
            topology,
            ActinUtil.get_all_polymer_actin_types("actin#ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#pointed_ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#barbed_ATP")
            + ["actin#branch_barbed_ATP_1", "actin#branch_ATP_1"],
            parameters["verbose"],
            "Couldn't find ATP-actin",
        )
        if v is None:
            recipe.change_topology_type("Actin-Polymer#Fail-Hydrolysis-Actin")
            return recipe
        ReaddyUtil.set_flags(topology, recipe, v, [], ["ATP"], True)
        return recipe

    @staticmethod
    def reaction_function_hydrolyze_arp(topology):
        """
        reaction function to hydrolyze a arp2/3
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Hydrolyze Arp2/3")
        v = ReaddyUtil.get_random_vertex_of_types(
            topology, ["arp3#ATP"], parameters["verbose"], "Couldn't find ATP-arp3"
        )
        if v is None:
            recipe.change_topology_type("Actin-Polymer#Fail-Hydrolysis-Arp")
            return recipe
        ReaddyUtil.set_flags(topology, recipe, v, [], ["ATP"], True)
        return recipe

    @staticmethod
    def reaction_function_nucleotide_exchange_actin(topology):
        """
        reaction function to exchange ATP for ADP in free actin
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Nucleotide Exchange Actin")
        v = ReaddyUtil.get_vertex_of_type(
            topology,
            "actin#free",
            True,
            parameters["verbose"],
            "Couldn't find ADP-actin",
        )
        if v is None:
            recipe.change_topology_type("Actin-Polymer#Fail-Nucleotide-Exchange-Actin")
            return recipe
        ReaddyUtil.set_flags(topology, recipe, v, ["ATP"], [], True)
        return recipe

    @staticmethod
    def reaction_function_nucleotide_exchange_arp(topology):
        """
        reaction function to exchange ATP for ADP in free Arp2/3
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Nucleotide Exchange Arp2/3")
        v = ReaddyUtil.get_vertex_of_type(
            topology, "arp3", True, parameters["verbose"], "Couldn't find ADP-arp3"
        )
        if v is None:
            recipe.change_topology_type("Actin-Polymer#Fail-Nucleotide-Exchange-Arp")
            return recipe
        ReaddyUtil.set_flags(topology, recipe, v, ["ATP"], [], True)
        return recipe

    @staticmethod
    def do_arp23_unbind(topology, recipe, with_ATP):
        """
        dissociate an arp2/3 from a mother filament
        """
        v_arp2 = ActinUtil.get_random_arp2(topology, with_ATP, False)
        if v_arp2 is None:
            state = "ATP" if with_ATP else "ADP"
            recipe.change_topology_type("Actin-Polymer#Fail-Arp-Unbind-" + state)
            if parameters["verbose"]:
                print(f"Couldn't find unbranched {state}-arp2")
            return recipe
        actin_types = (
            ActinUtil.get_all_polymer_actin_types("actin")
            + ActinUtil.get_all_polymer_actin_types("actin#ATP")
            + ActinUtil.get_all_polymer_actin_types("actin#pointed")
            + ActinUtil.get_all_polymer_actin_types("actin#pointed_ATP")
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
        ReaddyUtil.set_flags(topology, recipe, v_arp2, ["free"], [])
        recipe.change_topology_type("Actin-Polymer#Shrinking")

    @staticmethod
    def reaction_function_arp23_unbind_ATP(topology):
        """
        reaction function to dissociate an arp2/3 with ATP from a mother filament
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Remove Arp2/3 ATP")
        ActinUtil.do_arp23_unbind(topology, recipe, True)
        return recipe

    @staticmethod
    def reaction_function_arp23_unbind_ADP(topology):
        """
        reaction function to dissociate an arp2/3 with ADP from a mother filament
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Remove Arp2/3 ADP")
        ActinUtil.do_arp23_unbind(topology, recipe, False)
        return recipe

    @staticmethod
    def do_debranching(topology, recipe, with_ATP):
        """
        reaction function to detach a branch filament from arp2/3
        """
        v_arp2 = ActinUtil.get_random_arp2(topology, with_ATP, True)
        if v_arp2 is None:
            state = "ATP" if with_ATP else "ADP"
            if parameters["verbose"]:
                print(f"Couldn't find arp2 with {state}")
            recipe.change_topology_type(f"Actin-Polymer#Fail-Debranch-{state}")
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

    @staticmethod
    def reaction_function_debranching_ATP(topology):
        """
        reaction function to detach a branch filament from arp2/3 with ATP
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Debranching ATP")
        ActinUtil.do_debranching(topology, recipe, True)
        return recipe

    @staticmethod
    def reaction_function_debranching_ADP(topology):
        """
        reaction function to detach a branch filament from arp2/3 with ADP
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Debranching ADP")
        ActinUtil.do_debranching(topology, recipe, False)
        return recipe

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
            recipe.change_topology_type("Actin-Polymer#Fail-Cap-Unbind")
            return recipe
        v_actin = ReaddyUtil.get_neighbor_of_types(
            topology,
            v_cap,
            ActinUtil.get_all_polymer_actin_types("actin")
            + ActinUtil.get_all_polymer_actin_types("actin#ATP")
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
    def add_arp23_types(system, diffCoeff):
        """
        add particle and topology types for Arp2/3 dimer
        """
        system.topologies.add_type("Arp23-Dimer")
        system.add_topology_species("arp2", diffCoeff)
        system.add_topology_species("arp2#branched", diffCoeff)
        system.add_topology_species("arp2#free", diffCoeff)
        system.add_topology_species("arp3", diffCoeff)
        system.add_topology_species("arp3#ATP", diffCoeff)
        system.add_topology_species("arp3#new", diffCoeff)
        system.add_topology_species("arp3#new_ATP", diffCoeff)

    @staticmethod
    def add_cap_types(system, diffCoeff):
        """
        add particle and topology types for capping protein
        """
        system.topologies.add_type("Cap")
        system.add_topology_species("cap", diffCoeff)
        system.add_topology_species("cap#new", diffCoeff)
        system.add_topology_species("cap#bound", diffCoeff)

    @staticmethod
    def add_actin_types(system, diffCoeff):
        """
        add particle and topology types for actin

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
        system.topologies.add_type("Actin-Polymer#Fail-Pointed-Shrink-ATP")
        system.topologies.add_type("Actin-Polymer#Fail-Pointed-Shrink-ADP")
        system.topologies.add_type("Actin-Polymer#Fail-Barbed-Shrink-ATP")
        system.topologies.add_type("Actin-Polymer#Fail-Barbed-Shrink-ADP")
        system.topologies.add_type("Actin-Polymer#Fail-Hydrolysis-Actin")
        system.topologies.add_type("Actin-Polymer#Fail-Hydrolysis-Arp")
        system.topologies.add_type("Actin-Polymer#Fail-Branch-ATP")
        system.topologies.add_type("Actin-Polymer#Fail-Branch-ADP")
        system.topologies.add_type("Actin-Polymer#Fail-Arp-Bind-ATP")
        system.topologies.add_type("Actin-Polymer#Fail-Arp-Bind-ADP")
        system.topologies.add_type("Actin-Polymer#Fail-Debranch-ATP")
        system.topologies.add_type("Actin-Polymer#Fail-Debranch-ADP")
        system.topologies.add_type("Actin-Polymer#Fail-Arp-Unbind-ATP")
        system.topologies.add_type("Actin-Polymer#Fail-Arp-Unbind-ADP")
        system.topologies.add_type("Actin-Polymer#Fail-Nucleotide-Exchange-Actin")
        system.topologies.add_type("Actin-Polymer#Fail-Nucleotide-Exchange-Arp")
        system.topologies.add_type("Actin-Polymer#Fail-Cap-Unbind")
        system.add_topology_species("actin#free", diffCoeff)
        system.add_topology_species("actin#free_ATP", diffCoeff)
        system.add_topology_species("actin#new", diffCoeff)
        system.add_topology_species("actin#new_ATP", diffCoeff)
        for i in range(1, 4):
            system.add_topology_species(f"actin#{i}", diffCoeff)
            system.add_topology_species(f"actin#ATP_{i}", diffCoeff)
            system.add_topology_species(f"actin#mid_{i}", diffCoeff)
            system.add_topology_species(f"actin#mid_ATP_{i}", diffCoeff)
            system.add_topology_species(f"actin#pointed_{i}", diffCoeff)
            system.add_topology_species(f"actin#pointed_ATP_{i}", diffCoeff)
            system.add_topology_species(f"actin#barbed_{i}", diffCoeff)
            system.add_topology_species(f"actin#barbed_ATP_{i}", diffCoeff)
        system.add_topology_species("actin#branch_1", diffCoeff)
        system.add_topology_species("actin#branch_ATP_1", diffCoeff)
        system.add_topology_species("actin#branch_barbed_1", diffCoeff)
        system.add_topology_species("actin#branch_barbed_ATP_1", diffCoeff)

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
            ],
            0,
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
            ],
            1,
            force_constant,
            bond_length,
            system,
        )
        util.add_bond(
            ["actin#branch_1", "actin#branch_ATP_1"],
            [
                "actin#2",
                "actin#ATP_2",
                "actin#mid_2",
                "actin#mid_ATP_2",
                "actin#barbed_2",
                "actin#barbed_ATP_2",
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
            ],
            0,
            ["actin#new", "actin#new_ATP"],
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
            ["actin#new", "actin#new_ATP"],
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
            ],
            -1,
            ["actin#", "actin#ATP_", "actin#mid_", "actin#mid_ATP_"],
            0,
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
            ],
            1,
            force_constant,
            angle,
            system,
        )
        util.add_angle(
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2", "actin#mid_2", "actin#mid_ATP_2"],
            [
                "actin#3",
                "actin#ATP_3",
                "actin#mid_3",
                "actin#mid_ATP_3",
                "actin#barbed_3",
                "actin#barbed_ATP_3",
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
            ],
            -1,
            ["actin#", "actin#ATP_", "actin#mid_", "actin#mid_ATP_"],
            0,
            ["actin#", "actin#ATP_", "actin#mid_", "actin#mid_ATP_"],
            1,
            [
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
            ],
            2,
            force_constant,
            angle,
            system,
        )
        util.add_dihedral(
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2", "actin#mid_2", "actin#mid_ATP_2"],
            ["actin#3", "actin#ATP_3", "actin#mid_3", "actin#mid_ATP_3"],
            [
                "actin#1",
                "actin#ATP_1",
                "actin#mid_1",
                "actin#mid_ATP_1",
                "actin#barbed_1",
                "actin#barbed_ATP_1",
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
            ],
            0,
            ["arp2", "arp2#branched", "arp2#free"],
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
            ],
            0,
            ["arp3", "arp3#ATP", "arp3#new", "arp3#new_ATP"],
            None,
            force_constant,
            ActinStructure.arp3_to_mother_distance(),
            system,
        )
        util.add_bond(  # arp2 to arp3 bonds
            ["arp2", "arp2#branched", "arp2#free"],
            ["arp3", "arp3#ATP", "arp3#new", "arp3#new_ATP"],
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
            ["actin#2", "actin#ATP_2", "actin#barbed_2", "actin#barbed_ATP_2"],
            force_constant,
            ActinStructure.arp2_daughter1_daughter2_angle(),
            system,
        )
        util.add_polymer_angle_1D(
            ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"],
            0,
            ["actin#", "actin#ATP_"],
            1,
            ["arp3", "arp3#ATP"],
            None,
            force_constant,
            ActinStructure.mother1_mother2_arp3_angle(),
            system,
        )
        util.add_polymer_angle_1D(
            ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"],
            1,
            ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"],
            0,
            ["arp3", "arp3#ATP"],
            None,
            force_constant,
            ActinStructure.mother3_mother2_arp3_angle(),
            system,
        )
        angle = ActinStructure.mother0_mother1_arp2_angle()
        util.add_polymer_angle_1D(
            ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"],
            0,
            ["actin#", "actin#ATP_"],
            1,
            ["arp2", "arp2#branched"],
            None,
            force_constant,
            angle,
            system,
        )
        util.add_angle(
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2"],
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
            ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"],
            1,
            ["actin#", "actin#ATP_"],
            0,
            ["actin#", "actin#ATP_"],
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
            ],
            -1,
            ["actin#", "actin#ATP_"],
            0,
            ["actin#", "actin#ATP_"],
            1,
            ["arp2", "arp2#branched"],
            None,
            force_constant,
            angle,
            system,
        )
        util.add_dihedral(
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2"],
            ["actin#3", "actin#ATP_3"],
            ["arp2", "arp2#branched"],
            force_constant,
            angle,
            system,
        )
        util.add_polymer_dihedral_1D(
            ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"],
            1,
            ["actin#", "actin#ATP_"],
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
            ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"],
            0,
            ["actin#", "actin#ATP_"],
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
            ["actin#2", "actin#ATP_2"],
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
            ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"],
            0,
            ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"],
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
            ["actin#2", "actin#ATP_2", "actin#barbed_2", "actin#barbed_ATP_2"],
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
            ["actin#2", "actin#ATP_2", "actin#barbed_2", "actin#barbed_ATP_2"],
            force_constant,
            ActinStructure.arp3_arp2_daughter1_daughter2_dihedral_angle(),
            system,
        )
        util.add_dihedral(
            ["arp2#branched"],
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2"],
            ["actin#3", "actin#ATP_3", "actin#barbed_3", "actin#barbed_ATP_3"],
            force_constant,
            ActinStructure.arp2_daughter1_daughter2_daughter3_dihedral_angle(),
            system,
        )
        # mother to daughter
        angle = ActinStructure.mother0_mother1_arp2_daughter1_dihedral_angle()
        util.add_polymer_dihedral_1D(
            ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"],
            -1,
            ["actin#", "actin#ATP_"],
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
            ["actin#2", "actin#ATP_2"],
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
            ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"],
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
                "actin#branch_1",
                "actin#branch_ATP_1",
                "actin#2",
                "actin#ATP_2",
                "actin#pointed_2",
                "actin#pointed_ATP_2",
                "actin#3",
                "actin#ATP_3",
                "actin#pointed_3",
                "actin#pointed_ATP_3",
            ],
            ["arp2#branched"],
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2", "actin#barbed_2", "actin#barbed_ATP_2"],
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
            ["actin#", "actin#ATP_"],
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
            ],
            0,
            ["actin#", "actin#ATP_", "actin#mid_", "actin#mid_ATP_"],
            1,
            ["cap#bound"],
            None,
            force_constant,
            angle,
            system,
        )
        util.add_angle(
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2"],
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
            ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"],
            -1,
            ["actin#", "actin#ATP_"],
            0,
            ["actin#", "actin#ATP_"],
            1,
            ["cap#bound"],
            None,
            force_constant,
            angle,
            system,
        )
        util.add_dihedral(
            ["actin#branch_1", "actin#branch_ATP_1"],
            ["actin#2", "actin#ATP_2"],
            ["actin#3", "actin#ATP_3"],
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
    def add_repulsions(force_constant, system, util):
        """
        add repulsions
        """
        util.add_repulsion(
            [
                "actin#pointed_1",
                "actin#pointed_ATP_1",
                "actin#pointed_2",
                "actin#pointed_ATP_2",
                "actin#pointed_3",
                "actin#pointed_ATP_3",
                "actin#1",
                "actin#ATP_1",
                "actin#2",
                "actin#ATP_2",
                "actin#3",
                "actin#ATP_3",
                "actin#branch_1",
                "actin#branch_ATP_1",
                "actin#branch_barbed_1",
                "actin#branch_barbed_ATP_1",
                "actin#barbed_1",
                "actin#barbed_ATP_1",
                "actin#barbed_2",
                "actin#barbed_ATP_2",
                "actin#barbed_3",
                "actin#barbed_ATP_3",
                "arp2",
                "arp2#branched",
                "arp2#free",
                "arp3",
                "arp3#ATP",
                "cap",
                "cap#bound",
                "actin#free",
                "actin#free_ATP",
            ],
            [
                "actin#pointed_1",
                "actin#pointed_ATP_1",
                "actin#pointed_2",
                "actin#pointed_ATP_2",
                "actin#pointed_3",
                "actin#pointed_ATP_3",
                "actin#1",
                "actin#ATP_1",
                "actin#2",
                "actin#ATP_2",
                "actin#3",
                "actin#ATP_3",
                "actin#branch_1",
                "actin#branch_ATP_1",
                "actin#branch_barbed_1",
                "actin#branch_barbed_ATP_1",
                "actin#barbed_1",
                "actin#barbed_ATP_1",
                "actin#barbed_2",
                "actin#barbed_ATP_2",
                "actin#barbed_3",
                "actin#barbed_ATP_3",
                "arp2",
                "arp2#branched",
                "arp2#free",
                "arp3",
                "arp3#ATP",
                "cap",
                "cap#bound",
                "actin#free",
                "actin#free_ATP",
            ],
            force_constant,
            ActinStructure.actin_to_actin_repulsion_distance(),
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
    def add_spatial_dimerize_reaction(system):
        """
        attach two monomers
        """
        system.topologies.add_spatial_reaction(
            "Dimerize: "
            "Actin-Monomer(actin#free_ATP) + Actin-Monomer(actin#free_ATP) -> "
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
    def add_spatial_trimerize_reaction(system):
        """
        attach a monomer to a dimer
        """
        for i in range(1, 4):
            system.topologies.add_spatial_reaction(
                f"Trimerize{i}: "
                f"Actin-Dimer(actin#barbed_ATP_{i}) + Actin-Monomer(actin#free_ATP) -> "
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
    def add_nonspatial_trimerize_reaction(system):
        """
        attach an implicit monomer to a dimer
        """
        system.topologies.add_structural_reaction(
            "Nonspatial_Trimerize",
            topology_type="Actin-Dimer",
            reaction_function=ActinUtil.reaction_function_nonspatial_trimerize,
            rate_function=lambda x: parameters["trimerize_nonspatial_rate"],
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
    def add_spatial_nucleate_reaction(system):
        """
        reversibly attach a monomer to a trimer
        """
        for i in range(1, 4):
            system.topologies.add_spatial_reaction(
                f"Barbed_Growth_Nucleate_ATP{i}: "
                f"Actin-Trimer(actin#barbed_ATP_{i}) + Actin-Monomer(actin#free_ATP) "
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
    def add_nonspatial_nucleate_reaction(system):
        """
        attach an implicit monomer to a trimer
        """
        system.topologies.add_structural_reaction(
            "Nonspatial_Nucleate_ATP",
            topology_type="Actin-Trimer",
            reaction_function=ActinUtil.reaction_function_nonspatial_barbed_grow_ATP,
            rate_function=lambda x: parameters["nucleate_nonspatial_ATP_rate"],
        )
        system.topologies.add_structural_reaction(
            "Nonspatial_Nucleate_ADP",
            topology_type="Actin-Trimer",
            reaction_function=ActinUtil.reaction_function_nonspatial_barbed_grow_ADP,
            rate_function=lambda x: parameters["nucleate_nonspatial_ADP_rate"],
        )

    @staticmethod
    def add_spatial_pointed_growth_reaction(system):
        """
        attach a monomer to the pointed end of a filament
        """
        for i in range(1, 4):
            system.topologies.add_spatial_reaction(
                f"Pointed_Growth_ATP1{i}: Actin-Polymer(actin#pointed_{i}) + "
                "Actin-Monomer(actin#free_ATP) -> "
                f"Actin-Polymer#GrowingPointed(actin#{i}--actin#new_ATP)",
                rate=parameters["pointed_growth_ATP_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Pointed_Growth_ATP2{i}: Actin-Polymer(actin#pointed_ATP_{i}) + "
                "Actin-Monomer(actin#free_ATP) -> "
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
    def add_nonspatial_pointed_growth_reaction(system):
        """
        attach an implicit monomer to a pointed end
        """
        system.topologies.add_structural_reaction(
            "Nonspatial_Pointed_Growth_ATP",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_nonspatial_pointed_grow_ATP,
            rate_function=lambda x: parameters["pointed_growth_nonspatial_ATP_rate"],
        )
        system.topologies.add_structural_reaction(
            "Nonspatial_Pointed_Growth_ADP",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_nonspatial_pointed_grow_ADP,
            rate_function=lambda x: parameters["pointed_growth_nonspatial_ADP_rate"],
        )

    @staticmethod
    def add_pointed_shrink_reaction(system):
        """
        remove a monomer from the pointed end of a filament
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
            "Fail_Pointed_Shrink_ATP",
            topology_type="Actin-Polymer#Fail-Pointed-Shrink-ATP",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Fail_Pointed_Shrink_ADP",
            topology_type="Actin-Polymer#Fail-Pointed-Shrink-ADP",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Cleanup_Shrink",
            topology_type="Actin-Polymer#Shrinking",
            reaction_function=ActinUtil.reaction_function_cleanup_shrink,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_spatial_barbed_growth_reaction(system):
        """
        attach a monomer to the barbed end of a filament
        """
        for i in range(1, 4):
            system.topologies.add_spatial_reaction(
                f"Barbed_Growth_ATP1{i}: Actin-Polymer(actin#barbed_{i}) + "
                "Actin-Monomer(actin#free_ATP) -> "
                f"Actin-Polymer#GrowingBarbed(actin#{i}--actin#new_ATP)",
                rate=parameters["barbed_growth_ATP_rate"],
                radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Barbed_Growth_ATP2{i}: Actin-Polymer(actin#barbed_ATP_{i}) + "
                "Actin-Monomer(actin#free_ATP) -> "
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
            "Actin-Monomer(actin#free_ATP) -> "
            "Actin-Polymer#GrowingBarbed(actin#branch_1--actin#new_ATP)",
            rate=parameters["barbed_growth_ATP_rate"],
            radius=2 * parameters["actin_radius"] + parameters["reaction_distance"],
        )
        system.topologies.add_spatial_reaction(
            "Branch_Barbed_Growth_ATP2: Actin-Polymer(actin#branch_barbed_ATP_1) + "
            "Actin-Monomer(actin#free_ATP) -> "
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
    def add_nonspatial_barbed_growth_reaction(system):
        """
        attach an implicit monomer to a barbed end
        """
        system.topologies.add_structural_reaction(
            "Nonspatial_Barbed_Growth_ATP",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_nonspatial_barbed_grow_ATP,
            rate_function=lambda x: parameters["barbed_growth_nonspatial_ATP_rate"],
        )
        system.topologies.add_structural_reaction(
            "Nonspatial_Barbed_Growth_ADP",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_nonspatial_barbed_grow_ADP,
            rate_function=lambda x: parameters["barbed_growth_nonspatial_ADP_rate"],
        )

    @staticmethod
    def add_barbed_shrink_reaction(system):
        """
        remove a monomer from the barbed end of a filament
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
        system.topologies.add_structural_reaction(
            "Fail_Barbed_Shrink_ATP",
            topology_type="Actin-Polymer#Fail-Barbed-Shrink-ATP",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Fail_Barbed_Shrink_ADP",
            topology_type="Actin-Polymer#Fail-Barbed-Shrink-ADP",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
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
            "Fail_Hydrolysis_Actin",
            topology_type="Actin-Polymer#Fail-Hydrolysis-Actin",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Hydrolysis_Arp",
            topology_type="Actin-Polymer",
            reaction_function=ActinUtil.reaction_function_hydrolyze_arp,
            rate_function=lambda x: parameters["hydrolysis_arp_rate"],
        )
        system.topologies.add_structural_reaction(
            "Fail_Hydrolysis_Arp",
            topology_type="Actin-Polymer#Fail-Hydrolysis-Arp",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
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
                f"Actin-Polymer(actin#ATP_{i}) + Arp23-Dimer(arp3) -> "
                f"Actin-Polymer#Branching(actin#ATP_{i}--arp3#new)",
                rate=parameters["arp_bind_ATP_rate"],
                radius=parameters["actin_radius"]
                + parameters["arp23_radius"]
                + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Arp_Bind_ATP2{i}: "
                f"Actin-Polymer(actin#ATP_{i}) + Arp23-Dimer(arp3#ATP) -> "
                f"Actin-Polymer#Branching(actin#ATP_{i}--arp3#new_ATP)",
                rate=parameters["arp_bind_ATP_rate"],
                radius=parameters["actin_radius"]
                + parameters["arp23_radius"]
                + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Arp_Bind_ADP1{i}: "
                f"Actin-Polymer(actin#{i}) + Arp23-Dimer(arp3) -> "
                f"Actin-Polymer#Branching(actin#{i}--arp3#new)",
                rate=parameters["arp_bind_ADP_rate"],
                radius=parameters["actin_radius"]
                + parameters["arp23_radius"]
                + parameters["reaction_distance"],
            )
            system.topologies.add_spatial_reaction(
                f"Arp_Bind_ADP2{i}: "
                f"Actin-Polymer(actin#{i}) + Arp23-Dimer(arp3#ATP) -> "
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
        system.topologies.add_structural_reaction(
            "Cleanup_Fail_Arp_Bind_ATP",
            topology_type="Actin-Polymer#Fail-Branch-ATP",
            reaction_function=ActinUtil.reaction_function_cleanup_shrink,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Cleanup_Fail_Arp_Bind_ADP",
            topology_type="Actin-Polymer#Fail-Branch-ADP",
            reaction_function=ActinUtil.reaction_function_cleanup_shrink,
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
        system.topologies.add_structural_reaction(
            "Fail_Arp_Unbind_ATP",
            topology_type="Actin-Polymer#Fail-Arp-Unbind-ATP",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Fail_Arp_Unbind_ADP",
            topology_type="Actin-Polymer#Fail-Arp-Unbind-ADP",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_spatial_nucleate_branch_reaction(system):
        """
        add actin to arp2/3 to begin a branch
        """
        system.topologies.add_spatial_reaction(
            "Barbed_Growth_Branch_ATP: "
            "Actin-Polymer(arp2) + Actin-Monomer(actin#free_ATP) -> "
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
    def add_nonspatial_nucleate_branch_reaction(system):
        """
        attach implicit actin to arp2/3 to begin a branch
        """
        reaction_function = ActinUtil.reaction_function_nonspatial_nucleate_branch_ATP
        system.topologies.add_structural_reaction(
            "Nonspatial_Barbed_Growth_Branch_ATP",
            topology_type="Actin-Polymer",
            reaction_function=reaction_function,
            rate_function=lambda x: parameters["nucleate_branch_nonspatial_ATP_rate"],
        )
        reaction_function = ActinUtil.reaction_function_nonspatial_nucleate_branch_ADP
        system.topologies.add_structural_reaction(
            "Nonspatial_Barbed_Growth_Branch_ADP",
            topology_type="Actin-Polymer",
            reaction_function=reaction_function,
            rate_function=lambda x: parameters["nucleate_branch_nonspatial_ADP_rate"],
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
        system.topologies.add_structural_reaction(
            "Fail_Debranch_ATP",
            topology_type="Actin-Polymer#Fail-Debranch-ATP",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Fail_Debranch_ADP",
            topology_type="Actin-Polymer#Fail-Debranch-ADP",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
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
            "Finish_Cap-Bind",
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
        system.topologies.add_structural_reaction(
            "Fail_Cap_Unbind",
            topology_type="Actin-Polymer#Fail-Cap-Unbind",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
