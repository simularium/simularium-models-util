#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np
import scipy.linalg as linalg
import random
import readdy
import os
from shutil import rmtree
import math
import pandas as pd
from tqdm import tqdm


class ReaddyUtil:
    def __init__(self):
        """
        Utilities used for Simularium ReaDDy models
        """
        self.bond_pairs = []
        self.angle_triples = []
        self.dihedral_quads = []
        self.repulse_pairs = []

    @staticmethod
    def normalize(v):
        """
        normalize a vector
        """
        return v / np.linalg.norm(v)

    @staticmethod
    def analyze_reaction_count_over_time(reactions, reaction_name):
        """
        Get a list of the number of times a reaction happened
        between each analyzed timestep of the given reaction
        """
        if reaction_name not in reactions:
            print(f"Couldn't find reaction: {reaction_name}")
            return None
        return np.insert(reactions[reaction_name].to_numpy(), 0, 0.0)

    @staticmethod
    def get_perpendicular_components_of_vector(v, v_perpendicular):
        """
        Get the components of v that are perpendicular to v_perpendicular
        """
        return (
            v
            - (np.dot(v, v_perpendicular) / pow(np.linalg.norm(v_perpendicular), 2))
            * v_perpendicular
        )

    @staticmethod
    def get_angle_between_vectors(v1, v2, in_degrees=False):
        """
        get the angle between two vectors
        in radians unless in_degrees is True
        """
        result = np.arccos(
            np.clip(
                np.dot(ReaddyUtil.normalize(v1), ReaddyUtil.normalize(v2)), -1.0, 1.0
            )
        )
        return result if not in_degrees else np.rad2deg(result)

    @staticmethod
    def rotate(v, axis, angle):
        """
        rotate a vector around axis by angle (radians)
        """
        rotation = linalg.expm(np.cross(np.eye(3), ReaddyUtil.normalize(axis) * angle))
        return np.dot(rotation, np.copy(v))

    @staticmethod
    def get_rotation_matrix(v1, v2):
        """
        Cross the vectors and get a rotation matrix
        """
        v3 = np.cross(v2, v1)
        return np.array(
            [[v1[0], v2[0], v3[0]], [v1[1], v2[1], v3[1]], [v1[2], v2[2], v3[2]]]
        )

    @staticmethod
    def get_orientation_from_positions(positions):
        """
        orthonormalize and cross the vectors from position 2
        to the other positions to get a basis local to position 2,
        positions = [position 1, position 2, position 3]
        """
        v1 = ReaddyUtil.normalize(positions[0] - positions[1])
        v2 = ReaddyUtil.normalize(positions[2] - positions[1])
        v2 = ReaddyUtil.normalize(v2 - (np.dot(v1, v2) / np.dot(v1, v1)) * v1)
        return ReaddyUtil.get_rotation_matrix(v1, v2)

    @staticmethod
    def get_orientation_from_vectors(v1, v2):
        """
        orthonormalize and cross the vectors to get a rotation matrix
        """
        v2 = ReaddyUtil.normalize(v2 - (np.dot(v1, v2) / np.dot(v1, v1)) * v1)
        return ReaddyUtil.get_rotation_matrix(v1, v2)

    @staticmethod
    def get_random_perpendicular_vector(v):
        """
        get a random unit vector perpendicular to v
        """
        if v[0] == 0 and v[1] == 0:
            if v[2] == 0:
                raise ValueError("zero vector")
            return np.array([0, 1, 0])
        u = ReaddyUtil.normalize(np.array([-v[1], v[0], 0]))
        return ReaddyUtil.rotate(u, v, 2 * np.pi * random.random())

    @staticmethod
    def topology_to_string(topology):
        """
        get string with vertex types and ids in a topology
        """
        result = f"{topology.type} : \n"
        for vertex in topology.graph.get_vertices():
            result += f"{ReaddyUtil.vertex_to_string(topology, vertex)}\n"
            for neighbor in vertex:
                result += (
                    f" -- {ReaddyUtil.vertex_to_string(topology, neighbor.get())}\n"
                )
        return result

    @staticmethod
    def vertex_to_string(topology, vertex):
        """
        get string with type and id for vertex
        """
        return (
            topology.particle_type_of_vertex(vertex)
            + " ("
            + str(topology.particle_id_of_vertex(vertex))
            + ")"
        )

    @staticmethod
    def get_non_periodic_boundary_position(pos1, pos2, box_size):
        """
        if the distance between two positions is greater than box_size,
        move the second position across the box (for positioning calculations)
        """
        result = np.copy(pos2)
        for dim in range(3):
            if abs(pos2[dim] - pos1[dim]) > box_size[dim] / 2.0:
                result[dim] -= pos2[dim] / abs(pos2[dim]) * box_size[dim]
        return result

    @staticmethod
    def calculate_diffusionCoefficient(r0, eta, T):
        """
        calculates the theoretical diffusion constant of a spherical particle
            with radius r0[nm]
            in a media with viscosity eta [cP]
            at temperature T [Kelvin]

            returns nm^2/s
        """
        return (
            (1.38065 * 10 ** (-23) * T)
            / (6 * np.pi * eta * 10 ** (-3) * r0 * 10 ** (-9))
            * 10**18
            / 10**9
        )

    @staticmethod
    def calculate_nParticles(C, dim):
        """
        calculates the number of particles for a species
            at concentration C [uM]
            in a cuboidal container with dimensions dim = dimx, dimy, dimz [nm]

            returns unitless number
        """
        return int(round(C * 1e-30 * 6.022e23 * np.prod(dim)))

    @staticmethod
    def calculate_concentration(n, dim):
        """
        calculates the concentration for a species
            with number of particles n
            in a cuboidal container with dimensions dim = dimx, dimy, dimz [nm]

            returns concentration [uM]
        """
        return n / (1e-30 * 6.022e23 * np.prod(dim))

    @staticmethod
    def vertex_not_found(topology, verbose, error_msg, debug_msg):
        """
        If a vertex was not found, check if an exception should be raised
        or a debug message should be printed
        If error_msg == "", don't raise an exception
        If not verbose or debug_msg == "", don't print the message
        """
        if error_msg:
            raise Exception(error_msg + "\n" + ReaddyUtil.topology_to_string(topology))
        if verbose and debug_msg:
            print(debug_msg)

    @staticmethod
    def get_vertex_of_type(
        topology, vertex_type, exact_match, verbose=False, debug_msg="", error_msg=""
    ):
        """
        get the first vertex with a given type
        """
        for vertex in topology.graph.get_vertices():
            pt = topology.particle_type_of_vertex(vertex)
            if (not exact_match and vertex_type in pt) or (
                exact_match and pt == vertex_type
            ):
                return vertex
        ReaddyUtil.vertex_not_found(topology, verbose, error_msg, debug_msg)
        return None

    @staticmethod
    def get_first_vertex_of_types(
        topology, vertex_types, verbose=False, debug_msg="", error_msg=""
    ):
        """
        get the first vertex with any of the given types
        """
        for vertex in topology.graph.get_vertices():
            if topology.particle_type_of_vertex(vertex) in vertex_types:
                return vertex
        ReaddyUtil.vertex_not_found(topology, verbose, error_msg, debug_msg)
        return None

    @staticmethod
    def get_vertex_with_id(
        topology, vertex_id, verbose=False, debug_msg="", error_msg=""
    ):
        """
        get the first vertex with a given id
        """
        for vertex in topology.graph.get_vertices():
            if topology.particle_id_of_vertex(vertex) == vertex_id:
                return vertex
        ReaddyUtil.vertex_not_found(topology, verbose, error_msg, debug_msg)
        return None

    @staticmethod
    def get_first_neighbor(
        topology, vertex, exclude_vertices, verbose=False, debug_msg="", error_msg=""
    ):
        """
        get the first neighboring vertex
        """
        exclude_ids = []
        for v in exclude_vertices:
            exclude_ids.append(topology.particle_id_of_vertex(v))
        for neighbor in vertex:
            if topology.particle_id_of_vertex(neighbor) in exclude_ids:
                continue
            return neighbor.get()
        ReaddyUtil.vertex_not_found(topology, verbose, error_msg, debug_msg)
        return None

    @staticmethod
    def get_neighbor_of_type(
        topology,
        vertex,
        vertex_type,
        exact_match,
        exclude_vertices=[],
        verbose=False,
        debug_msg="",
        error_msg="",
    ):
        """
        get the first neighboring vertex of type vertex_type
        """
        exclude_ids = []
        for v in exclude_vertices:
            exclude_ids.append(topology.particle_id_of_vertex(v))
        for neighbor in vertex:
            if topology.particle_id_of_vertex(neighbor) in exclude_ids:
                continue
            v_neighbor = neighbor.get()
            pt = topology.particle_type_of_vertex(v_neighbor)
            if (not exact_match and vertex_type in pt) or (
                exact_match and pt == vertex_type
            ):
                return v_neighbor
        ReaddyUtil.vertex_not_found(topology, verbose, error_msg, debug_msg)
        return None

    @staticmethod
    def get_neighbor_of_types(
        topology,
        vertex,
        vertex_types,
        exclude_vertices,
        verbose=False,
        debug_msg="",
        error_msg="",
    ):
        """
        get the first neighboring vertex with any of the given types,
        excluding particles with the given ids
        """
        exclude_ids = []
        for v in exclude_vertices:
            exclude_ids.append(topology.particle_id_of_vertex(v))
        for neighbor in vertex:
            if topology.particle_id_of_vertex(neighbor) in exclude_ids:
                continue
            v_neighbor = neighbor.get()
            pt = topology.particle_type_of_vertex(v_neighbor)
            if pt in vertex_types:
                return v_neighbor
        ReaddyUtil.vertex_not_found(topology, verbose, error_msg, debug_msg)
        return None

    @staticmethod
    def get_vertices_of_type(
        topology, vertex_type, exact_match, verbose=False, debug_msg="", error_msg=""
    ):
        """
        get all vertices with a given type
        """
        v = []
        for vertex in topology.graph.get_vertices():
            pt = topology.particle_type_of_vertex(vertex)
            if (not exact_match and vertex_type in pt) or (
                exact_match and pt == vertex_type
            ):
                v.append(vertex)
        if len(v) == 0:
            ReaddyUtil.vertex_not_found(topology, verbose, error_msg, debug_msg)
        return v

    @staticmethod
    def get_neighbors_of_type(
        topology,
        vertex,
        vertex_type,
        exact_match,
        verbose=False,
        debug_msg="",
        error_msg="",
    ):
        """
        get all neighboring vertices with a given type
        """
        v = []
        for neighbor in vertex:
            v_neighbor = neighbor.get()
            pt = topology.particle_type_of_vertex(v_neighbor)
            if (not exact_match and vertex_type in pt) or (
                exact_match and pt == vertex_type
            ):
                v.append(v_neighbor)
        if len(v) == 0:
            ReaddyUtil.vertex_not_found(topology, verbose, error_msg, debug_msg)
        return v

    @staticmethod
    def get_random_vertex_of_type(
        topology, vertex_type, exact_match, verbose=False, debug_msg="", error_msg=""
    ):
        """
        get a random vertex with a given type
        """
        vertices = ReaddyUtil.get_vertices_of_type(topology, vertex_type, exact_match)
        if len(vertices) == 0:
            ReaddyUtil.vertex_not_found(topology, verbose, error_msg, debug_msg)
            return None
        return random.choice(vertices)

    @staticmethod
    def get_random_vertex_of_types(
        topology, vertex_types, verbose=False, debug_msg="", error_msg=""
    ):
        """
        get a random vertex with any of the given types
        """
        v = []
        for vertex_type in vertex_types:
            v += ReaddyUtil.get_vertices_of_type(topology, vertex_type, True)
        if len(v) == 0:
            ReaddyUtil.vertex_not_found(topology, verbose, error_msg, debug_msg)
            return None
        return random.choice(v)

    @staticmethod
    def vertex_satisfies_type(vertex_type, types_include, types_exclude):
        """
        check if vertex satisfies the type requirements
        """
        for t in types_include:
            if t not in vertex_type:
                return False
        for t in types_exclude:
            if t in vertex_type:
                return False
        return True

    @staticmethod
    def particle_type_with_flags(
        particle_type, add_flags, remove_flags, reverse_sort=False
    ):
        """
        get particle type with the flags added and removed
        """
        if "#" not in particle_type:
            for f in range(len(add_flags)):
                particle_type = particle_type + ("_" if f > 0 else "#") + add_flags[f]
            return particle_type
        flag_string = particle_type[particle_type.index("#") + 1 :]
        flags = flag_string.split("_")
        polymer_indices = ""
        if "tubulin" in particle_type and len(flags) > 1:
            polymer_indices = f"_{flags[-2]}_{flags[-1]}"
            flags = flags[:-2]
        for flag in remove_flags:
            if flag in flags:
                flags.remove(flag)
        for flag in add_flags:
            if flag not in flags:
                flags.append(flag)
        if "" in flags:
            flags.remove("")
        if len(flags) < 1:
            return particle_type[: particle_type.index("#")]
        flags.sort(reverse=reverse_sort)
        flag_string = ""
        for f in range(len(flags)):
            flag_string = flag_string + ("_" if f > 0 else "") + flags[f]
        particle_type = particle_type[: particle_type.index("#")]
        new_type = f"{particle_type}#{flag_string}{polymer_indices}"
        return new_type

    @staticmethod
    def set_flags(
        topology, recipe, vertex, add_flags, remove_flags, reverse_sort=False
    ):
        """
        set flags on a vertex
        """
        particle_type = topology.particle_type_of_vertex(vertex)
        particle_type = ReaddyUtil.particle_type_with_flags(
            particle_type, add_flags, remove_flags, reverse_sort
        )
        recipe.change_particle_type(vertex, particle_type)

    @staticmethod
    def calculate_polymer_number(number, offset, polymer_number_types):
        """
        calculates the polymer number
            from number
            by offset in [-2, 2]

            returns number in [1,5]
        """
        n = number + offset
        if n > polymer_number_types:
            n -= polymer_number_types
        if n < 1:
            n += polymer_number_types
        return int(n)

    @staticmethod
    def get_vertex_position(topology, vertex):
        """
        get the position of a vertex
        """
        pos = topology.position_of_vertex(vertex)
        return np.array([pos[0], pos[1], pos[2]])

    @staticmethod
    def vertices_are_equal(topology, vertex1, vertex2):
        """
        check if references are the same vertex
        """
        return topology.particle_id_of_vertex(
            vertex1
        ) == topology.particle_id_of_vertex(vertex2)

    @staticmethod
    def vertices_are_connected(topology, vertex1, vertex2):
        """
        check if the vertices share an edge
        """
        for neighbor in vertex1:
            if ReaddyUtil.vertices_are_equal(topology, vertex2, neighbor.get()):
                return True
        return False

    @staticmethod
    def reaction_function_reset_state(topology):
        """
        reaction function for removing flags from a topology
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        tt = topology.type
        recipe.change_topology_type(tt[: tt.index("#")])
        return recipe

    @staticmethod
    def rate_function_infinity(topology):
        """
        rate function for a reaction that should trigger immediately
        whenever reactants are available
        """
        return 1e30

    @staticmethod
    def clamp_polymer_offsets_2D(polymer_index_x, polymer_offsets):
        """
        clamp offsets so y polymer offset is incremented
        if new x polymer index is not in [1,3]
        """
        if len(polymer_offsets) < 2:
            return polymer_offsets
        offsets = copy.deepcopy(polymer_offsets)
        if offsets[0] != 0:
            if polymer_index_x + offsets[0] < 1:
                offsets[1] -= 1
            elif polymer_index_x + offsets[0] > 3:
                offsets[1] += 1
        return offsets

    @staticmethod
    def get_types_with_polymer_numbers_1D(
        particle_types, x, polymer_offset, polymer_number_types
    ):
        """
        creates a list of types with 1D polymer numbers
            for each type in particle types
            at polymer number x
            with polymer_offset dx in [-2, 2]

            returns list of types
        """
        types = []
        for t in particle_types:
            types.append(
                (
                    t
                    + str(
                        ReaddyUtil.calculate_polymer_number(
                            x, polymer_offset, polymer_number_types
                        )
                    )
                )
                if polymer_offset is not None
                else t
            )
        return types

    @staticmethod
    def get_types_with_polymer_numbers_2D(particle_types, x, y, polymer_offsets):
        """
        creates a list of types with 2D polymer numbers
            for each type in particle types
            at polymer number x_y
            with polymer_offsets [dx, dy] both in [-2, 2]

            returns list of types
        """
        types = []
        for t in particle_types:
            types.append(
                (
                    t
                    + str(ReaddyUtil.calculate_polymer_number(x, polymer_offsets[0]))
                    + "_"
                    + str(ReaddyUtil.calculate_polymer_number(y, polymer_offsets[1]))
                )
                if len(polymer_offsets) > 0
                else t
            )
        return types

    @staticmethod
    def get_random_unit_vector():
        """
        get a random unit vector
        """
        return ReaddyUtil.normalize(
            np.array([random.random(), random.random(), random.random()])
        )

    @staticmethod
    def get_random_boundary_position(box_size):
        """
        get a random position on one of the boundary faces
        """
        pos = box_size * np.random.uniform(size=(3)) - box_size / 2
        face = random.randint(0, 5)
        if face == 0:
            pos[0] = -box_size[0] / 2
        elif face == 1:
            pos[0] = box_size[0] / 2
        elif face == 2:
            pos[1] = -box_size[1] / 2
        elif face == 3:
            pos[1] = box_size[1] / 2
        elif face == 4:
            pos[2] = -box_size[2] / 2
        else:
            pos[2] = box_size[2] / 2
        return pos

    @staticmethod
    def try_remove_edge(topology, recipe, vertex1, vertex2):
        """
        try to remove an edge
        """
        if not ReaddyUtil.vertices_are_connected(topology, vertex1, vertex2):
            return False, (
                "Tried to remove non-existent edge! "
                + ReaddyUtil.vertex_to_string(topology, vertex1)
                + " -- "
                + ReaddyUtil.vertex_to_string(topology, vertex2)
            )
        recipe.remove_edge(vertex1, vertex2)
        return True, ""

    def add_bond(self, types1, types2, force_const, bond_length, system):
        """
        adds a bond to the system (if it hasn't been added already)
            from each type in types1
            to each type in types2
            with force constant force_const
            and length bond_length [nm]
        """
        for t1 in types1:
            for t2 in types2:
                if (t1, t2) not in self.bond_pairs and (t2, t1) not in self.bond_pairs:
                    system.topologies.configure_harmonic_bond(
                        t1, t2, force_const, bond_length
                    )
                    self.bond_pairs.append((t1, t2))
                    self.bond_pairs.append((t2, t1))

    def add_angle(self, types1, types2, types3, force_const, angle, system):
        """
        adds an angle to the system (if it hasn't been added already)
            from each type in types1
            through each type in types2
            to each type in types3
            with force constant force_const
            and angle [radians]
        """
        for t1 in types1:
            for t2 in types2:
                for t3 in types3:
                    if (t1, t2, t3) not in self.angle_triples and (
                        t3,
                        t2,
                        t1,
                    ) not in self.angle_triples:
                        system.topologies.configure_harmonic_angle(
                            t1, t2, t3, force_const, angle
                        )
                        self.angle_triples.append((t1, t2, t3))
                        self.angle_triples.append((t3, t2, t1))
                        # print(f"angles: {t1}, {t2}, {t3}")

    def add_dihedral(self, types1, types2, types3, types4, force_const, angle, system):
        """
        adds a cosine dihedral to the system (if it hasn't been added already)
            from each type in types1
            through each type in types2
            through each type in types3
            to each type in types4
            with force constant force_const
            and angle [radians]
        """
        for t1 in types1:
            for t2 in types2:
                for t3 in types3:
                    for t4 in types4:
                        if (t1, t2, t3, t4) not in self.dihedral_quads and (
                            t4,
                            t3,
                            t2,
                            t1,
                        ) not in self.dihedral_quads:
                            system.topologies.configure_cosine_dihedral(
                                t1, t2, t3, t4, force_const, 1.0, angle
                            )
                            system.topologies.configure_cosine_dihedral(
                                t4, t3, t2, t1, force_const, 1.0, angle
                            )
                            self.dihedral_quads.append((t1, t2, t3, t4))
                            self.dihedral_quads.append((t4, t3, t2, t1))
                            # print(f"dihedrals: {t1}, {t2}, {t3}, {t4}")
                            # print("from add_dihedrals:", self.dihedral_quads)

    def add_repulsion(self, types1, types2, force_const, distance, system):
        """
        adds a pairwise repulsion to the system (if it hasn't been added already)
            between each type in types1
            and each type in types2
            with force constant force_const
            with equilibrium distance [nm]
        """
        for t1 in types1:
            for t2 in types2:
                if (t1, t2) not in self.repulse_pairs and (
                    t2,
                    t1,
                ) not in self.repulse_pairs:
                    system.potentials.add_harmonic_repulsion(
                        t1, t2, force_const, distance
                    )
                    self.repulse_pairs.append((t1, t2))
                    self.repulse_pairs.append((t2, t1))

    def add_polymer_bond_1D(
        self,
        particle_types1,
        polymer_offset1,
        particle_types2,
        polymer_offset2,
        force_const,
        bond_length,
        system,
        polymer_number_types,
    ):
        """
        adds a bond between all polymer numbers
            from types particle_types1
            with offset polymer_offset1
            to types particle_types2
            with offset polymer_offset2
            with force constant force_const
            and length bond_length [nm]
        """
        polymer_number_types = int(polymer_number_types)
        for x in range(1, polymer_number_types + 1):
            self.add_bond(
                (
                    ReaddyUtil.get_types_with_polymer_numbers_1D(
                        particle_types1, x, polymer_offset1, polymer_number_types
                    )
                    if polymer_offset1 is not None
                    else particle_types1
                ),
                (
                    ReaddyUtil.get_types_with_polymer_numbers_1D(
                        particle_types2, x, polymer_offset2, polymer_number_types
                    )
                    if polymer_offset2 is not None
                    else particle_types2
                ),
                force_const,
                bond_length,
                system,
            )

    def add_polymer_bond_2D(
        self,
        particle_types1,
        polymer_offsets1,
        particle_types2,
        polymer_offsets2,
        force_const,
        bond_length,
        system,
    ):
        """
        adds a bond between all polymer numbers
            from types particle_types1
            with offsets polymer_offsets1
            to types particle_types2
            with offsets polymer_offsets2
            with force constant force_const
            and length bond_length [nm]
        """
        for x in range(1, 4):
            for y in range(1, 4):
                offsets1 = ReaddyUtil.clamp_polymer_offsets_2D(x, polymer_offsets1)
                offsets2 = ReaddyUtil.clamp_polymer_offsets_2D(x, polymer_offsets2)
                self.add_bond(
                    ReaddyUtil.get_types_with_polymer_numbers_2D(
                        particle_types1, x, y, offsets1
                    ),
                    ReaddyUtil.get_types_with_polymer_numbers_2D(
                        particle_types2, x, y, offsets2
                    ),
                    force_const,
                    bond_length,
                    system,
                )

    def add_polymer_angle_1D(
        self,
        particle_types1,
        polymer_offset1,
        particle_types2,
        polymer_offset2,
        particle_types3,
        polymer_offset3,
        force_const,
        angle,
        system,
        polymer_number_types,
    ):
        """
        adds an angle between all polymer numbers
            with offset polymer_offset
            of types particle_types
            with force constant force_const
            and angle [radians]
        """

        for x in range(1, polymer_number_types + 1):
            self.add_angle(
                ReaddyUtil.get_types_with_polymer_numbers_1D(
                    particle_types1, x, polymer_offset1, polymer_number_types
                ),
                ReaddyUtil.get_types_with_polymer_numbers_1D(
                    particle_types2, x, polymer_offset2, polymer_number_types
                ),
                ReaddyUtil.get_types_with_polymer_numbers_1D(
                    particle_types3, x, polymer_offset3, polymer_number_types
                ),
                force_const,
                angle,
                system,
            )

    def add_polymer_angle_2D(
        self,
        particle_types1,
        polymer_offsets1,
        particle_types2,
        polymer_offsets2,
        particle_types3,
        polymer_offsets3,
        force_const,
        angle,
        system,
    ):
        """
        adds an angle between all polymer numbers
            with offsets polymer_offsets
            of types particle_types
            with force constant force_const
            and angle [radians]
        """
        for x in range(1, 4):
            for y in range(1, 4):
                offsets1 = ReaddyUtil.clamp_polymer_offsets_2D(x, polymer_offsets1)
                offsets2 = ReaddyUtil.clamp_polymer_offsets_2D(x, polymer_offsets2)
                offsets3 = ReaddyUtil.clamp_polymer_offsets_2D(x, polymer_offsets3)
                self.add_angle(
                    ReaddyUtil.get_types_with_polymer_numbers_2D(
                        particle_types1, x, y, offsets1
                    ),
                    ReaddyUtil.get_types_with_polymer_numbers_2D(
                        particle_types2, x, y, offsets2
                    ),
                    ReaddyUtil.get_types_with_polymer_numbers_2D(
                        particle_types3, x, y, offsets3
                    ),
                    force_const,
                    angle,
                    system,
                )

    def add_polymer_dihedral_1D(
        self,
        particle_types1,
        polymer_offset1,
        particle_types2,
        polymer_offset2,
        particle_types3,
        polymer_offset3,
        particle_types4,
        polymer_offset4,
        force_const,
        angle,
        system,
        polymer_number_types,
    ):
        """
        adds a cosine dihedral between all polymer numbers
            with offset polymer_offset
            of types particle_types
            with force constant force_const
            and angle [radians]
        """

        for x in range(1, polymer_number_types + 1):
            self.add_dihedral(
                ReaddyUtil.get_types_with_polymer_numbers_1D(
                    particle_types1, x, polymer_offset1, polymer_number_types
                ),
                ReaddyUtil.get_types_with_polymer_numbers_1D(
                    particle_types2, x, polymer_offset2, polymer_number_types
                ),
                ReaddyUtil.get_types_with_polymer_numbers_1D(
                    particle_types3, x, polymer_offset3, polymer_number_types
                ),
                ReaddyUtil.get_types_with_polymer_numbers_1D(
                    particle_types4, x, polymer_offset4, polymer_number_types
                ),
                force_const,
                angle,
                system,
            )

    def add_polymer_dihedral_2D(
        self,
        particle_types1,
        polymer_offsets1,
        particle_types2,
        polymer_offsets2,
        particle_types3,
        polymer_offsets3,
        particle_types4,
        polymer_offsets4,
        force_const,
        angle,
        system,
    ):
        """
        adds a cosine dihedral between all polymer numbers
            with offsets polymer_offsets
            of types particle_types
            with force constant force_const
            and angle [radians]
        """
        for x in range(1, 4):
            for y in range(1, 4):
                offsets1 = ReaddyUtil.clamp_polymer_offsets_2D(x, polymer_offsets1)
                offsets2 = ReaddyUtil.clamp_polymer_offsets_2D(x, polymer_offsets2)
                offsets3 = ReaddyUtil.clamp_polymer_offsets_2D(x, polymer_offsets3)
                offsets4 = ReaddyUtil.clamp_polymer_offsets_2D(x, polymer_offsets4)
                self.add_dihedral(
                    ReaddyUtil.get_types_with_polymer_numbers_2D(
                        particle_types1, x, y, offsets1
                    ),
                    ReaddyUtil.get_types_with_polymer_numbers_2D(
                        particle_types2, x, y, offsets2
                    ),
                    ReaddyUtil.get_types_with_polymer_numbers_2D(
                        particle_types3, x, y, offsets3
                    ),
                    ReaddyUtil.get_types_with_polymer_numbers_2D(
                        particle_types4, x, y, offsets4
                    ),
                    force_const,
                    angle,
                    system,
                )

    @staticmethod
    def create_readdy_simulation(
        system, n_cpu, sim_name="", total_steps=0, record=False, save_checkpoints=False
    ):
        """
        Create the ReaDDy simulation
        """
        simulation = system.simulation("CPU")
        simulation.kernel_configuration.n_threads = n_cpu
        if record:
            simulation.output_file = sim_name + ".h5"
            if os.path.exists(simulation.output_file):
                os.remove(simulation.output_file)
            recording_stride = max(int(total_steps / 1000.0), 1)
            simulation.record_trajectory(recording_stride)
            simulation.observe.topologies(recording_stride)
            simulation.observe.particles(recording_stride)
            simulation.observe.reaction_counts(1)
            simulation.progress_output_stride = recording_stride
        if save_checkpoints:
            checkpoint_stride = max(int(total_steps / 10.0), 1)
            checkpoint_path = f"checkpoints/{os.path.basename(sim_name)}/"
            if os.path.exists(checkpoint_path):
                rmtree(checkpoint_path)
            simulation.make_checkpoints(checkpoint_stride, checkpoint_path, 0)
        return simulation

    @staticmethod
    def get_current_particle_edges(current_topologies):
        """
        During a running simulation,
        get all the edges in the ReaDDy topologies
        as (particle1 id, particle2 id)
        from readdy.simulation.current_topologies
        """
        result = []
        for top in current_topologies:
            for v1, v2 in top.graph.edges:
                p1_id = top.particle_id_of_vertex(v1)
                p2_id = top.particle_id_of_vertex(v2)
                if p1_id <= p2_id:
                    result.append((p1_id, p2_id))
        return result

    @staticmethod
    def get_current_monomers(current_topologies):
        """
        During a running simulation,
        get data for topologies of particles
        """
        edges = ReaddyUtil.get_current_particle_edges(current_topologies)
        result = {
            "topologies": {},
            "particles": {},
        }
        for index, topology in enumerate(current_topologies):
            particle_ids = []
            for p in topology.particles:
                particle_ids.append(p.id)
                neighbor_ids = []
                for edge in edges:
                    if p.id == edge[0]:
                        neighbor_ids.append(edge[1])
                    elif p.id == edge[1]:
                        neighbor_ids.append(edge[0])
                result["particles"][p.id] = {
                    "type_name": p.type,
                    "position": p.pos,
                    "neighbor_ids": neighbor_ids,
                }
            result["topologies"][index] = {
                "type_name": topology.type,
                "particle_ids": particle_ids,
            }
        return result

    @staticmethod
    def _shape_frame_edge_data_from_file(time_index, topology_records):
        """
        After a simulation has finished,
        get all the edges at the given time index
        as (particle1 id, particle2 id)

        topology_records from
        readdy.Trajectory(h5_file_path).read_observable_topologies()
        """
        result = []
        for top in topology_records[time_index]:
            for e1, e2 in top.edges:
                if e1 <= e2:
                    ix1 = top.particles[e1]
                    ix2 = top.particles[e2]
                    result.append((ix1, ix2))
        return result

    @staticmethod
    def _shape_frame_monomer_data_from_file(
        time_index, topology_records, ids, types, positions, traj
    ):
        """
        After a simulation has finished,
        get data for topologies of particles

        traj from readdy.Trajectory(h5_file_path)
        topology_records from traj.read_observable_topologies()
        ids, types, positions from traj.read_observable_particles()
        """
        edges = ReaddyUtil._shape_frame_edge_data_from_file(
            time_index, topology_records
        )
        result = {
            "topologies": {},
            "particles": {},
        }
        for index, top in enumerate(topology_records[time_index]):
            result["topologies"][index] = {
                "type_name": top.type,
                "particle_ids": top.particles,
            }
        for p in range(len(ids[time_index])):
            p_id = ids[time_index][p]
            position = positions[time_index][p]
            neighbor_ids = []
            for edge in edges:
                if p_id == edge[0]:
                    neighbor_ids.append(edge[1])
                elif p_id == edge[1]:
                    neighbor_ids.append(edge[0])
            result["particles"][ids[time_index][p]] = {
                "type_name": traj.species_name(types[time_index][p]),
                "position": np.array([position[0], position[1], position[2]]),
                "neighbor_ids": neighbor_ids,
            }
        return result

    @staticmethod
    def _shape_monomer_data_from_file(
        min_time,
        max_time,
        time_inc,
        times,
        topology_records,
        ids,
        types,
        positions,
        traj,
    ):
        """
        For each time point, get monomer data and times
        """
        print("Shaping data for analysis...")
        result = []
        new_times = []
        for t in tqdm(range(len(times))):
            if t >= min_time and t <= max_time and t % time_inc == 0:
                result.append(
                    ReaddyUtil._shape_frame_monomer_data_from_file(
                        t, topology_records, ids, types, positions, traj
                    )
                )
                new_times.append(times[t])
        return result, np.array(new_times)

    @staticmethod
    def monomer_data_and_reactions_from_file(
        h5_file_path,
        stride=1,
        timestep=0.1,
        reaction_names=None,
        pickle_file_path=None,
        save_pickle_file=False,
    ):
        """
        For data saved in a ReaDDy .h5 file:

        For each time point, get monomer data as a dictionary:
        {
          "topologies" : mapping of topology id to data for each topology:
            [id: int] : {
                "type_name" : str,
                "particle_ids" : List[int]
            }
          "particles" : mapping of particle id to data for each particle:
            [id: int] : {
                "type_name" : str,
                "position" : np.ndarray,
                "neighbor_ids" : List[int]
            }
        }

        Also return the counts of reactions over time,
        the timestamps for each frame,
        and the reaction time increment in seconds
        """
        if pickle_file_path is not None:
            print("Loading pickle file for shaped data")
            import pickle

            data = []
            with open(pickle_file_path, "rb") as f:
                while True:
                    try:
                        data.append(pickle.load(f))
                    except EOFError:
                        break
            monomer_data, reactions, times, time_inc_s = data[0]
            return monomer_data, reactions, times, time_inc_s
        else:
            trajectory = readdy.Trajectory(h5_file_path)
            _, topology_records = trajectory.read_observable_topologies()
            (
                times,
                types,
                ids,
                positions,
            ) = trajectory.read_observable_particles()
            monomer_data, times = ReaddyUtil._shape_monomer_data_from_file(
                0,
                times.shape[0],
                stride,
                times,
                topology_records,
                ids,
                types,
                positions,
                trajectory,
            )
            times = timestep / 1e3 * times  # index --> microseconds
            # times = timestep * times  # index --> nanoseconds
            reactions = None
            time_inc_s = None
            if reaction_names is not None:
                recorded_steps = stride * (len(times) - 1)
                reactions = ReaddyUtil.load_reactions(
                    trajectory, stride, reaction_names, recorded_steps
                )
                time_inc_s = times[-1] * 1e-6 / (len(times) - 1)
            data = [monomer_data, reactions, times, time_inc_s]
            if save_pickle_file:
                import pickle

                fname = h5_file_path + ".dat"
                with open(fname, "wb") as f:
                    pickle.dump(data, f)
        return monomer_data, reactions, times, time_inc_s

    @staticmethod
    def vector_is_invalid(v):
        """
        check if any of a 3D vector's components are NaN
        """
        return math.isnan(v[0]) or math.isnan(v[1]) or math.isnan(v[2])

    @staticmethod
    def analyze_frame_get_count_of_topologies_of_type(
        time_index, topology_type, topology_records, traj
    ):
        """
        Get the number of topologies of a given type at the given time index
        """
        result = 0
        for top in topology_records[time_index]:
            if traj.topology_type_name(top.type) == topology_type:
                result += 1
        return result

    @staticmethod
    def analyze_frame_get_neighbor_ids_of_types(
        particle_id,
        particle_types,
        frame_particle_data,
        exact_match,
    ):
        """
        Get a list of ids for all the neighbors of particle_id with particle type
        in the given list of types in the given frame of data
        """
        result = []
        for neighbor_id in frame_particle_data["particles"][particle_id][
            "neighbor_ids"
        ]:
            if neighbor_id in frame_particle_data["particles"]:
                type_name = frame_particle_data["particles"][neighbor_id]["type_name"]
                for particle_type in particle_types:
                    if (exact_match and type_name == particle_type) or (
                        not exact_match and particle_type in type_name
                    ):
                        result.append(neighbor_id)
                        break
        return result

    @staticmethod
    def analyze_frame_get_ids_for_types(particle_types, frame_particle_data):
        """
        Get a list of ids for all the particles with particle type
        in the given list of types in the given frame of data
        """
        result = []
        for p_id in frame_particle_data["particles"]:
            if frame_particle_data["particles"][p_id]["type_name"] in particle_types:
                result.append(p_id)
        return result

    @staticmethod
    def analyze_frame_get_id_for_neighbor_of_types(
        particle_id,
        neighbor_types,
        frame_particle_data,
        exclude_ids=[],
        exact_match=True,
    ):
        """
        Get the id for the first neighbor with one of the neighbor_types
        in the given frame of data
        """
        for neighbor_id in frame_particle_data["particles"][particle_id][
            "neighbor_ids"
        ]:
            if neighbor_id not in frame_particle_data["particles"]:
                neighbor_id = str(neighbor_id)
            if neighbor_id in exclude_ids:
                continue
            current_neighbor_type = frame_particle_data["particles"][neighbor_id][
                "type_name"
            ]
            for neighbor_type in neighbor_types:
                if (exact_match and current_neighbor_type == neighbor_type) or (
                    not exact_match
                    and (
                        neighbor_type in current_neighbor_type
                        or current_neighbor_type in neighbor_type
                    )
                ):
                    return neighbor_id
        return None

    @staticmethod
    def analyze_frame_get_chain_of_types(
        start_particle_id,
        neighbor_types,
        frame_particle_data,
        chain_length=0,
        last_particle_id=None,
        result=[],
        next_neighbor_index=None,
        exact_match=True,
    ):
        """
        Starting from the particle with start_particle_id,
        get ids for a chain of particles with neighbor_types in the given frame of data,
        avoiding the particle with last_particle_id,
        if chain_length = 0, return entire chain
        """
        if next_neighbor_index is not None:
            n_types = [neighbor_types[next_neighbor_index]]
        else:
            n_types = neighbor_types

        n_id = ReaddyUtil.analyze_frame_get_id_for_neighbor_of_types(
            start_particle_id,
            n_types,
            frame_particle_data,
            [last_particle_id] if last_particle_id is not None else [],
            exact_match=exact_match,
        )
        if n_id is None:
            return result
        result.append(n_id)

        if chain_length == 0:
            return result

        next_neighbor_index = (next_neighbor_index + 1) % len(neighbor_types)

        return ReaddyUtil.analyze_frame_get_chain_of_types(
            n_id,
            neighbor_types,
            frame_particle_data,
            chain_length - 1 if chain_length > 0 else 0,
            start_particle_id,
            result,
            next_neighbor_index=next_neighbor_index,
            exact_match=exact_match,
        )

    @staticmethod
    def load_reactions(trajectory, stride, total_reactions_mapping, recorded_steps=1e3):
        """
        Read reaction counts per frame from a ReaDDy trajectory
        and create a DataFrame with the number of times each
        ReaDDy reaction and total set of reactions has happened
        by each time step / stride
        """
        print("Loading reactions...")
        reaction_times, readdy_reactions = trajectory.read_observable_reaction_counts()
        flat_readdy_reactions = {}
        for rxn_type in readdy_reactions:
            flat_readdy_reactions = dict(
                flat_readdy_reactions, **readdy_reactions[rxn_type]
            )
        reaction_names = list(np.column_stack(flat_readdy_reactions)[0])
        reaction_data = np.column_stack([x for x in flat_readdy_reactions.values()])
        interval = int((reaction_data.shape[0] - 1) * stride / float(recorded_steps))
        reaction_data = np.sum(
            reaction_data[:-1].reshape(-1, interval, len(reaction_names)), axis=1
        )
        reactions_df = pd.DataFrame(reaction_data, columns=reaction_names)
        for total_rxn_name in total_reactions_mapping:
            if total_rxn_name in reactions_df:
                continue
            reactions_df[total_rxn_name] = 0.0
            for rxn_name in total_reactions_mapping[total_rxn_name]:
                if rxn_name in reactions_df.columns:
                    reactions_df[total_rxn_name] += reactions_df[rxn_name]
                else:
                    print(f"Couldn't find {rxn_name} in ReaDDy reactions.")
        return reactions_df

    # read in box size
    @staticmethod
    def get_box_size(input_size):
        if isinstance(input_size, str):
            lengths = input_size.split(",")
            if len(lengths) != 3:
                print("INCORRECT BOX SIZE. PLEASE CHECK INPUT FILE.")
            return np.array([float(length) for length in lengths])
        else:
            return np.array([float(input_size)] * 3)
