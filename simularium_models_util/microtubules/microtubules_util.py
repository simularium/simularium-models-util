#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import readdy
import math
import random
import copy

from ..common import ReaddyUtil


parameters = {}


def set_parameters(p):
    global parameters
    parameters = p
    return p


class MicrotubulesUtil:
    def __init__(self, parameters):
        """
        Utility functions for ReaDDy microtubules models

        Parameters need to be accessible in ReaDDy callbacks
        which can't be instance methods, so parameters are global
        """
        set_parameters(parameters)

    @staticmethod
    def get_random_tubulin_neighbors(
        topology, types_include, types_exclude, GTP_state, polymer_offsets
    ):
        """
        get a random pair of neighbor tubulins of the given types,
        GTP state, and polymer offsets
        """
        vertices = []
        for v in topology.graph.get_vertices():
            pt = topology.particle_type_of_vertex(v)
            if ReaddyUtil.vertex_satisfies_type(
                pt,
                types_include[0] + (["GDP"] if GTP_state == "GDP" else []),
                types_exclude[0],
            ):
                vertices.append(v)
        if GTP_state == "GDP":
            types_include[1] += ["GDP"]
        neighbors = []
        for v in vertices:
            n = MicrotubulesUtil.get_neighboring_tubulin(topology, v, polymer_offsets)
            if n is not None:
                extra_types = []
                if (
                    GTP_state == "GTP"
                    and "GTP" not in topology.particle_type_of_vertex(v)
                ):
                    extra_types = ["GTP"]
                nt = topology.particle_type_of_vertex(n)
                if ReaddyUtil.vertex_satisfies_type(
                    nt, types_include[1] + extra_types, types_exclude[1]
                ):
                    neighbors.append([v, n])
        if len(neighbors) == 0:
            return None
        return random.choice(neighbors)

    @staticmethod
    def get_neighboring_tubulin(topology, vertex, polymer_offsets):
        """
        get the next tubulin neighbor in the branch from site named direction
        """
        return ReaddyUtil.get_neighbor_of_type(
            topology,
            vertex,
            MicrotubulesUtil.polymer_indices_to_string(
                MicrotubulesUtil.increment_polymer_indices(
                    MicrotubulesUtil.get_polymer_indices(
                        topology.particle_type_of_vertex(vertex)
                    ),
                    polymer_offsets,
                )
            ),
            False,
        )

    @staticmethod
    def tubulin_has_sites(topology, tubulin):
        """
        does the tubulin have sites attached?
        """
        return (
            ReaddyUtil.get_neighbor_of_type(topology, tubulin, "site#out", False)
            is not None
        )

    @staticmethod
    def get_tubulin_sites(topology, tubulin):
        """
        get the site particles attached to this tubulin vertex
        """
        if not MicrotubulesUtil.tubulin_has_sites(topology, tubulin):
            return None
        return [
            ReaddyUtil.get_neighbor_of_type(topology, tubulin, "site#out", False),
            ReaddyUtil.get_neighbor_of_type(topology, tubulin, "site#1", False),
            ReaddyUtil.get_neighbor_of_type(topology, tubulin, "site#2", False),
            ReaddyUtil.get_neighbor_of_type(topology, tubulin, "site#3", False),
            ReaddyUtil.get_neighbor_of_type(topology, tubulin, "site#4", False),
        ]

    @staticmethod
    def polymer_indices_to_string(polymer_indices):
        """
        get the x and y polymer index for a particle
        """
        return f"_{polymer_indices[0]}_{polymer_indices[1]}"

    @staticmethod
    def increment_polymer_indices(polymer_indices, polymer_offsets):
        """
        increment the x and y polymer index for a particle
        """
        polymer_offsets = ReaddyUtil.clamp_polymer_offsets_2D(
            polymer_indices[0], polymer_offsets
        )
        x = ReaddyUtil.calculate_polymer_number(polymer_indices[0], polymer_offsets[0])
        y = ReaddyUtil.calculate_polymer_number(polymer_indices[1], polymer_offsets[1])
        return [x, y]

    @staticmethod
    def get_polymer_indices(particle_type):
        """
        get the x and y polymer index for a particle
        """
        if "tubulin" not in particle_type:
            return []
        flag_string = particle_type[particle_type.index("#") + 1 :]
        flags = flag_string.split("_")
        if len(flags) <= 1:
            return []
        x = int(flags[-2])
        y = int(flags[-1])
        return [x, y]

    @staticmethod
    def get_polymer_offsets(particle_types):
        """
        get the offsets between two particle's polymer indices
        """
        polymer_indices = [
            MicrotubulesUtil.get_polymer_indices(particle_types[0]),
            MicrotubulesUtil.get_polymer_indices(particle_types[1]),
        ]
        offsets = []
        for i in range(2):
            offset = int(polymer_indices[1][i] - polymer_indices[0][i])
            if abs(offset) > 1:
                offset = int(round(-offset / abs(offset)))
            offsets.append(offset)
        return offsets

    @staticmethod
    def get_filament_lengths(topology, tubulin_minus, tubulin_plus):
        """
        get the lengths of the filaments, cut off the larger one
        at one larger than the smaller
        """
        next_tub_minus = MicrotubulesUtil.get_neighboring_tubulin(
            topology, tubulin_minus, [-1, 0]
        )
        next_tub_plus = MicrotubulesUtil.get_neighboring_tubulin(
            topology, tubulin_plus, [1, 0]
        )
        if next_tub_minus is None and next_tub_plus is None:
            return np.array([1, 1])
        elif next_tub_minus is None:
            return np.array([1, 2])
        elif next_tub_plus is None:
            return np.array([2, 1])
        else:
            return MicrotubulesUtil.get_filament_lengths(
                topology, next_tub_minus, next_tub_plus
            ) + np.array([1, 1])

    @staticmethod
    def filament_is_crosslinked(topology, tubulin, direction):
        """
        does the fragment starting at the given tubulin
        and going the given direction (-1 or 1) along a filament
        have any crosslinks to other filaments?
        """
        for i in range(2):
            neighbor_tub = MicrotubulesUtil.get_neighboring_tubulin(
                topology, tubulin, [0, -1 if i == 0 else 1]
            )
            if neighbor_tub is not None:
                return True
        next_tub = MicrotubulesUtil.get_neighboring_tubulin(
            topology, tubulin, [direction, 0]
        )
        if next_tub is None:
            return False
        return MicrotubulesUtil.filament_is_crosslinked(topology, next_tub, direction)

    @staticmethod
    def remove_bent_filament_site_bonds(topology, recipe, tubulin_minus, tubulin_plus):
        """
        remove bonds between the sites of tubulin1 and tubulin2
        """
        v_sites_minus = MicrotubulesUtil.get_tubulin_sites(topology, tubulin_minus)
        if v_sites_minus is None:
            raise Exception(
                "Failed to find sites on bent tubulin ("
                + ReaddyUtil.vertex_to_string(topology, tubulin_minus)
                + ")\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        v_sites_plus = MicrotubulesUtil.get_tubulin_sites(topology, tubulin_plus)
        if v_sites_plus is None:
            raise Exception(
                "Failed to find sites on bent tubulin ("
                + ReaddyUtil.vertex_to_string(topology, tubulin_plus)
                + ")\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        for i in range(3):
            if v_sites_plus[i] is None or v_sites_minus[i] is None:
                continue
            removed, message = ReaddyUtil.try_remove_edge(
                topology, recipe, v_sites_plus[i], v_sites_minus[i]
            )
            if not removed:
                raise Exception(
                    message + "\n" + ReaddyUtil.topology_to_string(topology)
                )
        if v_sites_plus[3] is not None and v_sites_minus[4] is not None:
            removed, message = ReaddyUtil.try_remove_edge(
                topology, recipe, v_sites_plus[3], v_sites_minus[4]
            )
            if not removed:
                raise Exception(
                    message + "\n" + ReaddyUtil.topology_to_string(topology)
                )

    @staticmethod
    def remove_tubulin_sites(topology, recipe, tubulin):
        """
        emit a tubulin's site particles
        """
        v_sites = MicrotubulesUtil.get_tubulin_sites(topology, tubulin)
        if v_sites is None:
            return False
        for site in range(5):
            if v_sites[site] is not None:
                recipe.separate_vertex(v_sites[site])
                recipe.change_particle_type(v_sites[site], "site#remove")
        return True

    @staticmethod
    def set_free(topology, recipe, tubulins):
        """
        emit each tubulin's site particles and change its type to tubulin#free
        """
        for tubulin in tubulins:
            if not MicrotubulesUtil.remove_tubulin_sites(topology, recipe, tubulin):
                return False
            tub_type = "A" if "A" in topology.particle_type_of_vertex(tubulin) else "B"
            recipe.change_particle_type(tubulin, f"tubulin{tub_type}#free")
        return True

    @staticmethod
    def topology_is_microtubule(topology):
        """
        does the topology have tubulins with ring bonds?
        if so it has multiple protofilaments and is not just an oligomer
        """
        for vertex in topology.graph.get_vertices():
            pt = topology.particle_type_of_vertex(vertex)
            if "tubulin" in pt:
                if (
                    MicrotubulesUtil.get_neighboring_tubulin(topology, vertex, [0, -1])
                    is not None
                ):
                    return True
                if (
                    MicrotubulesUtil.get_neighboring_tubulin(topology, vertex, [0, 1])
                    is not None
                ):
                    return True
        return False

    @staticmethod
    def get_attaching_sites(topology):
        """
        get the ring sites that just attached in a spatial reaction
        """
        sites1 = ReaddyUtil.get_vertices_of_type(topology, "site#1", True)
        for site1 in sites1:
            site2 = ReaddyUtil.get_neighbor_of_type(topology, site1, "site#2", True)
            if site2 is not None:
                return [site1, site2]
        return []

    @staticmethod
    def cancel_attach(topology, recipe, sites):
        """
        reverse an attach spatial reaction
        """
        removed, message = ReaddyUtil.try_remove_edge(
            topology, recipe, sites[0], sites[1]
        )
        if not removed:
            raise Exception(message + "\n" + ReaddyUtil.topology_to_string(topology))
        recipe.change_topology_type("Microtubule")

    @staticmethod
    def tubulins_can_attach(tubulin_types):
        """
        check if tubulins can attach laterally
        """
        offsets = MicrotubulesUtil.get_polymer_offsets(tubulin_types)
        return offsets[0] == 0 and offsets[1] == -1

    @staticmethod
    def tubulin_is_crosslinked(topology, tubulin):
        """
        is the tubulin connected to a neighbor on each ring side?
        """
        return (
            MicrotubulesUtil.get_neighboring_tubulin(topology, tubulin, [0, -1])
            is not None
            and MicrotubulesUtil.get_neighboring_tubulin(topology, tubulin, [0, 1])
            is not None
        )

    @staticmethod
    def check_remove_tubulin_sites(
        topology,
        recipe,
        tubulin,
        tubulin_crosslinked=False,
        neighbor_tubulin=None,
        neighbor_crosslinked=False,
    ):
        """
        check if the tubulin's sites should be removed, if so remove them
        """
        if not MicrotubulesUtil.tubulin_has_sites(topology, tubulin):
            return False
        crosslinked = []
        neighbor_tubulins = [
            MicrotubulesUtil.get_neighboring_tubulin(topology, tubulin, [-1, 0]),
            MicrotubulesUtil.get_neighboring_tubulin(topology, tubulin, [1, 0]),
        ]
        for nt in range(2):
            if (
                neighbor_tubulin is not None
                and neighbor_tubulins[nt] is not None
                and ReaddyUtil.vertices_are_equal(
                    topology, neighbor_tubulins[nt], neighbor_tubulin
                )
            ):
                crosslinked.append(neighbor_crosslinked)
            else:
                if neighbor_tubulins[nt] is not None:
                    crosslinked.append(
                        MicrotubulesUtil.tubulin_is_crosslinked(
                            topology, neighbor_tubulins[nt]
                        )
                    )
                else:
                    crosslinked.append(nt == 0)
        if False not in crosslinked and (
            tubulin_crosslinked
            or MicrotubulesUtil.tubulin_is_crosslinked(topology, tubulin)
        ):
            MicrotubulesUtil.remove_tubulin_sites(topology, recipe, tubulin)
            return True
        return False

    @staticmethod
    def tubulin_is_not_crosslinked(topology, tubulin):
        """
        is the tubulin missing at least one ring side connection?
        """
        return (
            MicrotubulesUtil.get_neighboring_tubulin(topology, tubulin, [0, -1]) is None
            or MicrotubulesUtil.get_neighboring_tubulin(topology, tubulin, [0, 1])
            is None
        )

    @staticmethod
    def add_tubulin_sites(
        topology, recipe, tubulin, site1_type="site#new", site2_type="site#new"
    ):
        """
        add a new particle attached to the tubulin for each site
        """

        pos_tub = ReaddyUtil.get_vertex_position(topology, tubulin)
        for i in range(5):
            recipe.append_particle(
                [tubulin],
                site1_type if i == 1 else site2_type if i == 2 else "site#new",
                pos_tub + np.array([0, 0, 1.5]),
            )

    @staticmethod
    def check_add_tubulin_sites(
        topology, recipe, tubulin, site1_type="site#new", site2_type="site#new"
    ):
        """
        check if the tubulin should have sites added, if so add them
        """
        if MicrotubulesUtil.tubulin_has_sites(topology, tubulin):
            if not (site1_type == "site#new" and site2_type == "site#new"):
                sites = MicrotubulesUtil.get_tubulin_sites(topology, tubulin)
                if site1_type != "site#new":
                    recipe.change_particle_type(sites[1], site1_type)
                if site2_type != "site#new":
                    recipe.change_particle_type(sites[2], site2_type)
            return False
        MicrotubulesUtil.add_tubulin_sites(
            topology, recipe, tubulin, site1_type, site2_type
        )
        return True

    @staticmethod
    def get_new_sites(topology, tubulin):
        """
        get new site particles attached to a tubulin
        """
        v_new_sites = ReaddyUtil.get_neighbors_of_type(
            topology, tubulin, "site#", False
        )
        return v_new_sites if len(v_new_sites) >= 1 else None

    @staticmethod
    def setup_sites(
        topology,
        recipe,
        v_new_sites,
        position,
        side,
        normal,
        tangent,
        site_state_ring,
        site_state_filament,
    ):
        """
        set positions, types, and intra-tubulin edges for new site particles
        """
        recipe.change_particle_type(v_new_sites[0], "site#out")
        recipe.change_particle_position(v_new_sites[0], position + normal)
        recipe.change_particle_type(v_new_sites[1], f"site#1{site_state_ring}")
        recipe.change_particle_position(v_new_sites[1], position + side)
        recipe.add_edge(v_new_sites[0], v_new_sites[1])  # edge to site#out
        recipe.change_particle_type(v_new_sites[2], f"site#2{site_state_ring}")
        recipe.change_particle_position(v_new_sites[2], position - side)
        recipe.add_edge(v_new_sites[0], v_new_sites[2])  # edge to site#out
        recipe.change_particle_type(v_new_sites[3], "site#3")
        recipe.change_particle_position(v_new_sites[3], position - tangent)
        recipe.add_edge(v_new_sites[0], v_new_sites[3])  # edge to site#out
        recipe.add_edge(v_new_sites[1], v_new_sites[3])  # edge to site#1
        recipe.add_edge(v_new_sites[2], v_new_sites[3])  # edge to site#2
        recipe.change_particle_type(v_new_sites[4], f"site#4{site_state_filament}")
        recipe.change_particle_position(v_new_sites[4], position + tangent)
        recipe.add_edge(v_new_sites[0], v_new_sites[4])  # edge to site#out
        recipe.add_edge(v_new_sites[1], v_new_sites[4])  # edge to site#1
        recipe.add_edge(v_new_sites[2], v_new_sites[4])  # edge to site#2

    @staticmethod
    def connect_sites_between_tubulins(topology, recipe, v_sites_minus, v_sites_plus):
        """
        add inter-tubulin edges between sites
        """
        recipe.add_edge(v_sites_plus[0], v_sites_minus[0])  # + site0 -- - site0
        recipe.add_edge(v_sites_plus[1], v_sites_minus[1])  # + site1 -- - site1
        recipe.add_edge(v_sites_plus[2], v_sites_minus[2])  # + site2 -- - site2
        recipe.add_edge(v_sites_plus[3], v_sites_minus[4])  # + site3 -- - site4

    @staticmethod
    def get_all_polymer_tubulin_types(particle_type):
        """
        gets a list of all polymer numbers
        ("type1_1", "type1_2", "type1_3", "type2_1", ... "type3_3")
            for type particle_type

            returns list of types
        """
        result = []
        for x in range(1, 4):
            for y in range(1, 4):
                result.append(f"{particle_type}{x}_{y}")
        return result

    @staticmethod
    def add_polymer_topology_species(particle_type, diffCoeff, system):
        """
        adds topology species for all polymer numbers
        ("type1_1", "type1_2", "type1_3", "type2_1", ... "type3_3")
            for type particle_type
            with diffusion coefficient diffCoeff [nm^2/s]
        """
        types = MicrotubulesUtil.get_all_polymer_tubulin_types(particle_type)
        for t in types:
            system.add_topology_species(t, diffCoeff)

    @staticmethod
    def get_microtubule_positions_and_types(
        n_filaments,
        n_rings,
        n_frayed_rings_plus,
        n_frayed_rings_minus,
        frayed_angle,
        radius,
        use_GTP=True,
    ):
        """
        get lists of positions and types for particles in a microtubule
            with n_filaments protofilaments
            and n_rings rings
            and n_frayed_rings_plus rings at + end with outward bend
            and n_frayed_rings_minus rings at - end with outward bend
            and frayed_angle [radians] rotation of normal per bent tubulin
            and radius [nm]
        """
        frayed_y_pos = 0.0
        for ring in range(n_frayed_rings_minus):
            angle = (ring + 0.5) * frayed_angle
            quad = math.floor(angle / (np.pi / 2))
            acute_angle = angle % (np.pi / 2)
            frayed_y_pos += (
                (-1 if quad > 1 else 1)
                * 4
                * (np.sin(acute_angle) if quad % 2 == 0 else np.cos(acute_angle))
            )
        n_frayed_rings = n_frayed_rings_minus + n_frayed_rings_plus
        positions = []
        types = []
        i = 0
        for filament in range(n_filaments):
            tube_angle = (
                n_filaments - filament
            ) * 2.0 * np.pi / n_filaments + np.pi / 2.0
            tube_normal = ReaddyUtil.normalize(
                np.array([math.cos(tube_angle), math.sin(tube_angle), 0.0])
            )
            tube_tangent = np.array([0.0, 0.0, 1.0])
            side = ReaddyUtil.normalize(np.cross(tube_normal, tube_tangent))
            if n_frayed_rings > 0:
                tangent = ReaddyUtil.rotate(
                    np.copy(tube_tangent),
                    side,
                    (n_frayed_rings_minus + 0.5) * frayed_angle,
                )
                normal = ReaddyUtil.rotate(
                    np.copy(tube_normal),
                    side,
                    (n_frayed_rings_minus + 0.5) * frayed_angle,
                )
            else:
                tangent = tube_tangent
                normal = tube_normal
            pos = (radius + frayed_y_pos) * tube_normal + (
                12.0 / 13.0 * filament - 2.0 * n_rings
            ) * tube_tangent
            for ring in range(n_rings):
                number1 = ring % 3 + 1
                number2 = (filament + math.floor(ring / 3)) % 3 + 1
                GTP_state = (
                    "GTP"
                    if random.random() <= (ring - n_rings / 3.0) / (n_rings / 3.0)
                    else "GDP"
                )
                positions.append(copy.copy(pos))
                bent = (
                    ring < n_frayed_rings_minus
                    or ring > n_rings - 1 - n_frayed_rings_plus
                )
                edge = n_frayed_rings > 0 and (
                    ring == n_frayed_rings_minus
                    or ring == n_rings - 1 - n_frayed_rings_plus
                )
                tub_type = "A" if ring % 2 == 0 else "B"
                bent_type = "bent_" if bent else ""
                tub_GTP_state = GTP_state + "_" if use_GTP else ""
                types.append(
                    f"tubulin{tub_type}#{tub_GTP_state}{bent_type}{number1}_{number2}"
                )
                i += 1
                if bent or edge:  # needs site scaffolds
                    types.append("site#out")
                    i += 1
                    positions.append(pos + 1.5 * normal)
                    site_state = f"_{GTP_state}" if not edge else ""
                    types.append(f"site#1{site_state}")
                    i += 1
                    positions.append(pos - 1.5 * side)
                    types.append(f"site#2{site_state}")
                    i += 1
                    positions.append(pos + 1.5 * side)
                    types.append("site#3")
                    i += 1
                    positions.append(pos - 1.5 * tangent)
                    site_state = f"_{GTP_state}" if ring == n_rings - 1 else ""
                    types.append(f"site#4{site_state}")
                    i += 1
                    positions.append(pos + 1.5 * tangent)
                    tangent = ReaddyUtil.rotate(
                        tangent, side, -frayed_angle * (0.5 if edge else 1.0)
                    )
                    normal = ReaddyUtil.rotate(
                        normal, side, -frayed_angle * (0.5 if edge else 1.0)
                    )
                pos += 4.0 * tangent
        return positions, types

    @staticmethod
    def add_edges(
        microtubule, n_filaments, n_rings, n_frayed_rings_plus, n_frayed_rings_minus
    ):
        """
        add edges to a microtubule topology
            with n_filaments protofilaments
            and n_rings rings
            and n_frayed_rings_plus rings at + end with outward bend
            and n_frayed_rings_minus rings at - end with outward bend
        """
        sites_per_scaffold = 5
        n_frayed_rings = n_frayed_rings_minus + n_frayed_rings_plus
        particles_per_filament = n_rings + sites_per_scaffold * (
            n_frayed_rings + (2 if n_frayed_rings > 0 else 0)
        )
        total_particles = n_filaments * particles_per_filament
        for filament in range(n_filaments):
            for ring in range(n_rings):
                end = (
                    "minus"
                    if n_frayed_rings_minus > 0 and ring <= n_frayed_rings_minus + 1
                    else "plus"
                )
                bent = (
                    ring < n_frayed_rings_minus
                    or ring > n_rings - 1 - n_frayed_rings_plus
                )
                edge = n_frayed_rings > 0 and (
                    ring == n_frayed_rings_minus
                    or ring == n_rings - 1 - n_frayed_rings_plus
                )
                i = (
                    filament * particles_per_filament
                    + ring
                    + sites_per_scaffold
                    * (
                        (ring if end == "minus" else n_frayed_rings_minus + 1)
                        if n_frayed_rings_minus > 0
                        else 0
                    )
                    + sites_per_scaffold
                    * (
                        ring - ((n_rings - n_frayed_rings_plus) - 1)
                        if end == "plus" and bent
                        else 0
                    )
                )
                if ring < n_rings - 1:  # bond along filament
                    i_filament = i + 1 + (sites_per_scaffold if bent or edge else 0)
                    microtubule.get_graph().add_edge(i, i_filament)
                if not bent:  # bond along ring
                    # as long as not in + end overhang
                    if not (
                        filament == n_filaments - 1
                        and ring > (n_rings - 1 - n_frayed_rings_plus) - 3
                    ):
                        i_ring = i + particles_per_filament
                        if filament == n_filaments - 1:
                            i_ring += (
                                8 if end == "minus" and edge else 3
                            ) - total_particles
                        microtubule.get_graph().add_edge(i, i_ring)
                if bent or edge:  # bent site bonds
                    microtubule.get_graph().add_edge(i, i + 1)  # tub -- site0
                    microtubule.get_graph().add_edge(i, i + 2)  # tub -- site1
                    microtubule.get_graph().add_edge(i, i + 3)  # tub -- site2
                    microtubule.get_graph().add_edge(i, i + 4)  # tub -- site3
                    microtubule.get_graph().add_edge(i, i + 5)  # tub -- site4
                    microtubule.get_graph().add_edge(i + 1, i + 2)  # site0 -- site1
                    microtubule.get_graph().add_edge(i + 1, i + 3)  # site0 -- site2
                    microtubule.get_graph().add_edge(i + 1, i + 4)  # site0 -- site3
                    microtubule.get_graph().add_edge(i + 1, i + 5)  # site0 -- site4
                    microtubule.get_graph().add_edge(i + 2, i + 4)  # site1 -- site3
                    microtubule.get_graph().add_edge(i + 2, i + 5)  # site1 -- site4
                    microtubule.get_graph().add_edge(i + 3, i + 4)  # site2 -- site3
                    microtubule.get_graph().add_edge(i + 3, i + 5)  # site2 -- site4
                    # longitudinal bonds on bent rings
                    if ring > 0 and (bent or (edge and end == "minus")):
                        # site0 -- prev site0
                        microtubule.get_graph().add_edge(i + 1, i - 5)
                        # site1 -- prev site1
                        microtubule.get_graph().add_edge(i + 2, i - 4)
                        # site2 -- prev site2
                        microtubule.get_graph().add_edge(i + 3, i - 3)
                        # site3 -- prev site4
                        microtubule.get_graph().add_edge(i + 4, i - 1)

    @staticmethod
    def add_microtubule(
        n_rings,
        n_frayed_rings_minus,
        n_frayed_rings_plus,
        position_offset,
        simulation,
        use_GTP=True,
    ):
        """
        add seed microtubule to the simulation
            with n_filaments protofilaments
            and n_rings rings
            and n_frayed_rings_minus rings at - end with outward bend
            and n_frayed_rings_plus rings at + end with outward bend
            and position_offset
        """
        if n_rings - (n_frayed_rings_minus + n_frayed_rings_plus) < 2:
            raise Exception(
                "Too many frayed rings, " "protofilaments will not form a microtubule"
            )
        n_filaments = 13
        frayed_angle = np.deg2rad(10.0)
        max_frayed_rings = math.floor((360.0 + frayed_angle / 2.0) / frayed_angle)
        n_frayed_rings_minus = min(max_frayed_rings, n_frayed_rings_minus)
        n_frayed_rings_plus = min(max_frayed_rings, n_frayed_rings_plus)
        positions, types = MicrotubulesUtil.get_microtubule_positions_and_types(
            n_filaments,
            n_rings,
            n_frayed_rings_plus,
            n_frayed_rings_minus,
            frayed_angle,
            10.86,
            use_GTP,
        )
        microtubule = simulation.add_topology(
            "Microtubule", types, positions + position_offset
        )
        MicrotubulesUtil.add_edges(
            microtubule, n_filaments, n_rings, n_frayed_rings_plus, n_frayed_rings_minus
        )

    @staticmethod
    def add_tubulin_dimers(simulation, n_tubulin, box_size):
        """
        add seed tubulin dimers to the simulation
        """
        positions = np.random.uniform(size=(n_tubulin, 3)) * box_size - box_size * 0.5
        for p in range(len(positions)):
            to_B = 4.0 * ReaddyUtil.normalize(
                np.array([random.random(), random.random(), random.random()])
            )
            top = simulation.add_topology(
                "Dimer",
                ["tubulinA#free", "tubulinB#free"],
                np.array([positions[p], positions[p] + to_B]),
            )
            top.get_graph().add_edge(0, 1)

    @staticmethod
    def add_tubulin_types(system, diffCoeff):
        """
        add tubulin topology and particle types to the system

        microtubules are 2D polymers and to encode polarity in each dimension,
        there are 3 x 3 = 9 polymer types. These are represented as "type#x_y"
        where x and y are both in [1,3]. spatially, the types are mapped like so:

                                  x_(y+1)
                                   /
                            A ____/____ B
                            /   /    /
        - end     (x-1)_y__/___x_y__/_____(x+1)_y     + end
                          /   /    /
                         /___/____/
                     C     /     D
                          /
                      x_(y-1)
        """
        system.topologies.add_type("Dimer")
        system.add_topology_species("tubulinA#free", diffCoeff)
        system.add_topology_species("tubulinB#free", diffCoeff)
        system.topologies.add_type("Oligomer")
        system.topologies.add_type("Oligomer#Fail-Shrink-GTP")
        system.topologies.add_type("Oligomer#Fail-Shrink-GDP")
        system.topologies.add_type("Microtubule")
        system.topologies.add_type("Microtubule#Growing1-GTP")
        system.topologies.add_type("Microtubule#Growing1-GDP")
        system.topologies.add_type("Microtubule#Growing2-GTP")
        system.topologies.add_type("Microtubule#Growing2-GDP")
        system.topologies.add_type("Microtubule#Shrinking")
        system.topologies.add_type("Microtubule#Fail-Shrink-GTP")
        system.topologies.add_type("Microtubule#Fail-Shrink-GDP")
        system.topologies.add_type("Microtubule#Attaching")
        system.topologies.add_type("Microtubule#Fail-Attach")
        system.topologies.add_type("Microtubule#Detaching-GTP")
        system.topologies.add_type("Microtubule#Detaching-GDP")
        system.topologies.add_type("Microtubule#Fail-Hydrolyze")
        tube_tubulin_types = [
            "tubulinA#GTP_",
            "tubulinA#GDP_",
            "tubulinB#GTP_",
            "tubulinB#GDP_",
        ]
        bent_tubulin_types = [
            "tubulinA#GTP_bent_",
            "tubulinA#GDP_bent_",
            "tubulinB#GTP_bent_",
            "tubulinB#GDP_bent_",
        ]
        all_tubulin_types = tube_tubulin_types + bent_tubulin_types
        for tubulin_type in all_tubulin_types:
            MicrotubulesUtil.add_polymer_topology_species(
                tubulin_type, diffCoeff, system
            )
        site_types = [
            "site#out",
            "site#1",
            "site#1_GTP",
            "site#1_GDP",
            "site#1_detach",
            "site#2",
            "site#2_GTP",
            "site#2_GDP",
            "site#2_detach",
            "site#3",
            "site#4",
            "site#4_GTP",
            "site#4_GDP",
            "site#new",
        ]
        system.add_species("site#remove", 0)
        for site_type in site_types:
            system.add_topology_species(site_type, diffCoeff)

    @staticmethod
    def do_grow1(topology, GTP_state):
        """
        start adding a tubulin dimer to the end of a protofilament:
        add additional particles
        """
        if parameters["verbose"]:
            print("Grow 1")
        recipe = readdy.StructuralReactionRecipe(topology)
        v_newB = ReaddyUtil.get_vertex_of_type(topology, "tubulinB#free", True)
        if v_newB is None:
            raise Exception(
                "Failed to find tubulinB#free vertex\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        v_newA = ReaddyUtil.get_neighbor_of_type(
            topology, v_newB, "tubulinA#free", True
        )
        if v_newA is None:
            raise Exception(
                "Failed to find tubulinA#free vertex\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        MicrotubulesUtil.add_tubulin_sites(topology, recipe, v_newB)
        MicrotubulesUtil.add_tubulin_sites(topology, recipe, v_newA)
        recipe.change_topology_type(f"Microtubule#Growing2-{GTP_state}")
        return recipe

    @staticmethod
    def reaction_function_grow1_GTP(topology):
        """
        start adding a tubulin dimer to the end of a protofilament:
        add additional particles
        """
        return MicrotubulesUtil.do_grow1(topology, "GTP")

    @staticmethod
    def reaction_function_grow1_GDP(topology):
        """
        start adding a tubulin dimer to the end of a protofilament:
        add additional particles
        """
        return MicrotubulesUtil.do_grow1(topology, "GDP")

    @staticmethod
    def do_grow2(topology):
        """
        finish adding a tubulin dimer to the end of a protofilament:
        set types, positions, and edges
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if parameters["verbose"]:
            print("Grow 2")
        v_newB = ReaddyUtil.get_vertex_of_type(topology, "tubulinB#free", True)
        if v_newB is None:
            raise Exception(
                "Failed to find tubulinB#free vertex\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        v_newA = ReaddyUtil.get_neighbor_of_type(
            topology, v_newB, "tubulinA#free", True
        )
        if v_newA is None:
            raise Exception(
                "Failed to find tubulinA#free vertex\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        v_site4 = ReaddyUtil.get_neighbor_of_type(topology, v_newA, "site#4", True)
        if v_site4 is None:
            raise Exception(
                "Failed to find site#4 vertex\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        v_endB = ReaddyUtil.get_neighbor_of_type(topology, v_site4, "tubulinB", False)
        if v_endB is None:
            raise Exception(
                "Failed to find neighboring tubulin vertex\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        v_sites = MicrotubulesUtil.get_tubulin_sites(topology, v_endB)
        if v_sites is None:
            raise Exception(
                "TubulinB at end does not have sites!\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        v_new_sitesB = ReaddyUtil.get_neighbors_of_type(
            topology, v_newB, "site#new", True
        )
        if len(v_new_sitesB) != 5:
            raise Exception(
                f"Found {len(v_new_sitesB)} new particles on tubulinB, expected 5\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        v_new_sitesA = ReaddyUtil.get_neighbors_of_type(
            topology, v_newA, "site#new", True
        )
        if len(v_new_sitesA) != 5:
            raise Exception(
                f"Found {len(v_new_sitesA)} new particles on tubulinA, expected 5\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        pos_endB = ReaddyUtil.get_vertex_position(topology, v_endB)
        pos_site0 = ReaddyUtil.get_vertex_position(topology, v_sites[0])
        pos_site1 = ReaddyUtil.get_vertex_position(topology, v_sites[1])
        side = ReaddyUtil.normalize(pos_site1 - pos_endB)
        normal = ReaddyUtil.normalize(pos_site0 - pos_endB)
        tangent = np.cross(normal, side)
        # remove temporary edge (site#4--tubulinA#free)]
        removed, message = ReaddyUtil.try_remove_edge(
            topology, recipe, v_sites[4], v_newA
        )
        if not removed:
            raise Exception(message + "\n" + ReaddyUtil.topology_to_string(topology))
        recipe.add_edge(v_endB, v_newA)  # endB -- newA
        # new tubulinA
        tangent = ReaddyUtil.rotate(np.copy(tangent), side, np.deg2rad(10.0))
        normal = ReaddyUtil.rotate(np.copy(normal), side, np.deg2rad(10.0))
        pos_newA = pos_endB + 4.0 * tangent
        polymer_indicesA = MicrotubulesUtil.increment_polymer_indices(
            MicrotubulesUtil.get_polymer_indices(
                topology.particle_type_of_vertex(v_endB)
            ),
            [1, 0],
        )
        recipe.change_particle_type(
            v_newA,
            "tubulinA#GTP_bent"
            + MicrotubulesUtil.polymer_indices_to_string(polymer_indicesA),
        )
        recipe.change_particle_position(v_newA, pos_newA)
        MicrotubulesUtil.setup_sites(
            topology, recipe, v_new_sitesA, pos_newA, side, normal, tangent, "_GTP", ""
        )
        MicrotubulesUtil.connect_sites_between_tubulins(
            topology, recipe, v_sites, v_new_sitesA
        )
        # new tubulinB
        tangent = ReaddyUtil.rotate(np.copy(tangent), side, np.deg2rad(10.0))
        normal = ReaddyUtil.rotate(np.copy(normal), side, np.deg2rad(10.0))
        pos_newB = pos_newA + 4.0 * tangent
        polymer_indicesB = MicrotubulesUtil.increment_polymer_indices(
            polymer_indicesA, [1, 0]
        )
        recipe.change_particle_type(
            v_newB,
            "tubulinB#GTP_bent"
            + MicrotubulesUtil.polymer_indices_to_string(polymer_indicesB),
        )
        recipe.change_particle_position(v_newB, pos_newB)
        MicrotubulesUtil.setup_sites(
            topology,
            recipe,
            v_new_sitesB,
            pos_newB,
            side,
            normal,
            tangent,
            "_GTP",
            "_GTP",
        )
        MicrotubulesUtil.connect_sites_between_tubulins(
            topology, recipe, v_new_sitesA, v_new_sitesB
        )
        recipe.change_topology_type("Microtubule")
        return recipe

    @staticmethod
    def reaction_function_grow2_GTP(topology):
        """
        finish adding a tubulin dimer to the end of a protofilament:
        set types, positions, and edges
        """
        return MicrotubulesUtil.do_grow2(topology)

    @staticmethod
    def reaction_function_grow2_GDP(topology):
        """
        finish adding a tubulin dimer to the end of a protofilament:
        set types, positions, and edges
        """
        return MicrotubulesUtil.do_grow2(topology)

    @staticmethod
    def do_shrink1(topology, GTP_state):
        """
        start removing a tubulin dimer from the end of a protofilament:
        remove or detach particles, change particle types
        """
        if parameters["verbose"]:
            print("Shrink")
        recipe = readdy.StructuralReactionRecipe(topology)
        tubulins = MicrotubulesUtil.get_random_tubulin_neighbors(
            topology, [["B#", "bent"], ["A#", "bent"]], [[], []], GTP_state, [1, 0]
        )
        if tubulins is None:
            recipe.change_topology_type(f"{topology.type}#Fail-Shrink-{GTP_state}")
            if parameters["verbose"]:
                print(
                    "Shrink cancelled: Couldn't find "
                    "2 bent tubulin vertices to separate"
                )
            return recipe
        # are both fragments crosslinked to other filaments?
        if MicrotubulesUtil.filament_is_crosslinked(
            topology, tubulins[0], -1
        ) and MicrotubulesUtil.filament_is_crosslinked(topology, tubulins[1], 1):
            recipe.change_topology_type(f"{topology.type}#Fail-Shrink-{GTP_state}")
            if parameters["verbose"]:
                print(
                    "Shrink cancelled: both fragments (starting at "
                    + ReaddyUtil.vertex_to_string(topology, tubulins[0])
                    + " and "
                    + ReaddyUtil.vertex_to_string(topology, tubulins[1])
                    + ") are crosslinked"
                )
            return recipe
        # will either of the cut fragments be a dimer?
        is_dimer = []
        filament_lengths = MicrotubulesUtil.get_filament_lengths(
            topology, tubulins[0], tubulins[1]
        )
        for t in range(len(tubulins)):
            is_dimer.append(filament_lengths[t] < 3)
        if not is_dimer[0]:
            site4 = ReaddyUtil.get_neighbor_of_type(
                topology, tubulins[0], "site#4", False
            )
            if site4 is None:
                raise Exception(
                    "Shrink cancelled: Couldn't find site4 to set reactive\n"
                    + ReaddyUtil.topology_to_string(topology)
                )
        # if both fragments are bigger than dimers, just disconnect them
        if True not in is_dimer:
            MicrotubulesUtil.remove_bent_filament_site_bonds(
                topology, recipe, tubulins[0], tubulins[1]
            )
        # detach at least one dimer
        else:
            tubulins_to_detach = []
            for i in range(len(tubulins)):
                if not is_dimer[i]:
                    continue
                tubulins_to_detach.append(tubulins[i])
                tubulins_to_detach.append(
                    MicrotubulesUtil.get_neighboring_tubulin(
                        topology, tubulins[i], [-1 if i == 0 else 1, 0]
                    )
                )
            if not MicrotubulesUtil.set_free(topology, recipe, tubulins_to_detach):
                raise Exception(
                    "Failed to find sites for tubulins being released\n"
                    + ReaddyUtil.topology_to_string(topology)
                )
        removed, message = ReaddyUtil.try_remove_edge(
            topology, recipe, tubulins[0], tubulins[1]
        )
        if not removed:
            raise Exception(message + "\n" + ReaddyUtil.topology_to_string(topology))
        # make the new plus end reactive
        if not is_dimer[0]:
            recipe.change_particle_type(site4, f"site#4_{GTP_state}")
        # remove sites from the new minus end of the second fragment
        # if it's not a dimer and it and its plus neighbor
        # are both fully attached laterally
        if not is_dimer[1]:
            MicrotubulesUtil.check_remove_tubulin_sites(topology, recipe, tubulins[1])
        recipe.change_topology_type("Microtubule#Shrinking")
        return recipe

    @staticmethod
    def reaction_function_shrink_GTP(topology):
        """
        start removing a tubulin dimer from the end of a protofilament:
        remove or detach particles, change particle types
        """
        return MicrotubulesUtil.do_shrink1(topology, "GTP")

    @staticmethod
    def reaction_function_shrink_GDP(topology):
        """
        start removing a tubulin dimer from the end of a protofilament:
        remove or detach particles, change particle types
        """
        return MicrotubulesUtil.do_shrink1(topology, "GDP")

    @staticmethod
    def reaction_function_shrink2(topology):
        """
        finish removing a tubulin dimer from the end of a protofilament:
        change topology types
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        if len(topology.graph.get_vertices()) == 2:
            recipe.change_topology_type("Dimer")
        elif MicrotubulesUtil.topology_is_microtubule(topology):
            recipe.change_topology_type("Microtubule")
        else:
            recipe.change_topology_type("Oligomer")
        return recipe

    @staticmethod
    def reaction_function_attach(topology):
        """
        attach tubulins laterally
        """
        if parameters["verbose"]:
            print("Attach")
        recipe = readdy.StructuralReactionRecipe(topology)
        attaching_sites = MicrotubulesUtil.get_attaching_sites(topology)
        if len(attaching_sites) == 0:
            raise Exception(
                "Failed to find attaching vertices\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        tubulins = []
        for site in attaching_sites:
            tubulins.append(
                ReaddyUtil.get_neighbor_of_type(topology, site, "tubulin", False)
            )
            if tubulins[len(tubulins) - 1] is None:
                raise Exception(
                    "Attach cancelled: failed to find attaching site's "
                    f"({ReaddyUtil.vertex_to_string(topology, site)}) tubulin\n"
                    + ReaddyUtil.topology_to_string(topology)
                )
        # make sure sites don't belong to same tubulin
        tubulin_ids = [
            topology.particle_id_of_vertex(tubulins[0]),
            topology.particle_id_of_vertex(tubulins[1]),
        ]
        if tubulin_ids[0] == tubulin_ids[1]:
            MicrotubulesUtil.cancel_attach(topology, recipe, attaching_sites)
            if parameters["verbose"]:
                print(
                    "Attach cancelled: sites ("
                    + ReaddyUtil.vertex_to_string(topology, attaching_sites[0])
                    + ", "
                    + ReaddyUtil.vertex_to_string(topology, attaching_sites[1])
                    + ") were on same tubulin ("
                    + ReaddyUtil.vertex_to_string(topology, tubulins[0])
                )
            return recipe
        # make sure sites belong to tubulins that can attach
        tubulin_types = [
            topology.particle_type_of_vertex(tubulins[0]),
            topology.particle_type_of_vertex(tubulins[1]),
        ]
        if not MicrotubulesUtil.tubulins_can_attach(tubulin_types):
            MicrotubulesUtil.cancel_attach(topology, recipe, attaching_sites)
            if parameters["verbose"]:
                print(
                    "Attach cancelled: tubulins ("
                    + ReaddyUtil.vertex_to_string(topology, tubulins[0])
                    + ", "
                    + ReaddyUtil.vertex_to_string(topology, tubulins[1])
                    + ") can't attach"
                )
            return recipe
        for i in range(2):
            crosslinked = (
                MicrotubulesUtil.get_neighboring_tubulin(
                    topology, tubulins[i], [0, 1 if i == 0 else -1]
                )
                is not None
            )
            if crosslinked:
                MicrotubulesUtil.check_remove_tubulin_sites(
                    topology, recipe, tubulins[i], True
                )
            else:
                ReaddyUtil.set_flags(topology, recipe, tubulins[i], [], ["bent"])
            prev_tubulin = MicrotubulesUtil.get_neighboring_tubulin(
                topology, tubulins[i], [-1, 0]
            )
            if prev_tubulin is not None:
                MicrotubulesUtil.check_remove_tubulin_sites(
                    topology, recipe, prev_tubulin, False, tubulins[i], crosslinked
                )
            next_tubulin = MicrotubulesUtil.get_neighboring_tubulin(
                topology, tubulins[i], [1, 0]
            )
            if next_tubulin is not None:
                MicrotubulesUtil.check_remove_tubulin_sites(
                    topology, recipe, next_tubulin, False, tubulins[i], crosslinked
                )
        removed, message = ReaddyUtil.try_remove_edge(
            topology, recipe, attaching_sites[0], attaching_sites[1]
        )
        if not removed:
            raise Exception(message + "\n" + ReaddyUtil.topology_to_string(topology))
        recipe.add_edge(tubulins[0], tubulins[1])
        recipe.change_topology_type("Microtubule")
        return recipe

    @staticmethod
    def reaction_function_detach1(topology):
        """
        add new sites in preparation to detach tubulins laterally
        """
        if parameters["verbose"]:
            print("Detach")
        recipe = readdy.StructuralReactionRecipe(topology)
        GTP_state = (
            "GTP"
            if (
                random.random()
                <= parameters["ring_detach_GTP_rate"]
                / (
                    parameters["ring_detach_GTP_rate"]
                    + parameters["ring_detach_GDP_rate"]
                )
            )
            else "GDP"
        )
        detaching_tubulins = MicrotubulesUtil.get_random_tubulin_neighbors(
            topology,
            [["tubulin"], ["tubulin"]],
            [["bent"], ["bent"]],
            GTP_state,
            [0, -1],
        )
        if detaching_tubulins is None:
            if parameters["verbose"]:
                print("Detach cancelled: Couldn't find 2 tubulin vertices to detach")
            return recipe
        for i in range(2):
            MicrotubulesUtil.check_add_tubulin_sites(
                topology,
                recipe,
                detaching_tubulins[i],
                "site#new" if i == 0 else "site#1_detach",
                "site#2_detach" if i == 0 else "site#new",
            )
            if (
                MicrotubulesUtil.get_neighboring_tubulin(
                    topology, detaching_tubulins[i], [0, 1 if i == 0 else -1]
                )
                is None
            ):
                ReaddyUtil.set_flags(
                    topology, recipe, detaching_tubulins[i], ["bent"], []
                )
            for j in range(2):
                neighbor_tubulin = MicrotubulesUtil.get_neighboring_tubulin(
                    topology, detaching_tubulins[i], [-1 if j == 0 else 1, 0]
                )
                if neighbor_tubulin is not None:
                    MicrotubulesUtil.check_add_tubulin_sites(
                        topology, recipe, neighbor_tubulin
                    )
        recipe.change_topology_type(f"Microtubule#Detaching-{GTP_state}")
        return recipe

    @staticmethod
    def reaction_function_detach2(topology):
        """
        detach tubulins laterally
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        detaching_sites = [
            ReaddyUtil.get_vertex_of_type(topology, "site#2_detach", False),
            ReaddyUtil.get_vertex_of_type(topology, "site#1_detach", False),
        ]
        if None in detaching_sites:
            site0 = (
                "None"
                if detaching_sites[0] is None
                else ReaddyUtil.vertex_to_string(detaching_sites[0])
            )
            site1 = (
                "None"
                if detaching_sites[1] is None
                else ReaddyUtil.vertex_to_string(detaching_sites[1])
            )
            raise Exception(
                f"Failed to find detaching sites, ({site0}, {site1})\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        detaching_tubulins = [
            ReaddyUtil.get_neighbor_of_type(
                topology, detaching_sites[0], "tubulin", False
            ),
            ReaddyUtil.get_neighbor_of_type(
                topology, detaching_sites[1], "tubulin", False
            ),
        ]
        if None in detaching_tubulins:
            tub0 = (
                "None"
                if detaching_tubulins[0] is None
                else ReaddyUtil.vertex_to_string(detaching_tubulins[0])
            )
            tub1 = (
                "None"
                if detaching_tubulins[1] is None
                else ReaddyUtil.vertex_to_string(detaching_tubulins[1])
            )
            raise Exception(
                f"Failed to find detaching tubulins ({tub0}, {tub1})\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        for i in range(2):
            tubulins = [
                MicrotubulesUtil.get_neighboring_tubulin(
                    topology, detaching_tubulins[i], [-1, 0]
                ),
                detaching_tubulins[i],
                MicrotubulesUtil.get_neighboring_tubulin(
                    topology, detaching_tubulins[i], [1, 0]
                ),
            ]
            if tubulins[0] is None and tubulins[2] is None:
                raise Exception(
                    "Tubulin has no neighbors on filament?!\n"
                    + ReaddyUtil.topology_to_string(topology)
                )
            pos_tubulin = ReaddyUtil.get_vertex_position(topology, tubulins[1])
            pos_filament = ReaddyUtil.get_vertex_position(
                topology, tubulins[0] if tubulins[0] is not None else tubulins[2]
            )
            pos_ring = ReaddyUtil.get_vertex_position(
                topology, detaching_tubulins[1 if i == 0 else 0]
            )
            frayed_angle = np.deg2rad(10.0)
            tangent = (1 if tubulins[0] is not None else -1) * ReaddyUtil.normalize(
                pos_tubulin - pos_filament
            )
            side = (1 if i == 0 else -1) * ReaddyUtil.normalize(pos_ring - pos_tubulin)
            normal = -np.cross(np.copy(tangent), np.copy(side))
            sites = 3 * [None]
            added_sites = 3 * [False]
            for j in range(3):
                tangent = ReaddyUtil.rotate(
                    np.copy(tangent), side, frayed_angle * (-1 if j > 0 else 0.5)
                )
                normal = ReaddyUtil.rotate(
                    np.copy(normal), side, frayed_angle * (-1 if j > 0 else 0.5)
                )
                if tubulins[j] is not None:
                    sites[j] = MicrotubulesUtil.get_tubulin_sites(topology, tubulins[j])
                    tubulin_state = (
                        "_GTP"
                        if "GTP" in topology.particle_type_of_vertex(tubulins[j])
                        else "_GDP"
                    )
                    site_state_ring = (
                        tubulin_state
                        if "bent" in topology.particle_type_of_vertex(tubulins[j])
                        else ""
                    )
                    if sites[j] is None:
                        sites[j] = MicrotubulesUtil.get_new_sites(topology, tubulins[j])
                        if sites[j] is None:
                            raise Exception(
                                ReaddyUtil.vertex_to_string(topology, tubulins[j])
                                + f" is missing sites [{i}]\n"
                                + ReaddyUtil.topology_to_string(topology)
                            )
                        added_sites[j] = True
                        site_state_filament = (
                            tubulin_state
                            if MicrotubulesUtil.get_neighboring_tubulin(
                                topology, tubulins[j], [1, 0]
                            )
                            is None
                            else ""
                        )
                        MicrotubulesUtil.setup_sites(
                            topology,
                            recipe,
                            sites[j],
                            ReaddyUtil.get_vertex_position(topology, tubulins[j]),
                            side,
                            normal,
                            tangent,
                            site_state_ring,
                            site_state_filament,
                        )
                    else:
                        if sites[j][1] is None or sites[j][2] is None:
                            raise Exception(
                                "A ring site was None!\n"
                                + ReaddyUtil.topology_to_string(topology)
                            )
                        recipe.change_particle_type(
                            sites[j][1], f"site#1{site_state_ring}"
                        )
                        recipe.change_particle_type(
                            sites[j][2], f"site#2{site_state_ring}"
                        )
                if (
                    j > 0
                    and (sites[j - 1] is not None and sites[j] is not None)
                    and (added_sites[j - 1] or added_sites[j])
                ):
                    MicrotubulesUtil.connect_sites_between_tubulins(
                        topology, recipe, sites[j - 1], sites[j]
                    )
        removed, message = ReaddyUtil.try_remove_edge(
            topology, recipe, detaching_tubulins[0], detaching_tubulins[1]
        )
        if not removed:
            raise Exception(message + "\n" + ReaddyUtil.topology_to_string(topology))
        recipe.change_topology_type("Microtubule")
        return recipe

    @staticmethod
    def reaction_function_hydrolyze(topology):
        """
        hydrolyze GTP to GDP in a random tubulin
        """
        if parameters["verbose"]:
            print("Hydrolyze")
        recipe = readdy.StructuralReactionRecipe(topology)
        tubulin = ReaddyUtil.get_random_vertex_of_type(topology, "#GTP", False)
        if tubulin is None:
            if parameters["verbose"]:
                print("Hydrolyze cancelled: Couldn't find GTP-tubulin")
            return recipe
        sites = MicrotubulesUtil.get_tubulin_sites(topology, tubulin)
        if sites is not None:
            for s in range(1, 4):
                if "GTP" in topology.particle_type_of_vertex(sites[s]):
                    ReaddyUtil.set_flags(topology, recipe, sites[s], ["GDP"], ["GTP"])
        ReaddyUtil.set_flags(topology, recipe, tubulin, ["GDP"], ["GTP"])
        return recipe

    @staticmethod
    def rate_function_shrink_GTP(topology):
        """
        rate function for removing a GTP-tubulin dimer from the end of a protofilament
        """
        return parameters["protofilament_shrink_GTP_rate"]

    @staticmethod
    def rate_function_shrink_GDP(topology):
        """
        rate function for removing a GDP-tubulin dimer from the end of a protofilament
        """
        return parameters["protofilament_shrink_GDP_rate"]

    @staticmethod
    def rate_function_detach_ring(topology):
        """
        rate function for detaching protofilaments laterally
        at ring sites for GTP-tubulin
        """
        return parameters["ring_detach_GTP_rate"] + parameters["ring_detach_GDP_rate"]

    @staticmethod
    def rate_function_hydrolyze(topology):
        """
        rate function for hydrolyzing GTP to GDP in tubulin Bs
        """
        return parameters["hydrolyze_rate"]

    @staticmethod
    def add_bonds_between_tubulins(tubulin_types, force_constant, system, util):
        """
        add bonds between tubulins
        """
        util.add_polymer_bond_2D(  # bonds between protofilaments
            tubulin_types, [0, 0], tubulin_types, [0, -1], force_constant, 5.2, system
        )
        util.add_polymer_bond_2D(  # bonds between rings
            tubulin_types, [0, 0], tubulin_types, [-1, 0], force_constant, 4.0, system
        )
        util.add_bond(  # dimer bond and temporary bond
            ["tubulinB#free"], ["tubulinA#free"], force_constant, 4.0, system
        )

    @staticmethod
    def add_tubulin_site_bonds(tubulin_types, site_types, force_constant, system, util):
        """
        add bonds between a tubulin and its sites
        """
        util.add_polymer_bond_2D(
            tubulin_types, [0, 0], site_types, [], force_constant, 1.5, system
        )
        util.add_bond(
            ["tubulinA#free", "tubulinB#free"], site_types, force_constant, 1.5, system
        )
        util.add_bond(
            ["site#out"],
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#1_detach",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
                "site#2_detach",
            ],
            force_constant,
            2.12,
            system,
        )
        util.add_bond(
            ["site#out"],
            ["site#3", "site#4", "site#4_GTP", "site#4_GDP"],
            force_constant,
            2.12,
            system,
        )
        util.add_bond(
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#1_detach",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
                "site#2_detach",
            ],
            ["site#3", "site#4", "site#4_GTP", "site#4_GDP"],
            force_constant,
            2.12,
            system,
        )
        util.add_bond(  # temporary (grow reactions)
            ["tubulinA#free"], ["site#4", "site#4_GTP", "site#4_GDP"], 1.0, 2.5, system
        )
        util.add_bond(  # temporary (attach reactions)
            ["site#1", "site#1_GTP", "site#1_GDP", "site#1_detach"],
            ["site#2", "site#2_GTP", "site#2_GDP", "site#2_detach"],
            1.0,
            2.5,
            system,
        )
        util.add_bond(  # temporary (detach reactions)
            ["site#new"], site_types, 1.0, 4.0, system
        )

    @staticmethod
    def add_bent_site_bonds(force_constant, system, util):
        """
        add bonds between sites on bent tubulins
        """
        util.add_bond(["site#out"], ["site#out"], force_constant, 3.73, system)
        util.add_bond(
            ["site#1", "site#1_GTP", "site#1_GDP", "site#1_detach"],
            ["site#1", "site#1_GTP", "site#1_GDP", "site#1_detach"],
            force_constant,
            4.0,
            system,
        )
        util.add_bond(
            ["site#2", "site#2_GTP", "site#2_GDP", "site#2_detach"],
            ["site#2", "site#2_GTP", "site#2_GDP", "site#2_detach"],
            force_constant,
            4.0,
            system,
        )
        util.add_bond(
            ["site#3"],
            ["site#4", "site#4_GTP", "site#4_GDP"],
            force_constant,
            1.0,
            system,
        )

    @staticmethod
    def add_angles_between_tubulins(tubulin_type_sets, force_constant, system, util):
        """
        add angles between tubulins
        """
        util.add_polymer_angle_2D(
            tubulin_type_sets[2],
            [0, 1],
            tubulin_type_sets[2],
            [0, 0],
            tubulin_type_sets[2],
            [-1, 0],
            force_constant,
            1.75,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_type_sets[2],
            [0, 1],
            tubulin_type_sets[2],
            [0, 0],
            tubulin_type_sets[2],
            [1, 0],
            force_constant,
            1.40,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_type_sets[2],
            [0, -1],
            tubulin_type_sets[2],
            [0, 0],
            tubulin_type_sets[2],
            [-1, 0],
            force_constant,
            1.40,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_type_sets[2],
            [0, -1],
            tubulin_type_sets[2],
            [0, 0],
            tubulin_type_sets[2],
            [1, 0],
            force_constant,
            1.75,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_type_sets[0],
            [-1, 0],
            tubulin_type_sets[0],
            [0, 0],
            tubulin_type_sets[0],
            [1, 0],
            force_constant,
            np.pi,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_type_sets[0],
            [-1, 0],
            tubulin_type_sets[0],
            [0, 0],
            tubulin_type_sets[1],
            [1, 0],
            force_constant,
            3.05,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_type_sets[0],
            [1, 0],
            tubulin_type_sets[0],
            [0, 0],
            tubulin_type_sets[1],
            [-1, 0],
            force_constant,
            3.05,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_type_sets[2],
            [-1, 0],
            tubulin_type_sets[1],
            [0, 0],
            tubulin_type_sets[2],
            [1, 0],
            force_constant,
            2.97,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_type_sets[2],
            [0, -1],
            tubulin_type_sets[2],
            [0, 0],
            tubulin_type_sets[2],
            [0, 1],
            force_constant,
            2.67,
            system,
        )

    @staticmethod
    def add_tubulin_site_angles(tubulin_types, force_constant, system, util):
        """
        add angles between a tubulin and its sites
        """
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, 0],
            ["site#out"],
            [],
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
                "site#3",
                "site#4",
                "site#4_GTP",
                "site#4_GDP",
            ],
            [],
            force_constant,
            0.79,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, 0],
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
                "site#3",
                "site#4",
                "site#4_GTP",
                "site#4_GDP",
            ],
            [],
            ["site#out"],
            [],
            force_constant,
            0.79,
            system,
        )
        util.add_polymer_angle_2D(
            ["site#out"],
            [],
            tubulin_types,
            [0, 0],
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
                "site#3",
                "site#4",
                "site#4_GTP",
                "site#4_GDP",
            ],
            [],
            force_constant,
            np.pi / 2,
            system,
        )
        util.add_angle(
            ["site#out"],
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
            ],
            ["site#3", "site#4", "site#4_GTP", "site#4_GDP"],
            force_constant,
            1.05,
            system,
        )
        util.add_angle(
            ["site#out"],
            ["site#3", "site#4", "site#4_GTP", "site#4_GDP"],
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
            ],
            force_constant,
            1.05,
            system,
        )
        util.add_angle(
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
            ],
            ["site#out"],
            ["site#3", "site#4", "site#4_GTP", "site#4_GDP"],
            force_constant,
            1.05,
            system,
        )
        util.add_polymer_angle_2D(
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
            ],
            [],
            tubulin_types,
            [0, 0],
            ["site#3", "site#4", "site#4_GTP", "site#4_GDP"],
            [],
            force_constant,
            np.pi / 2,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, 0],
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
            ],
            [],
            ["site#3", "site#4", "site#4_GTP", "site#4_GDP"],
            [],
            force_constant,
            0.79,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, 0],
            ["site#3", "site#4", "site#4_GTP", "site#4_GDP"],
            [],
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
            ],
            [],
            force_constant,
            0.79,
            system,
        )
        util.add_angle(
            ["site#1", "site#1_GTP", "site#1_GDP"],
            ["site#out"],
            ["site#2", "site#2_GTP", "site#2_GDP"],
            force_constant,
            np.pi / 2,
            system,
        )
        util.add_angle(
            ["site#3"],
            ["site#out"],
            ["site#4", "site#4_GTP", "site#4_GDP"],
            force_constant,
            np.pi / 2,
            system,
        )
        util.add_polymer_angle_2D(
            ["site#1", "site#1_GTP", "site#1_GDP"],
            [],
            tubulin_types,
            [0, 0],
            ["site#2", "site#2_GTP", "site#2_GDP"],
            [],
            force_constant,
            np.pi,
            system,
        )
        util.add_polymer_angle_2D(
            ["site#3"],
            [],
            tubulin_types,
            [0, 0],
            ["site#4", "site#4_GTP", "site#4_GDP"],
            [],
            force_constant,
            np.pi,
            system,
        )
        util.add_angle(
            ["site#3"],
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
            ],
            ["site#4", "site#4_GTP", "site#4_GDP"],
            force_constant,
            np.pi / 2,
            system,
        )
        util.add_angle(
            ["site#1", "site#1_GTP", "site#1_GDP"],
            ["site#3", "site#4", "site#4_GTP", "site#4_GDP"],
            ["site#2", "site#2_GTP", "site#2_GDP"],
            force_constant,
            np.pi / 2,
            system,
        )

    @staticmethod
    def add_bent_site_angles(tubulin_types, force_constant, system, util):
        """
        add angles between sites on bent tubulins
        """
        util.add_angle(
            ["site#1", "site#1_GTP", "site#1_GDP"],
            ["site#1", "site#1_GTP", "site#1_GDP"],
            ["site#out"],
            force_constant,
            1.51,
            system,
        )
        util.add_angle(
            ["site#2", "site#2_GTP", "site#2_GDP"],
            ["site#2", "site#2_GTP", "site#2_GDP"],
            ["site#out"],
            force_constant,
            1.51,
            system,
        )
        util.add_angle(
            ["site#1", "site#1_GTP", "site#1_GDP"],
            ["site#out"],
            ["site#out"],
            force_constant,
            1.63,
            system,
        )
        util.add_angle(
            ["site#2", "site#2_GTP", "site#2_GDP"],
            ["site#out"],
            ["site#out"],
            force_constant,
            1.63,
            system,
        )
        util.add_polymer_angle_2D(
            ["site#out"],
            [],
            ["site#out"],
            [],
            tubulin_types,
            [0, 0],
            force_constant,
            1.66,
            system,
        )
        util.add_polymer_angle_2D(
            ["site#1", "site#1_GTP", "site#1_GDP"],
            [],
            ["site#1", "site#1_GTP", "site#1_GDP"],
            [],
            tubulin_types,
            [0, 0],
            force_constant,
            np.pi / 2,
            system,
        )
        util.add_polymer_angle_2D(
            ["site#2", "site#2_GTP", "site#2_GDP"],
            [],
            ["site#2", "site#2_GTP", "site#2_GDP"],
            [],
            tubulin_types,
            [0, 0],
            force_constant,
            np.pi / 2,
            system,
        )
        util.add_polymer_angle_2D(
            ["site#4", "site#4_GTP", "site#4_GDP"],
            [],
            ["site#3"],
            [],
            tubulin_types,
            [0, 0],
            force_constant,
            3.05,
            system,
        )
        util.add_polymer_angle_2D(
            ["site#3"],
            [],
            ["site#4", "site#4_GTP", "site#4_GDP"],
            [],
            tubulin_types,
            [0, 0],
            force_constant,
            3.05,
            system,
        )
        util.add_angle(
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
            ],
            ["site#4", "site#4_GTP", "site#4_GDP"],
            ["site#3"],
            force_constant,
            2.35,
            system,
        )
        util.add_angle(
            [
                "site#1",
                "site#1_GTP",
                "site#1_GDP",
                "site#2",
                "site#2_GTP",
                "site#2_GDP",
            ],
            ["site#3"],
            ["site#4", "site#4_GTP", "site#4_GDP"],
            force_constant,
            2.35,
            system,
        )
        util.add_angle(
            ["site#out"], ["site#out"], ["site#out"], force_constant, 2.97, system
        )
        util.add_angle(
            ["site#1", "site#1_GTP", "site#1_GDP"],
            ["site#1", "site#1_GTP", "site#1_GDP"],
            ["site#1", "site#1_GTP", "site#1_GDP"],
            force_constant,
            2.97,
            system,
        )
        util.add_angle(
            ["site#2", "site#2_GTP", "site#2_GDP"],
            ["site#2", "site#2_GTP", "site#2_GDP"],
            ["site#2", "site#2_GTP", "site#2_GDP"],
            force_constant,
            2.97,
            system,
        )

    @staticmethod
    def add_edge_site_angles(tubulin_types, force_constant, system, util):
        """
        add angles between sites at the edge between tube and bent tubulins
        """
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, -1],
            tubulin_types,
            [0, 0],
            ["site#out"],
            [],
            force_constant,
            1.81,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, -1],
            tubulin_types,
            [0, 0],
            ["site#1", "site#1_GTP", "site#1_GDP"],
            [],
            force_constant,
            0.3,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, -1],
            tubulin_types,
            [0, 0],
            ["site#2", "site#2_GTP", "site#2_GDP"],
            [],
            force_constant,
            2.84,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, -1],
            tubulin_types,
            [0, 0],
            ["site#3"],
            [],
            force_constant,
            1.4,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, -1],
            tubulin_types,
            [0, 0],
            ["site#4", "site#4_GTP", "site#4_GDP"],
            [],
            force_constant,
            1.75,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [-1, 0],
            tubulin_types,
            [0, 0],
            ["site#out"],
            [],
            force_constant,
            np.pi / 2,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [-1, 0],
            tubulin_types,
            [0, 0],
            ["site#1", "site#1_GTP", "site#1_GDP"],
            [],
            force_constant,
            np.pi / 2,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [-1, 0],
            tubulin_types,
            [0, 0],
            ["site#2", "site#2_GTP", "site#2_GDP"],
            [],
            force_constant,
            np.pi / 2,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [-1, 0],
            tubulin_types,
            [0, 0],
            ["site#3"],
            [],
            force_constant,
            0.0,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [-1, 0],
            tubulin_types,
            [0, 0],
            ["site#4", "site#4_GTP", "site#4_GDP"],
            [],
            force_constant,
            np.pi,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, 1],
            tubulin_types,
            [0, 0],
            ["site#out"],
            [],
            force_constant,
            1.81,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, 1],
            tubulin_types,
            [0, 0],
            ["site#1", "site#1_GTP", "site#1_GDP"],
            [],
            force_constant,
            2.84,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, 1],
            tubulin_types,
            [0, 0],
            ["site#2", "site#2_GTP", "site#2_GDP"],
            [],
            force_constant,
            0.3,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, 1],
            tubulin_types,
            [0, 0],
            ["site#3"],
            [],
            force_constant,
            1.75,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, 1],
            tubulin_types,
            [0, 0],
            ["site#4", "site#4_GTP", "site#4_GDP"],
            [],
            force_constant,
            1.4,
            system,
        )

    @staticmethod
    def add_polymer_repulsion(particle_types, force_const, distance, system, util):
        """
        adds a pairwise repulsion between all polymer numbers
            of types particle_types
            with force constant force_const
            with equilibrium distance [nm]
        """
        types = []
        for t in particle_types:
            types += MicrotubulesUtil.get_all_polymer_tubulin_types(t)

        util.add_repulsion(types, types, force_const, distance, system)

    @staticmethod
    def add_growth_reaction(system, rate_GTP, rate_GDP, reaction_distance):
        """
        add dimers to the ends of protofilaments
        """
        system.topologies.add_spatial_reaction(
            "Start_Grow_GTP: Microtubule(site#4_GTP) + Dimer(tubulinA#free) -> \
            Microtubule#Growing1-GTP(site#4--tubulinA#free)",
            rate=rate_GTP,
            radius=0.5 + reaction_distance,
        )
        system.topologies.add_spatial_reaction(
            "Start_Grow_GDP: Microtubule(site#4_GDP) + Dimer(tubulinA#free) -> \
            Microtubule#Growing1-GDP(site#4--tubulinA#free)",
            rate=rate_GDP,
            radius=0.5 + reaction_distance,
        )
        system.topologies.add_structural_reaction(
            "Setup_Grow_GTP",
            topology_type="Microtubule#Growing1-GTP",
            reaction_function=MicrotubulesUtil.reaction_function_grow1_GTP,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Setup_Grow_GDP",
            topology_type="Microtubule#Growing1-GDP",
            reaction_function=MicrotubulesUtil.reaction_function_grow1_GDP,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Grow_GTP",
            topology_type="Microtubule#Growing2-GTP",
            reaction_function=MicrotubulesUtil.reaction_function_grow2_GTP,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Grow_GDP",
            topology_type="Microtubule#Growing2-GDP",
            reaction_function=MicrotubulesUtil.reaction_function_grow2_GDP,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_shrink_reaction(system):
        """
        separate dimers and oligomers from the ends
        of frayed protofilaments and oligomers
        """
        system.topologies.add_structural_reaction(
            "Shrink_MT_GTP",
            topology_type="Microtubule",
            reaction_function=MicrotubulesUtil.reaction_function_shrink_GTP,
            rate_function=MicrotubulesUtil.rate_function_shrink_GTP,
        )
        system.topologies.add_structural_reaction(
            "Shrink_MT_GDP",
            topology_type="Microtubule",
            reaction_function=MicrotubulesUtil.reaction_function_shrink_GDP,
            rate_function=MicrotubulesUtil.rate_function_shrink_GDP,
        )
        system.topologies.add_structural_reaction(
            "Shrink_Oligo_GTP",
            topology_type="Oligomer",
            reaction_function=MicrotubulesUtil.reaction_function_shrink_GTP,
            rate_function=MicrotubulesUtil.rate_function_shrink_GTP,
        )
        system.topologies.add_structural_reaction(
            "Shrink_Oligo_GDP",
            topology_type="Oligomer",
            reaction_function=MicrotubulesUtil.reaction_function_shrink_GDP,
            rate_function=MicrotubulesUtil.rate_function_shrink_GDP,
        )
        system.topologies.add_structural_reaction(
            "Finish_Shrink",
            topology_type="Microtubule#Shrinking",
            reaction_function=MicrotubulesUtil.reaction_function_shrink2,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Fail_Shrink_MT_GTP",
            topology_type="Microtubule#Fail-Shrink-GTP",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Fail_Shrink_MT_GDP",
            topology_type="Microtubule#Fail-Shrink-GDP",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Fail_Shrink_Oligo_GTP",
            topology_type="Oligomer#Fail-Shrink-GTP",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Fail_Shrink_Oligo_GDP",
            topology_type="Oligomer#Fail-Shrink-GDP",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_attach_reaction(system, rate_GTP, rate_GDP, reaction_distance):
        """
        attach protofilaments laterally
        """
        system.topologies.add_spatial_reaction(
            "Start_Attach_GTP1: Microtubule(site#1_GTP) + Microtubule(site#2_GTP) -> \
            Microtubule#Attaching(site#1--site#2) [self=true]",
            rate=rate_GTP,
            radius=1.0 + reaction_distance,
        )
        system.topologies.add_spatial_reaction(
            "Start_Attach_GTP2: Microtubule(site#1_GTP) + Microtubule(site#2_GDP) -> \
            Microtubule#Attaching(site#1--site#2) [self=true]",
            rate=rate_GTP,
            radius=1.0 + reaction_distance,
        )
        system.topologies.add_spatial_reaction(
            "Start_Attach_GTP3: Microtubule(site#1_GDP) + Microtubule(site#2_GTP) -> \
            Microtubule#Attaching(site#1--site#2) [self=true]",
            rate=rate_GTP,
            radius=1.0 + reaction_distance,
        )
        system.topologies.add_spatial_reaction(
            "Start_Attach_GDP: Microtubule(site#1_GDP) + Microtubule(site#2_GDP) -> \
            Microtubule#Attaching(site#1--site#2) [self=true]",
            rate=rate_GDP,
            radius=1.0 + reaction_distance,
        )
        system.topologies.add_structural_reaction(
            "Setup_Attach",
            topology_type="Microtubule#Attaching",
            reaction_function=MicrotubulesUtil.reaction_function_attach,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_detach_reaction(system):
        """
        detach protofilaments laterally
        """
        system.topologies.add_structural_reaction(
            "Start_Detach",
            topology_type="Microtubule",
            reaction_function=MicrotubulesUtil.reaction_function_detach1,
            rate_function=MicrotubulesUtil.rate_function_detach_ring,
        )
        system.topologies.add_structural_reaction(
            "Detach_GTP",
            topology_type="Microtubule#Detaching-GTP",
            reaction_function=MicrotubulesUtil.reaction_function_detach2,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
        system.topologies.add_structural_reaction(
            "Detach_GDP",
            topology_type="Microtubule#Detaching-GDP",
            reaction_function=MicrotubulesUtil.reaction_function_detach2,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_hydrolyze_reaction(system):
        """
        hydrolyze GTP-tubulinB to GDP-tubulinB
        """
        system.topologies.add_structural_reaction(
            "Hydrolyze",
            topology_type="Microtubule",
            reaction_function=MicrotubulesUtil.reaction_function_hydrolyze,
            rate_function=MicrotubulesUtil.rate_function_hydrolyze,
        )
        system.topologies.add_structural_reaction(
            "Fail_Hydrolyze",
            topology_type="Microtubule#Fail-Hydrolyze",
            reaction_function=ReaddyUtil.reaction_function_reset_state,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
