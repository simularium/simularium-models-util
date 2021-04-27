#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import readdy
import random

from ..common import ReaddyUtil
from ..microtubules.microtubules_util import MicrotubulesUtil


parameters = {}


def set_parameters(p):
    global parameters
    parameters = p
    return p


class KinesinUtil:
    def __init__(self, parameters):
        """
        Utility functions for ReaDDy kinesin models

        Parameters need to be accessible in ReaDDy callbacks
        which can't be instance methods, so parameters are global
        """
        set_parameters(parameters)

    @staticmethod
    def add_kinesin(position_offset, simulation):
        """
        add a kinesin to the simulation
        """
        positions = np.array(
            [
                [0.0, 3.0, 0.0],
                [0.0, 3.0, -5.0],
                [0.0, 0.0, 4.0],
                # [0., 30., 0.]
            ]
        )
        types = [
            "hips",
            "motor#ADP",
            "motor#ADP",
            # "cargo"
        ]
        kinesin = simulation.add_topology("Kinesin", types, positions + position_offset)
        for i in range(1, 3):
            kinesin.get_graph().add_edge(0, i)

    @staticmethod
    def set_kinesin_state(topology, recipe, from_motor_state, to_motor_state):
        """
        change the state of a motor and update the kinesin state to match
            for a dictionary of types and radii [nm]

            returns dictionary mapping all types to radii
        """
        motors = ReaddyUtil.get_vertices_of_type(topology, "motor", False)
        if len(motors) < 2:
            raise Exception(
                f"Failed to find 2 motors, found {len(motors)}\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        motor_types = [
            topology.particle_type_of_vertex(motors[0]),
            topology.particle_type_of_vertex(motors[1]),
        ]
        other_state = ""
        motors_in_from_state = []
        for i in range(2):
            if from_motor_state in motor_types[i]:
                motors_in_from_state.append(motors[i])
            else:
                other_state = motor_types[i][motor_types[i].index("#") + 1 :]
        if len(motors_in_from_state) < 1:
            if parameters["verbose"]:
                print(f"Couldn't find a motor in state {from_motor_state}")
            return None
        if len(motors_in_from_state) > 1:
            motor_to_set = random.choice(motors_in_from_state)
            other_state = from_motor_state
        else:
            motor_to_set = motors_in_from_state[0]
        recipe.change_particle_type(motor_to_set, f"motor#{to_motor_state}")
        if "ADP" in to_motor_state and "ADP" in other_state:
            recipe.change_topology_type("Microtubule-Kinesin#Releasing")
        else:
            new_states = [to_motor_state, other_state]
            new_states.sort()
            recipe.change_topology_type(
                f"Microtubule-Kinesin#{new_states[0]}-{new_states[1]}"
            )
        return motor_to_set

    @staticmethod
    def reaction_function_motor_bind_tubulin(topology):
        """
        bind a kinesin motor in ADP state to a free tubulinB
        """
        if parameters["verbose"]:
            print("Bind tubulin")
        recipe = readdy.StructuralReactionRecipe(topology)
        motor = KinesinUtil.set_kinesin_state(topology, recipe, "new", "apo")
        if motor is None:
            raise Exception(
                "Failed to find new motor\n" + ReaddyUtil.topology_to_string(topology)
            )
        tubulin = ReaddyUtil.get_neighbor_of_type(
            topology, motor, "tubulinB#bound", False
        )
        if tubulin is None:
            raise Exception(
                "Failed to find bound tubulin\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        if parameters["verbose"]:
            print(
                ReaddyUtil.vertex_to_string(topology, motor)
                + " ++ "
                + ReaddyUtil.vertex_to_string(topology, tubulin)
            )
        tubulin_pos = ReaddyUtil.get_vertex_position(topology, tubulin)
        # TODO calculate position offset from tubulin neighbors
        recipe.change_particle_position(motor, tubulin_pos + (0.0, 4.0, 0.0))
        return recipe

    @staticmethod
    def reaction_function_motor_bind_ATP(topology):
        """
        set bound apo motor's state to ATP (and implicitly simulate ATP binding)
        """
        if parameters["verbose"]:
            print("Bind ATP")
        recipe = readdy.StructuralReactionRecipe(topology)
        motor = KinesinUtil.set_kinesin_state(topology, recipe, "apo", "ATP")
        if motor is None:
            raise Exception(
                "Failed to find motor in apo state\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        if parameters["verbose"]:
            print(ReaddyUtil.vertex_to_string(topology, motor))
        return recipe

    @staticmethod
    def reaction_function_motor_release_tubulin(topology):
        """
        release a bound motor from tubulin
        """
        if parameters["verbose"]:
            print("Release tubulin")
        recipe = readdy.StructuralReactionRecipe(topology)
        motor = KinesinUtil.set_kinesin_state(topology, recipe, "ATP", "ADP")
        if motor is None:
            raise Exception(
                "Failed to find motor in ATP state\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        tubulin = ReaddyUtil.get_neighbor_of_type(
            topology, motor, "tubulinB#bound", False
        )
        if tubulin is None:
            raise Exception(
                "Failed to find bound tubulin\n"
                + ReaddyUtil.topology_to_string(topology)
            )
        if parameters["verbose"]:
            print(
                ReaddyUtil.vertex_to_string(topology, motor)
                + " -X- "
                + ReaddyUtil.vertex_to_string(topology, tubulin)
            )
        # ReaddyUtil.set_flags(topology, recipe, tubulin, [], ["bound"]) # TODO fix bug
        # workaround
        pt = topology.particle_type_of_vertex(tubulin)
        recipe.change_particle_type(tubulin, f"tubulinB#{pt[-3:]}")
        removed, message = ReaddyUtil.try_remove_edge(topology, recipe, motor, tubulin)
        if not removed:
            raise Exception(message + "\n" + ReaddyUtil.topology_to_string(topology))
        return recipe

    @staticmethod
    def reaction_function_cleanup_release_tubulin(topology):
        """
        cleanup after releasing a bound motor from tubulin
        """
        recipe = readdy.StructuralReactionRecipe(topology)
        motors = ReaddyUtil.get_vertices_of_type(topology, "motor", False)
        if len(motors) > 0:
            recipe.change_topology_type("Kinesin")
        else:
            recipe.change_topology_type("Microtubule")
        if parameters["verbose"]:
            print("Cleaned up release tubulin")
        return recipe

    @staticmethod
    def rate_function_motor_bind_ATP(topology):
        """
        rate function for a motor binding ATP
        """
        return parameters["motor_bind_ATP_rate"]

    @staticmethod
    def rate_function_motor_release_tubulin(topology):
        """
        rate function for a bound motor releasing from tubulin
        """
        return parameters["motor_release_tubulin_rate"]

    @staticmethod
    def add_kinesin_bonds_and_repulsions(motor_types, force_constant, system, util):
        """
        add bonds between tubulins
        """
        necklinker_force_constant = 0.002 * force_constant
        util.add_bond(["hips"], motor_types, necklinker_force_constant, 2.0, system)
        util.add_bond(["hips"], ["cargo"], force_constant, 30.0, system)
        util.add_repulsion(motor_types, motor_types, force_constant, 4.0, system)
        util.add_repulsion(["hips"], motor_types, force_constant, 2.0, system)
        util.add_repulsion(["hips"], ["cargo"], force_constant, 30.0, system)

    @staticmethod
    def add_tubulin_bonds_and_repulsions(tubulin_types, force_constant, system, util):
        """
        add bonds between tubulins
        """
        util.add_polymer_bond_2D(  # bonds between protofilaments
            tubulin_types, [0, 0], tubulin_types, [0, -1], force_constant, 5.2, system
        )
        util.add_polymer_bond_2D(  # bonds between rings
            tubulin_types, [0, 0], tubulin_types, [-1, 0], force_constant, 4.0, system
        )
        all_tubulin_types = []
        for t in range(len(tubulin_types)):
            all_tubulin_types += MicrotubulesUtil.get_all_polymer_tubulin_types(
                tubulin_types[t]
            )
        util.add_repulsion(
            all_tubulin_types, all_tubulin_types, force_constant, 4.0, system
        )

    @staticmethod
    def add_kinesin_angles_and_dihedrals(tubulin_types, force_constant, system, util):
        """
        add kinesin angles
        """
        # angles from tubulins to bound motor
        util.add_polymer_angle_2D(
            tubulin_types,
            [-1, 0],
            ["tubulinB#bound_"],
            [0, 0],
            ["motor#apo", "motor#ATP", "motor#ADP"],
            [],
            1e32,
            0.0,
            system,
        )
        # util.add_polymer_angle_2D(
        #     tubulin_types, [1, 0],
        #     ["tubulinB#bound_"], [0, 0],
        #     ["motor#apo", "motor#ATP", "motor#ADP"], [],
        #     1e32, np.pi / 2., system
        # )
        # util.add_polymer_angle_2D(
        #     tubulin_types, [0, -1],
        #     ["tubulinB#bound_"], [0, 0],
        #     ["motor#apo", "motor#ATP", "motor#ADP"], [],
        #     1e32, 1.84, system
        # )
        # util.add_polymer_angle_2D(
        #     tubulin_types, [0, 1],
        #     ["tubulinB#bound_"], [0, 0],
        #     ["motor#apo", "motor#ATP", "motor#ADP"], [],
        #     1e32, 1.54, system
        # )
        # # angle from bound tubulin to hips
        # util.add_polymer_angle_2D(
        #     ["tubulinB#bound_"], [0, 0],
        #     ["motor#ATP"], [],
        #     ["hips"], [],
        #     0.5 * force_constant, np.pi * 5./9., system
        # )
        # # angle from bound motor to free motor through hips
        # util.add_angle(
        #     ["motor#ATP"],
        #     ["hips"],
        #     ["motor#ADP"],
        #     0.1 * force_constant, np.pi * 8./9., system
        # )
        # # dihedrals from tubulins to hips
        # util.add_polymer_dihedral_2D(
        #     tubulin_types, [-1, 0],
        #     ["tubulinB#bound_"], [0, 0],
        #     ["motor#ATP"], [],
        #     ["hips"], [],
        #     1.5 * force_constant, np.pi * 17./18., system
        # )
        # util.add_polymer_dihedral_2D(
        #     tubulin_types, [1, 0],
        #     ["tubulinB#bound_"], [0, 0],
        #     ["motor#ATP"], [],
        #     ["hips"], [],
        #     1.5 * force_constant, np.pi / 18., system
        # )
        # util.add_polymer_dihedral_2D(
        #     tubulin_types, [0, -1],
        #     ["tubulinB#bound_"], [0, 0],
        #     ["motor#ATP"], [],
        #     ["hips"], [],
        #     1.5 * force_constant, 1.79, system
        # )
        # util.add_polymer_dihedral_2D(
        #     tubulin_types, [0, 1],
        #     ["tubulinB#bound_"], [0, 0],
        #     ["motor#ATP"], [],
        #     ["hips"], [],
        #     1.5 * force_constant, 1.44, system
        # )
        # # dihedral from bound tubulin to free motor
        # util.add_polymer_dihedral_2D(
        #     ["tubulinB#bound_"], [0, 0],
        #     ["motor#ATP"], [],
        #     ["hips"], [],
        #     ["motor#ADP"], [],
        #     0.5 * force_constant, np.pi * 4./9., system
        # )

    @staticmethod
    def add_angles_between_tubulins(tubulin_types, force_constant, system, util):
        """
        add angles between tubulins
        """
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, 1],
            tubulin_types,
            [0, 0],
            tubulin_types,
            [-1, 0],
            force_constant,
            1.75,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, 1],
            tubulin_types,
            [0, 0],
            tubulin_types,
            [1, 0],
            force_constant,
            1.40,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, -1],
            tubulin_types,
            [0, 0],
            tubulin_types,
            [-1, 0],
            force_constant,
            1.40,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, -1],
            tubulin_types,
            [0, 0],
            tubulin_types,
            [1, 0],
            force_constant,
            1.75,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [-1, 0],
            tubulin_types,
            [0, 0],
            tubulin_types,
            [1, 0],
            force_constant,
            np.pi,
            system,
        )
        util.add_polymer_angle_2D(
            tubulin_types,
            [0, -1],
            tubulin_types,
            [0, 0],
            tubulin_types,
            [0, 1],
            force_constant,
            2.67,
            system,
        )

    @staticmethod
    def add_motor_tubulin_interactions(
        motor_types, bound_tubulin_types, tubulin_types, force_constant, system, util
    ):
        """
        add repulsions between motors and tubulins
        """
        bound_types = []
        for t in bound_tubulin_types:
            bound_types += MicrotubulesUtil.get_all_polymer_tubulin_types(t)
        all_types = []
        for t in tubulin_types:
            all_types += MicrotubulesUtil.get_all_polymer_tubulin_types(t)
        util.add_bond(motor_types, bound_types, force_constant, 4.0, system)
        util.add_repulsion(motor_types, all_types, force_constant, 3.0, system)

    @staticmethod
    def add_motor_bind_tubulin_reaction(system, rate, reaction_distance):
        """
        bind a kinesin motor in ADP state to a free tubulinB
        """
        # spatial reactions
        polymer_numbers = MicrotubulesUtil.get_all_polymer_tubulin_types("")
        # kinesin_states = ["ADP-apo", "ADP-ATP"]
        i = 1
        for n in polymer_numbers:
            # first motor binding
            system.topologies.add_spatial_reaction(
                f"Bind_Tubulin#ADP-ADP{i}: Kinesin(motor#ADP) + Microtubule(tubulinB#{n}) -> \
                Microtubule-Kinesin#Binding(motor#new--tubulinB#bound_{n})",
                rate=rate,
                radius=4.0 + reaction_distance,
            )
            # # second motor binding
            # for s in kinesin_states:
            #     system.topologies.add_spatial_reaction(
            #         f"Bind_Tubulin#{s}{i}: \
            #         Microtubule-Kinesin#{s}(motor#ADP) + \
            #         Microtubule-Kinesin#{s}(tubulinB#{n}) -> \
            #         Microtubule-Kinesin#Binding(motor#new--\
            #         tubulinB#bound_{n}) [self=true]",
            #         rate=rate,
            #         radius=4.0 + reaction_distance,
            #     )
            i += 1
        # structural reaction
        system.topologies.add_structural_reaction(
            "Finish_Bind_Tubulin",
            topology_type="Microtubule-Kinesin#Binding",
            reaction_function=KinesinUtil.reaction_function_motor_bind_tubulin,
            rate_function=ReaddyUtil.rate_function_infinity,
        )

    @staticmethod
    def add_motor_bind_ATP_reaction(system):
        """
        set bound apo motor's state to ATP (and implicitly simulate ATP binding)
        """
        kinesin_states = ["ADP-apo", "ATP-apo", "apo-apo"]
        for state in kinesin_states:
            system.topologies.add_structural_reaction(
                f"Bind_ATP#{state}",
                topology_type=f"Microtubule-Kinesin#{state}",
                reaction_function=KinesinUtil.reaction_function_motor_bind_ATP,
                rate_function=KinesinUtil.rate_function_motor_bind_ATP,
            )

    @staticmethod
    def add_motor_release_tubulin_reaction(system):
        """
        release a bound motor from tubulin
        """
        kinesin_states = ["ADP-ATP", "ATP-ATP", "ATP-apo"]
        for state in kinesin_states:
            system.topologies.add_structural_reaction(
                f"Release_Tubulin#{state}",
                topology_type=f"Microtubule-Kinesin#{state}",
                reaction_function=KinesinUtil.reaction_function_motor_release_tubulin,
                rate_function=KinesinUtil.rate_function_motor_release_tubulin,
            )
        system.topologies.add_structural_reaction(
            "Cleanup_Release_Tubulin",
            topology_type="Microtubule-Kinesin#Releasing",
            reaction_function=KinesinUtil.reaction_function_cleanup_release_tubulin,
            rate_function=ReaddyUtil.rate_function_infinity,
        )
