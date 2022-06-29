#!/usr/bin/env python
# -*- coding: utf-8 -*-

from examples.test import actin_numbers_excel
import readdy
import numpy as np

from ..common import ReaddyUtil
from .actin_util import ActinUtil
from .actin_structure import ActinStructure


class ActinSimulation:
    def __init__(
        self,
        parameters,
        record=False,
        save_checkpoints=False,
    ):
        """
        Creates a ReaDDy branched actin simulation

        Ref: http://jcb.rupress.org/content/jcb/180/5/887.full.pdf

        Params = Dict[str, float]
        keys:
        total_steps, timestep, box_size, temperature_C, viscosity,
        force_constant, reaction_distance, n_cpu, actin_concentration,
        arp23_concentration, cap_concentration, seed_n_fibers, seed_fiber_length,
        actin_radius, arp23_radius, cap_radius, dimerize_rate, dimerize_reverse_rate,
        trimerize_rate, trimerize_reverse_rate, pointed_growth_ATP_rate,
        pointed_growth_ADP_rate, pointed_shrink_ATP_rate,
        pointed_shrink_ADP_rate, barbed_growth_ATP_rate,
        barbed_growth_ADP_rate, nucleate_ATP_rate, nucleate_ADP_rate,
        barbed_shrink_ATP_rate, barbed_shrink_ADP_rate, arp_bind_ATP_rate,
        arp_bind_ADP_rate, arp_unbind_ATP_rate, arp_unbind_ADP_rate,
        barbed_growth_branch_ATP_rate, barbed_growth_branch_ADP_rate,
        debranching_ATP_rate, debranching_ADP_rate, cap_bind_rate,
        cap_unbind_rate, hydrolysis_actin_rate, hydrolysis_arp_rate,
        nucleotide_exchange_actin_rate, nucleotide_exchange_arp_rate, verbose
        """
        self.parameters = parameters
        self.actin_util = ActinUtil(
            self.parameters, self.get_pointed_end_displacements()
        )
        self.create_actin_system()
        self.simulation = ReaddyUtil.create_readdy_simulation(
            self.system,
            self.parameters["n_cpu"],
            self.parameters["name"],
            self.parameters["total_steps"],
            record,
            save_checkpoints,
        )

    def create_actin_system(self):
        """
        Create the ReaDDy system for actin
        including particle types, constraints, and reactions
        """
        self.system = readdy.ReactionDiffusionSystem(
            box_size=[self.parameters["box_size"]] * 3,
            periodic_boundary_conditions=[bool(self.parameters["periodic_boundary"])]
            * 3,
        )
        self.parameters["temperature_K"] = self.parameters["temperature_C"] + 273.15
        self.system.temperature = self.parameters["temperature_K"]
        self.add_particle_types()
        ActinUtil.check_add_global_box_potential(self.system, int(self.parameters["actin_number_types"]))
        self.add_constraints()
        self.add_reactions()

    def add_particle_types(self):
        """
        Add particle and topology types for actin particles
        to the ReaDDy system
        """
        actin_number_types = int(self.parameters["actin_number_types"])
        temperature = self.parameters["temperature_K"]
        viscosity = self.parameters["viscosity"]
        actin_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["actin_radius"], viscosity, temperature
        )  # nm^2/s
        arp23_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["arp23_radius"], viscosity, temperature
        )  # nm^2/s
        cap_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["cap_radius"], viscosity, temperature
        )  # nm^2/s
        self.actin_util.add_actin_types(self.system, actin_diffCoeff, actin_number_types)
        self.actin_util.add_arp23_types(self.system, arp23_diffCoeff)
        self.actin_util.add_cap_types(self.system, cap_diffCoeff)
        self.system.add_species("obstacle", 0.0)

    def add_constraints(self):
        """
        Add geometric constraints for connected actin particles,
        including bonds, angles, and repulsions, to the ReaDDy system
        """
        force_constant = self.parameters["force_constant"]
        util = ReaddyUtil()
        actin_number_types = int(self.parameters["actin_number_types"])
        # linear actin
        self.actin_util.add_bonds_between_actins(force_constant, self.system, util, actin_number_types)
        self.actin_util.add_filament_twist_angles(
            10 * force_constant, self.system, util, actin_number_types)
        self.actin_util.add_filament_twist_dihedrals(
            25 * force_constant, self.system, util, actin_number_types)
        # branch junction
        self.actin_util.add_branch_bonds(force_constant, self.system, util, actin_number_types)
        self.actin_util.add_branch_angles(10 * force_constant, self.system, util, actin_number_types)
        self.actin_util.add_branch_dihedrals(force_constant, self.system, util, actin_number_types)
        # capping protein
        self.actin_util.add_cap_bonds(force_constant, self.system, util, actin_number_types)
        self.actin_util.add_cap_angles(force_constant, self.system, util, actin_number_types)
        self.actin_util.add_cap_dihedrals(force_constant, self.system, util, actin_number_types)
        # repulsions
        self.actin_util.add_repulsions(
            self.parameters["actin_radius"],
            self.parameters["arp23_radius"],
            self.parameters["cap_radius"],
            self.parameters["obstacle_radius"],
            force_constant,
            self.system,
            util,
            self.parameters["actin_number_types"]
        )
        # box potentials
        self.actin_util.add_monomer_box_potentials(self.system)

    def add_reactions(self):
        """
        Add reactions to the ReaDDy system
        """
        actin_number_types = int(self.parameters["actin_number_types"])
        self.actin_util.add_dimerize_reaction(self.system)
        self.actin_util.add_trimerize_reaction(self.system, actin_number_types)
        self.actin_util.add_nucleate_reaction(self.system, actin_number_types)
        self.actin_util.add_pointed_growth_reaction(self.system, actin_number_types)
        self.actin_util.add_barbed_growth_reaction(self.system, actin_number_types)
        self.actin_util.add_nucleate_branch_reaction(self.system)
        self.actin_util.add_arp23_bind_reaction(self.system)
        self.actin_util.add_cap_bind_reaction(self.system)
        self.actin_util.add_dimerize_reverse_reaction(self.system)
        self.actin_util.add_trimerize_reverse_reaction(self.system)
        self.actin_util.add_pointed_shrink_reaction(self.system)
        self.actin_util.add_barbed_shrink_reaction(self.system)
        self.actin_util.add_hydrolyze_reaction(self.system)
        self.actin_util.add_actin_nucleotide_exchange_reaction(self.system)
        self.actin_util.add_arp23_nucleotide_exchange_reaction(self.system)
        self.actin_util.add_arp23_unbind_reaction(self.system)
        self.actin_util.add_debranch_reaction(self.system)
        self.actin_util.add_cap_unbind_reaction(self.system)
        if self.do_pointed_end_translation():
            self.actin_util.add_translate_reaction(self.system)

    def do_pointed_end_translation(self):
        return (
            self.parameters["orthogonal_seed"]
            and int(self.parameters["n_fixed_monomers_pointed"]) > 0
            and (
                self.parameters["displace_pointed_end_tangent"]
                or self.parameters["displace_pointed_end_radial"]
            )
        )

    def get_pointed_end_displacements(self):
        """
        Get parameters for translation of the pointed end of an orthogonal seed
        """
        if not self.do_pointed_end_translation():
            return {}
        if (
            self.parameters["displace_pointed_end_tangent"]
            and self.parameters["displace_pointed_end_radial"]
        ):
            raise Exception(
                "Cannot apply tangent and radial displacements simultaneously"
            )
        if self.parameters["displace_pointed_end_tangent"]:
            displacement = {
                "get_translation": ActinUtil.get_position_for_tangent_translation,
                "parameters": {
                    "total_displacement_nm": np.array(
                        [self.parameters["tangent_displacement_nm"], 0, 0]
                    ),
                    "total_steps": float(self.parameters["total_steps"]),
                },
            }
        if self.parameters["displace_pointed_end_radial"]:
            displacement = {
                "get_translation": ActinUtil.get_position_for_radial_translation,
                "parameters": {
                    "radius_nm": self.parameters["radial_displacement_radius_nm"],
                    "theta_init_radians": np.pi,
                    "theta_final_radians": np.pi
                    + np.deg2rad(self.parameters["radial_displacement_angle_deg"]),
                    "total_steps": float(self.parameters["total_steps"]),
                },
            }
        result = {}
        for monomer_index in range(int(self.parameters["n_fixed_monomers_pointed"])):
            result[monomer_index] = displacement
        return result

    def add_random_monomers(self):
        """
        Add randomly distributed actin monomers, Arp2/3 dimers,
        and capping protein according to concentrations and box size
        """
        self.actin_util.add_actin_monomers(
            ReaddyUtil.calculate_nParticles(
                self.parameters["actin_concentration"], self.parameters["box_size"]
            ),
            self.simulation,
        )
        self.actin_util.add_arp23_dimers(
            ReaddyUtil.calculate_nParticles(
                self.parameters["arp23_concentration"], self.parameters["box_size"]
            ),
            self.simulation,
        )
        self.actin_util.add_capping_protein(
            ReaddyUtil.calculate_nParticles(
                self.parameters["cap_concentration"], self.parameters["box_size"]
            ),
            self.simulation,
        )

    def add_random_linear_fibers(self, use_uuids=True):
        """
        Add randomly distributed and oriented linear fibers
        """
        self.actin_util.add_random_linear_fibers(
            self.simulation,
            int(self.parameters["seed_n_fibers"]),
            self.parameters["seed_fiber_length"],
            -1 if use_uuids else 0,
        )

    def add_fibers_from_data(self, fibers_data, use_uuids=True):
        """
        Add fibers specified in a list of FiberData

        fiber_data: List[FiberData]
        (FiberData for mother fibers only, which should have
        their daughters' FiberData attached to their nucleated arps)
        """
        self.actin_util.add_fibers_from_data(self.simulation, fibers_data, use_uuids)

    def add_monomers_from_data(self, monomer_data):
        """
        Add fibers and monomers specified in the monomer_data, in the form:
        monomer_data = {
            "topologies": {
                [topology ID] : {
                    "type_name": "[topology type]",
                    "particle_ids": [],
                },
            },
            "particles": {
                [particle ID] : {
                    "type_name": "[particle type]",
                    "position": np.zeros(3),
                    "neighbor_ids": [],
                },
            },
        }
        * IDs are ints
        """
        self.topologies = self.actin_util.add_monomers_from_data(
            self.simulation, monomer_data
        )

    def add_obstacles(self):
        """
        Add obstacle particles
        """
        n = 0
        while f"obstacle{n}_position_x" in self.parameters:
            self.simulation.add_particle(
                type="obstacle",
                position=[
                    float(self.parameters[f"obstacle{n}_position_x"]),
                    float(self.parameters[f"obstacle{n}_position_y"]),
                    float(self.parameters[f"obstacle{n}_position_z"]),
                ],
            )
            n += 1
        print(f"Added {n} obstacle(s).")

    def add_crystal_structure_monomers(self):
        """
        Add monomers exactly from the branched actin crystal structure
        """
        type_names = [
            "actin#pointed_ATP_1",
            "actin#ATP_2",
            "actin#ATP_3",
            "actin#ATP_1",
            "actin#ATP_2",
            "actin#ATP_3",
            "actin#ATP_1",
            "actin#barbed_ATP_2",
            "arp2#branched",
            "arp3#ATP",
            "actin#branch_ATP_1",
            "actin#ATP_2",
            "actin#barbed_ATP_3",
        ]
        positions = np.zeros((13, 3))
        positions[:8, :] = ActinStructure.mother_positions
        positions[8, :] = ActinStructure.arp2_position
        positions[9, :] = ActinStructure.arp3_position
        positions[10:, :] = ActinStructure.daughter_positions
        neighbor_ids = [
            [1],
            [0, 2],
            [1, 3],
            [2, 4, 8],
            [3, 5, 9],
            [4, 6],
            [5, 7],
            [6],
            [3, 9, 10],
            [4, 8],
            [8, 11],
            [10, 12],
            [11],
        ]
        monomer_data = {
            "topologies": {
                0: {
                    "type_name": "Actin-Polymer",
                    "particle_ids": [],
                }
            },
            "particles": {},
        }
        for index in range(len(type_names)):
            monomer_data["topologies"][0]["particle_ids"].append(index)
            monomer_data["particles"][index] = {
                "type_name": type_names[index],
                "position": np.array(positions[index]),
                "neighbor_ids": neighbor_ids[index],
            }
        self.add_monomers_from_data(monomer_data)

    def simulate(self, d_time):
        """
        Simulate in ReaDDy for the given d_time seconds
        """

        def loop():
            readdy_actions = self.simulation._actions
            init = readdy_actions.initialize_kernel()
            diffuse = readdy_actions.integrator_euler_brownian_dynamics(
                self.parameters["internal_timestep"]
            )
            calculate_forces = readdy_actions.calculate_forces()
            create_nl = readdy_actions.create_neighbor_list(
                self.system.calculate_max_cutoff().magnitude
            )
            update_nl = readdy_actions.update_neighbor_list()
            react = readdy_actions.reaction_handler_uncontrolled_approximation(
                self.parameters["internal_timestep"]
            )
            observe = readdy_actions.evaluate_observables()
            init()
            create_nl()
            calculate_forces()
            update_nl()
            observe(0)
            n_steps = int(d_time * 1e9 / self.parameters["internal_timestep"])
            for t in range(1, n_steps + 1):
                diffuse()
                update_nl()
                react()
                update_nl()
                calculate_forces()
                observe(t)

        self.simulation._run_custom_loop(loop)

    def get_current_monomers(self):
        """
        During a running simulation,
        get data for topologies of particles
        from readdy.simulation.current_topologies
        as monomers
        """
        return ReaddyUtil.get_current_monomers(self.simulation.current_topologies)
