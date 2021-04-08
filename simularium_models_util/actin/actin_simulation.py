#!/usr/bin/env python
# -*- coding: utf-8 -*-

import readdy
import os
from shutil import rmtree

from ..common import ReaddyUtil
from .actin_util import ActinUtil


class ActinSimulation:
    def __init__(self, parameters, record=False, save_checkpoints=False):
        """
        Creates a ReaDDy branched actin simulation
        
        Ref: http://jcb.rupress.org/content/jcb/180/5/887.full.pdf

        Params = Dict[str, float]
        keys:
        verbose, box_size, actin_concentration, arp23_concentration, 
        cap_concentration, dimerize_rate, dimerize_reverse_rate, 
        trimerize_rate, trimerize_reverse_rate, pointed_growth_ATP_rate, 
        pointed_growth_ADP_rate, pointed_shrink_ATP_rate, 
        pointed_shrink_ADP_rate, barbed_growth_ATP_rate, 
        barbed_growth_ADP_rate, nucleate_ATP_rate, nucleate_ADP_rate, 
        barbed_shrink_ATP_rate, barbed_shrink_ADP_rate, arp_bind_ATP_rate, 
        arp_bind_ADP_rate, arp_unbind_ATP_rate, arp_unbind_ADP_rate, 
        barbed_growth_branch_ATP_rate, barbed_growth_branch_ADP_rate, 
        debranching_ATP_rate, debranching_ADP_rate, cap_bind_rate, 
        cap_unbind_rate, hydrolysis_actin_rate, hydrolysis_arp_rate, 
        nucleotide_exchange_actin_rate, nucleotide_exchange_arp_rate, 
        temperature_C, viscosity, force_constant, reaction_distance, 
        actin_radius, arp23_radius, cap_radius, n_cpu
        """
        self.parameters = parameters
        self.actin_util = ActinUtil(self.parameters)
        self.create_actin_system()
        self.create_actin_simulation(record, save_checkpoints)

    def create_actin_system(self):
        """
        Create the ReaDDy system for actin 
        including particle types, constraints, and reactions
        """
        self.system = readdy.ReactionDiffusionSystem([self.parameters["box_size"]] * 3)
        self.parameters["temperature_K"] = self.parameters["temperature_C"] + 273.15
        self.system.temperature = self.parameters["temperature_K"]
        self.add_actin_types()
        self.add_actin_constraints()
        self.add_actin_reactions()

    def add_actin_types(self):
        """
        Add particle and topology types for actin particles 
        to the ReaDDy system
        """
        temperature = self.parameters["temperature_K"]
        viscosity = self.parameters["viscosity"]
        actin_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["actin_radius"], 
            viscosity, 
            temperature
        ) #nm^2/s
        arp23_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["arp23_radius"], 
            viscosity, 
            temperature
        ) #nm^2/s
        cap_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["actin_radius"], 
            viscosity, 
            temperature
        ) #nm^2/s
        self.actin_util.add_actin_types(self.system, actin_diffCoeff)
        self.actin_util.add_arp23_types(self.system, actin_diffCoeff)
        self.actin_util.add_cap_types(self.system, cap_diffCoeff)

    def add_actin_constraints(self):
        """
        Add geometric constraints for connected actin particles, 
        including bonds, angles, and repulsions, to the ReaDDy system
        """
        force_constant = self.parameters["force_constant"]
        util = ReaddyUtil()
        # linear actin
        self.actin_util.add_bonds_between_actins(force_constant, self.system, util)
        self.actin_util.add_filament_twist_angles(10 * force_constant, self.system, util)
        self.actin_util.add_filament_twist_dihedrals(25 * force_constant, self.system, util)
        # branch junction
        self.actin_util.add_branch_bonds(force_constant, self.system, util)
        self.actin_util.add_branch_angles(10 * force_constant, self.system, util)
        self.actin_util.add_branch_dihedrals(force_constant, self.system, util)
        # capping protein
        self.actin_util.add_cap_bonds(force_constant, self.system, util)
        self.actin_util.add_cap_angles(force_constant, self.system, util)
        self.actin_util.add_cap_dihedrals(force_constant, self.system, util)
        # repulsions
        self.actin_util.add_repulsions(force_constant, self.system, util)

    def add_actin_reactions(self):
        """
        Add reactions to the ReaDDy system
        """
        actin_radius = self.parameters["actin_radius"]
        reaction_distance = self.parameters["reaction_distance"]
        self.actin_util.add_dimerize_reaction(
            self.system, 
            self.parameters["dimerize_rate"], 
            2 * actin_radius + reaction_distance
        )
        self.actin_util.add_dimerize_reverse_reaction(self.system)
        self.actin_util.add_trimerize_reaction(
            self.system, 
            self.parameters["trimerize_rate"], 
            2 * actin_radius + reaction_distance
        )
        self.actin_util.add_trimerize_reverse_reaction(self.system)
        self.actin_util.add_nucleate_reaction(
            self.system, 
            self.parameters["nucleate_ATP_rate"], 
            self.parameters["nucleate_ADP_rate"],
            2 * actin_radius + reaction_distance
        )
        self.actin_util.add_pointed_growth_reaction(
            self.system, 
            self.parameters["pointed_growth_ATP_rate"], 
            self.parameters["pointed_growth_ADP_rate"],
            2 * actin_radius + reaction_distance
        )
        self.actin_util.add_pointed_shrink_reaction(self.system)
        self.actin_util.add_barbed_growth_reaction(
            self.system, 
            self.parameters["barbed_growth_ATP_rate"], 
            self.parameters["barbed_growth_ADP_rate"],
            2 * actin_radius + reaction_distance
        )
        self.actin_util.add_barbed_shrink_reaction(self.system)
        self.actin_util.add_hydrolyze_reaction(self.system)
        self.actin_util.add_actin_nucleotide_exchange_reaction(self.system)
        self.actin_util.add_arp23_nucleotide_exchange_reaction(self.system)
        self.actin_util.add_arp23_bind_reaction(
            self.system, 
            self.parameters["arp_bind_ATP_rate"], 
            self.parameters["arp_bind_ADP_rate"],
            actin_radius + self.parameters["arp23_radius"] + reaction_distance
        )
        self.actin_util.add_arp23_unbind_reaction(self.system)
        self.actin_util.add_nucleate_branch_reaction(
            self.system, 
            self.parameters["barbed_growth_branch_ATP_rate"], 
            self.parameters["barbed_growth_branch_ADP_rate"],
            actin_radius + self.parameters["arp23_radius"] + reaction_distance
        )
        self.actin_util.add_debranch_reaction(self.system)
        self.actin_util.add_cap_bind_reaction(
            self.system, 
            self.parameters["cap_bind_rate"], 
            actin_radius + self.parameters["cap_radius"] + reaction_distance
        )
        self.actin_util.add_cap_unbind_reaction(self.system)

    def create_actin_simulation(self, record=False, save_checkpoints=False):
        """
        Create the ReaDDy simulation for actin
        """
        self.simulation = self.system.simulation("CPU")
        self.simulation.kernel_configuration.n_threads = self.parameters["n_cpu"]
        if record:
            self.simulation.output_file = "{}.h5".format(self.parameters["name"])
            if os.path.exists(self.simulation.output_file):
                os.remove(self.simulation.output_file)
            recording_stride = max(int(self.parameters["total_steps"] / 1000.), 1)
            self.simulation.record_trajectory(recording_stride)
            self.simulation.observe.topologies(recording_stride)
            self.simulation.observe.particles(recording_stride)
            self.simulation.observe.reaction_counts(1)
            self.simulation.progress_output_stride = recording_stride
        if save_checkpoints:
            checkpoint_stride = max(int(self.parameters["total_steps"] / 10.), 1)
            checkpoint_path = "checkpoints/{}/".format(self.parameters["name"])
            if os.path.exists(checkpoint_path):
                rmtree(checkpoint_path)
            self.simulation.make_checkpoints(checkpoint_stride, checkpoint_path, 0)

    def add_random_monomers(self):
        """
        Add randomly distributed actin monomers, Arp2/3 dimers, 
        and capping protein according to concentrations and box size
        """
        self.actin_util.add_actin_monomers(
            ReaddyUtil.calculate_nParticles(
                self.parameters["actin_concentration"], self.parameters["box_size"]), 
            self.simulation
        )
        self.actin_util.add_arp23_dimers(
            ReaddyUtil.calculate_nParticles(
                self.parameters["arp23_concentration"], self.parameters["box_size"]), 
            self.simulation
        )
        self.actin_util.add_capping_protein(
            ReaddyUtil.calculate_nParticles(
                self.parameters["cap_concentration"], self.parameters["box_size"]), 
            self.simulation
        )

    def add_random_linear_fibers(self):
        """
        Add randomly distributed and oriented linear fibers
        """
        self.actin_util.add_random_linear_fibers(
            self.simulation, int(self.parameters["n_fibers"]), self.parameters["fiber_length"])

    def add_fibers_from_data(self, fibers_data):
        """
        Add fibers specified in a list of FiberData

        fiber_data: List[FiberData]
        (FiberData for mother fibers only, which should have 
        their daughters' FiberData attached to their nucleated arps)
        """
        self.actin_util.add_fibers_from_data(self.simulation, fibers_data)
