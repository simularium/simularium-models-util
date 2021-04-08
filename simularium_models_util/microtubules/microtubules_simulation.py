#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import readdy
import os
from shutil import rmtree

from ..common import ReaddyUtil
from .microtubules_util import MicrotubulesUtil


class MicrotubulesSimulation:
    def __init__(self, parameters, record=False, save_checkpoints=False):
        """
        Creates a ReaDDy microtubules simulation

        Ref: http://jcb.rupress.org/content/jcb/217/8/2691/F7.large.jpg

        Params = Dict[str, float]
        keys:
        total_steps, timestep, box_size, temperature_C, viscosity, 
        force_constant, grow_reaction_distance, attach_reaction_distance, 
        n_cpu, tubulin_concentration, seed_n_rings, seed_n_frayed_rings_minus,
        seed_n_frayed_rings_plus, seed_position_offset_x, seed_position_offset_y, 
        seed_position_offset_z, tubulin_radius, protofilament_growth_GTP_rate, 
        protofilament_growth_GDP_rate, protofilament_shrink_GTP_rate, 
        protofilament_shrink_GDP_rate, ring_attach_GTP_rate, ring_attach_GDP_rate, 
        ring_detach_GTP_rate, ring_detach_GDP_rate, hydrolyze_rate, verbose
        """
        self.parameters = parameters
        self.microtubules_util = MicrotubulesUtil(self.parameters)
        self.create_microtubules_system()
        self.create_microtubules_simulation(record, save_checkpoints)

    def create_microtubules_system(self):
        """
        Create the ReaDDy system for microtubules 
        including particle types, constraints, and reactions
        """
        self.system = readdy.ReactionDiffusionSystem([self.parameters["box_size"]] * 3)
        self.parameters["temperature_K"] = self.parameters["temperature_C"] + 273.15
        self.system.temperature = self.parameters["temperature_K"]
        self.add_microtubules_types()
        self.add_microtubules_constraints()
        self.add_microtubules_reactions()

    def add_microtubules_types(self):
        """
        Add particle and topology types for microtubules particles 
        to the ReaDDy system
        """
        tubulin_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["tubulin_radius"], 
            self.parameters["viscosity"], 
            self.parameters["temperature_K"]
        ) #nm^2/s
        self.microtubules_util.add_tubulin_types(self.system, tubulin_diffCoeff)

    def add_microtubules_constraints(self):
        """
        Add geometric constraints for connected microtubules particles, 
        including bonds, angles, and repulsions, to the ReaDDy system
        """
        force_constant = self.parameters["force_constant"]
        util = ReaddyUtil()
        tube_tubulin_types = ["tubulinA#GTP_", "tubulinA#GDP_",
                              "tubulinB#GTP_", "tubulinB#GDP_"]
        bent_tubulin_types = ["tubulinA#GTP_bent_", "tubulinA#GDP_bent_",
                              "tubulinB#GTP_bent_", "tubulinB#GDP_bent_"]
        all_tubulin_types = tube_tubulin_types + bent_tubulin_types
        site_types = ["site#out", "site#1", "site#1_GTP", "site#1_GDP", 
                      "site#1_detach", "site#2", "site#2_GTP", "site#2_GDP", 
                      "site#2_detach", "site#3", "site#4", "site#4_GTP", 
                      "site#4_GDP", "site#new"]
        # bonds
        self.microtubules_util.add_bonds_between_tubulins(all_tubulin_types, 1.2 * force_constant, self.system, util)
        self.microtubules_util.add_tubulin_site_bonds(all_tubulin_types, site_types, force_constant, self.system, util)
        self.microtubules_util.add_bent_site_bonds(force_constant, self.system, util)
        # angles
        self.microtubules_util.add_angles_between_tubulins(
            [tube_tubulin_types, bent_tubulin_types, all_tubulin_types],
            1.2 * force_constant, self.system, util)
        self.microtubules_util.add_tubulin_site_angles(all_tubulin_types, force_constant, self.system, util)
        self.microtubules_util.add_bent_site_angles(all_tubulin_types, force_constant, self.system, util)
        self.microtubules_util.add_edge_site_angles(all_tubulin_types, force_constant, self.system, util)
        # repulsions
        self.microtubules_util.add_polymer_repulsion(all_tubulin_types, force_constant, 4.2, self.system, util)

    def add_microtubules_reactions(self):
        """
        Add reactions to the ReaDDy system
        """
        self.microtubules_util.add_growth_reaction(
            self.system, self.parameters["protofilament_growth_GTP_rate"],
            self.parameters["protofilament_growth_GDP_rate"], self.parameters["grow_reaction_distance"])
        self.microtubules_util.add_shrink_reaction(self.system)
        self.microtubules_util.add_attach_reaction(
            self.system, self.parameters["ring_attach_GTP_rate"],
            self.parameters["ring_attach_GDP_rate"], self.parameters["attach_reaction_distance"])
        self.microtubules_util.add_detach_reaction(self.system)
        self.microtubules_util.add_hydrolyze_reaction(self.system)
        self.system.reactions.add("Cleanup_Sites: site#remove ->", rate=1e30)

    def create_microtubules_simulation(self, record=False, save_checkpoints=False):
        """
        Create the ReaDDy simulation for microtubules
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

    def add_microtubule_seed(self):
        """
        Add a microtubule seed
        """
        self.microtubules_util.add_microtubule(
            int(self.parameters["seed_n_rings"]), 
            int(self.parameters["seed_n_frayed_rings_minus"]), 
            int(self.parameters["seed_n_frayed_rings_plus"]), 
            np.array([
                self.parameters["seed_position_offset_x"], 
                self.parameters["seed_position_offset_y"], 
                self.parameters["seed_position_offset_z"]
            ]), 
            self.simulation
        )

    def add_random_tubulin_dimers(self):
        """
        Add randomly distributed tubulin dimers
        """
        self.microtubules_util.add_tubulin_dimers(
            self.simulation, 
            ReaddyUtil.calculate_nParticles(
                self.parameters["tubulin_concentration"], self.parameters["box_size"]), 
            self.parameters["box_size"]
        )
