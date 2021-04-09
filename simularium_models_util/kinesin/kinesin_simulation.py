#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import readdy

from ..common import ReaddyUtil
from ..microtubules import MicrotubulesUtil
from .kinesin_util import KinesinUtil


class KinesinSimulation:
    def __init__(self, parameters, record=False, save_checkpoints=False):
        """
        Creates a ReaDDy kinesin simulation

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
        self.kinesin_util = KinesinUtil(self.parameters)
        self.create_kinesin_system()
        self.simulation = ReaddyUtil.create_readdy_simulation(
            self.system, self.parameters["n_cpu"], self.parameters["name"], 
            self.parameters["total_steps"], record, save_checkpoints)

    def create_kinesin_system(self):
        """
        Create the ReaDDy system for kinesin 
        including particle types, constraints, and reactions
        """
        self.system = readdy.ReactionDiffusionSystem([self.parameters["box_size"]] * 3)
        self.parameters["temperature_K"] = self.parameters["temperature_C"] + 273.15
        self.system.temperature = self.parameters["temperature_K"]
        self.motor_types = ["motor#ADP", "motor#ATP", "motor#apo", "motor#new"]
        self.tubulin_types = ["tubulinA#", "tubulinB#", "tubulinB#bound_"]
        self.add_kinesin_types()
        self.add_kinesin_constraints()
        self.add_kinesin_reactions()

    def add_kinesin_types(self):
        """
        Add particle and topology types for kinesin particles 
        to the ReaDDy system
        """
        temperature_K = self.parameters["temperature_K"]
        viscosity = self.parameters["viscosity"]
        hips_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["hips_radius"], viscosity, temperature_K) #nm^2/s
        cargo_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["cargo_radius"], viscosity, temperature_K) #nm^2/s
        motor_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["motor_radius"], viscosity, temperature_K) #nm^2/s
        tubulin_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["tubulin_radius"], viscosity, temperature_K) #nm^2/s
        self.system.topologies.add_type("Kinesin")
        self.system.topologies.add_type("Microtubule-Kinesin#ADP-ATP")
        self.system.topologies.add_type("Microtubule-Kinesin#ADP-apo")
        self.system.topologies.add_type("Microtubule-Kinesin#ATP-ATP")
        self.system.topologies.add_type("Microtubule-Kinesin#ATP-apo")
        self.system.topologies.add_type("Microtubule-Kinesin#apo-apo")
        self.system.topologies.add_type("Microtubule-Kinesin#Binding")
        self.system.topologies.add_type("Microtubule-Kinesin#Releasing")
        self.system.add_topology_species("hips", hips_diffCoeff)
        self.system.add_topology_species("cargo", cargo_diffCoeff)
        for motor_type in self.motor_types:
            self.system.add_topology_species(
                motor_type, motor_diffCoeff)
        self.system.topologies.add_type("Microtubule")
        for tubulin_type in self.tubulin_types:
            MicrotubulesUtil.add_polymer_topology_species(
                tubulin_type, tubulin_diffCoeff, self.system)

    def add_kinesin_constraints(self):
        """
        Add geometric constraints for connected kinesin particles, 
        including bonds, angles, and repulsions, to the ReaDDy system
        """
        force_constant = self.parameters["force_constant"]
        microtubule_force_constant = self.parameters["microtubules_force_constant"]
        util = ReaddyUtil()
        self.kinesin_util.add_kinesin_bonds_and_repulsions(
            self.motor_types, force_constant, self.system, util)
        self.kinesin_util.add_kinesin_angles_and_dihedrals(
            self.tubulin_types[-1:], force_constant, self.system, util)
        self.kinesin_util.add_tubulin_bonds_and_repulsions(
            self.tubulin_types, microtubule_force_constant, self.system, util)
        self.kinesin_util.add_angles_between_tubulins(
            self.tubulin_types, microtubule_force_constant, self.system, util)
        self.kinesin_util.add_motor_tubulin_interactions(
            self.motor_types, self.tubulin_types[-2:], self.tubulin_types, force_constant, self.system, util)

    def add_kinesin_reactions(self):
        """
        Add reactions to the ReaDDy system
        """
        self.kinesin_util.add_motor_bind_tubulin_reaction(
            self.system, self.parameters["motor_bind_tubulin_rate"], self.parameters["reaction_distance"])
        self.kinesin_util.add_motor_bind_ATP_reaction(self.system)
        self.kinesin_util.add_motor_release_tubulin_reaction(self.system)

    def add_microtubule(self):
        """
        Add a microtubule
        """
        MicrotubulesUtil.add_microtubule(
            int(self.parameters["microtubule_n_rings"]), 0, 0, np.zeros(3), self.simulation, False)

    def add_kinesin(self):
        """
        Add a kinesin
        """
        self.kinesin_util.add_kinesin(
            np.array([
                self.parameters["kinesin_position_x"], 
                self.parameters["kinesin_position_y"], 
                self.parameters["kinesin_position_z"], 
            ]), 
            self.simulation
        )
