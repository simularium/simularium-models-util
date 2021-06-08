#!/usr/bin/env python
# -*- coding: utf-8 -*-

import readdy

from ..common import ReaddyUtil
from .actin_util import ActinUtil


class ActinSimulation:
    def __init__(self, parameters, record=False, save_checkpoints=False):
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
        self.actin_util = ActinUtil(self.parameters)
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
            self.parameters["actin_radius"], viscosity, temperature
        )  # nm^2/s
        arp23_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["arp23_radius"], viscosity, temperature
        )  # nm^2/s
        cap_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            self.parameters["actin_radius"], viscosity, temperature
        )  # nm^2/s
        self.actin_util.add_actin_types(self.system, actin_diffCoeff)
        self.actin_util.add_arp23_types(self.system, arp23_diffCoeff)
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
        self.actin_util.add_filament_twist_angles(
            10 * force_constant, self.system, util
        )
        self.actin_util.add_filament_twist_dihedrals(
            25 * force_constant, self.system, util
        )
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
        self.actin_util.add_spatial_dimerize_reaction(self.system)
        self.actin_util.add_spatial_trimerize_reaction(self.system)
        self.actin_util.add_spatial_nucleate_reaction(self.system)
        self.actin_util.add_spatial_pointed_growth_reaction(self.system)
        self.actin_util.add_spatial_barbed_growth_reaction(self.system)
        self.actin_util.add_spatial_nucleate_branch_reaction(self.system)
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
        if self.parameters["nonspatial_polymerization"]:
            self.actin_util.add_nonspatial_trimerize_reaction(self.system)
            self.actin_util.add_nonspatial_nucleate_reaction(self.system)
            self.actin_util.add_nonspatial_pointed_growth_reaction(self.system)
            self.actin_util.add_nonspatial_barbed_growth_reaction(self.system)
            self.actin_util.add_nonspatial_nucleate_branch_reaction(self.system)

    def add_random_monomers(self):
        """
        Add randomly distributed actin monomers, Arp2/3 dimers,
        and capping protein according to concentrations and box size
        """
        if not self.parameters["nonspatial_polymerization"]:
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

    def add_random_linear_fibers(self):
        """
        Add randomly distributed and oriented linear fibers
        """
        self.actin_util.add_random_linear_fibers(
            self.simulation,
            int(self.parameters["seed_n_fibers"]),
            self.parameters["seed_fiber_length"],
        )

    def add_fibers_from_data(self, fibers_data):
        """
        Add fibers specified in a list of FiberData

        fiber_data: List[FiberData]
        (FiberData for mother fibers only, which should have
        their daughters' FiberData attached to their nucleated arps)
        """
        self.actin_util.add_fibers_from_data(self.simulation, fibers_data)
