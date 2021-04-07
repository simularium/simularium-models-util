#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import readdy
import os
from shutil import rmtree

from simularium_models_util import *
from simularium_models_util.actin import *

# Ref: http://jcb.rupress.org/content/jcb/180/5/887.full.pdf

print("using ReaDDy {}".format(readdy.__version__))

# PARAMETERS ------------------------------------------------------------------

# parameters to adjust per run

verbose = set_verbose(False)

output_name = "actin"
box_size = set_box_size(150.) # nm
total_steps = int(1e3)

actin_concentration = 200. # uM
arp23_concentration = 9. # uM
cap_concentration = 0. # uM

shrink_rate_factor = 1e-5
grow_rate_factor = 1e10
dimerize_rate_factor = 1e-8
reverse_dimerize_rate_factor = 1e13
trimerize_rate_factor = 1e-8
reverse_trimerize_rate_factor = 1e13

reaction_distance = 1. #nm

# intrinsic model parameters

cpus = 4
temperature_C = 22. # Pollard 1986 measured rates at 22C
temperature_K = temperature_C + 273.15
eta = 8.1 #cP, viscosity in cytoplasm (what is viscosity in Pollard 1986?)

n_actin = ReaddyUtil.calculate_nParticles(actin_concentration, box_size)
actin_radius = 2. #nm
actin_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
    actin_radius, eta, temperature_K) #nm^2/s

n_arp23 = ReaddyUtil.calculate_nParticles(arp23_concentration, box_size)
arp23_radius = 2. #nm
arp23_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
    arp23_radius, eta, temperature_K) #nm^2/s

n_cap = ReaddyUtil.calculate_nParticles(cap_concentration, box_size)
cap_radius = 3. #nm
cap_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
    cap_radius, eta, temperature_K) #nm^2/s

# ATP polymerization rates from Joh,
# relative rates for ADP from Pollard Figure 33.8
rates = {}
rates["dimerize"] = dimerize_rate_factor * grow_rate_factor * 2.1e-4 #1/ns
rates["dimerize_reverse"] = reverse_dimerize_rate_factor * shrink_rate_factor * 1.4e-9 #1/ns
rates["trimerize"] = trimerize_rate_factor * grow_rate_factor * 2.1e-4 #1/ns
rates["trimerize_reverse"] = reverse_trimerize_rate_factor * shrink_rate_factor * 1.4e-9 #1/ns
rates["pointed_growth_ATP"] = grow_rate_factor * 2.4e-5 #1/ns
rates["pointed_growth_ADP"] = grow_rate_factor * (0.16 / 1.3) * 2.4e-5 #1/ns
rates["pointed_shrink_ATP"] = shrink_rate_factor * 0.8e-9 #1/ns
rates["pointed_shrink_ADP"] = shrink_rate_factor * (0.3 / 0.8) * 0.8e-9 #1/ns
rates["barbed_growth_ATP"] = grow_rate_factor * 2.1e-4 #1/ns
rates["barbed_growth_ADP"] = grow_rate_factor * (4. / 12.) * 2.1e-4 #1/ns
rates["nucleate_ATP"] = grow_rate_factor * 2.1e-4 #1/ns
rates["nucleate_ADP"] = grow_rate_factor * (4. / 12.) * 2.1e-4 #1/ns
rates["barbed_shrink_ATP"] = shrink_rate_factor * 1.4e-9 #1/ns
rates["barbed_shrink_ADP"] = shrink_rate_factor * (8. / 1.4) * 1.4e-9 #1/ns
rates["arp_bind_ATP"] = grow_rate_factor * 2.1e-4 #1/ns (add arp23 to filament)
rates["arp_bind_ADP"] = grow_rate_factor * (4. / 12.) * 2.1e-4 #1/ns
rates["arp_unbind_ATP"] = shrink_rate_factor * 1.4e-9 #1/ns (add arp23 to filament)
rates["arp_unbind_ADP"] = shrink_rate_factor * (8. / 1.4) * 1.4e-9 #1/ns
rates["barbed_growth_branch_ATP"] = grow_rate_factor * 2.1e-4 #1/ns
rates["barbed_growth_branch_ADP"] = grow_rate_factor * (4. / 12.) * 2.1e-4 #1/ns
rates["debranching_ATP"] = shrink_rate_factor * 1.4e-9 #1/ns
rates["debranching_ADP"] = shrink_rate_factor * (8. / 1.4) * 1.4e-9 #1/ns
rates["cap_bind"] = grow_rate_factor * 2.1e-4 #1/ns
rates["cap_unbind"] = shrink_rate_factor * 1.4e-9 #1/ns
rates["hydrolysis_actin"] = shrink_rate_factor * 3.5e-10 #1/ns
rates["hydrolysis_arp"] = shrink_rate_factor * 3.5e-10 #1/ns
rates["nucleotide_exchange_actin"] = 1e-10 #1/ns (no ref for this)
rates["nucleotide_exchange_arp"] = 1e-10 #1/ns (no ref for this)
set_rates(rates)

recording_stride = max(int(total_steps / 1000.), 1)
observe_stride = max(int(total_steps / 100.), 1)
checkpoint_stride = max(int(total_steps / 10.), 1)
timestep = 0.1 #ns


# REACTION-DIFFUSION SYSTEM ---------------------------------------------------

system = readdy.ReactionDiffusionSystem([box_size]*3)
system.temperature = temperature_K


# PARTICLES & TOPOLOGIES ------------------------------------------------------

'''
actin filaments are polymers and to encode polarity,there are 3 polymer types. 
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
'''

system.topologies.add_type("Arp23-Dimer")
system.add_topology_species("arp2", arp23_diffCoeff)
system.add_topology_species("arp2#branched", arp23_diffCoeff)
system.add_topology_species("arp3", arp23_diffCoeff)
system.add_topology_species("arp3#ATP", arp23_diffCoeff)
system.add_topology_species("arp3#new", arp23_diffCoeff)
system.add_topology_species("arp3#new_ATP", arp23_diffCoeff)

system.topologies.add_type("Cap")
system.add_topology_species("cap", cap_diffCoeff)
system.add_topology_species("cap#new", cap_diffCoeff)
system.add_topology_species("cap#bound", cap_diffCoeff)

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
system.add_topology_species("actin#free", actin_diffCoeff)
system.add_topology_species("actin#free_ATP", actin_diffCoeff)
system.add_topology_species("actin#new", actin_diffCoeff)
system.add_topology_species("actin#new_ATP", actin_diffCoeff)
for i in range(1, 4):
    system.add_topology_species(f"actin#{i}", actin_diffCoeff)
    system.add_topology_species(f"actin#ATP_{i}", actin_diffCoeff)
    system.add_topology_species(f"actin#pointed_{i}", actin_diffCoeff)
    system.add_topology_species(f"actin#pointed_ATP_{i}", actin_diffCoeff)
    system.add_topology_species(f"actin#barbed_{i}", actin_diffCoeff)
    system.add_topology_species(f"actin#barbed_ATP_{i}", actin_diffCoeff)
system.add_topology_species("actin#branch_1", actin_diffCoeff)
system.add_topology_species("actin#branch_ATP_1", actin_diffCoeff)
system.add_topology_species("actin#branch_barbed_1", actin_diffCoeff)
system.add_topology_species("actin#branch_barbed_ATP_1", actin_diffCoeff)


# CONSTRAINTS -----------------------------------------------------------------

force_constant = 250.
util = ReaddyUtil()

add_bonds_between_actins(force_constant, system, util)
add_filament_twist_angles(10 * force_constant, system, util)
add_filament_twist_dihedrals(25 * force_constant, system, util)

add_branch_bonds(force_constant, system, util)
add_branch_angles(10 * force_constant, system, util)
add_branch_dihedrals(force_constant, system, util)

add_cap_bonds(force_constant, system, util)
add_cap_angles(force_constant, system, util)
add_cap_dihedrals(force_constant, system, util)

add_repulsions(force_constant, system, util)


# REACTIONS -------------------------------------------------------------------

add_dimerize_reaction(
    system, 
    rates["dimerize"], 
    2 * actin_radius + reaction_distance
)
add_dimerize_reverse_reaction(system)

add_trimerize_reaction(
    system, 
    rates["trimerize"], 
    2 * actin_radius + reaction_distance
)
add_trimerize_reverse_reaction(system)

add_nucleate_reaction(
    system, 
    rates["nucleate_ATP"], 
    rates["nucleate_ADP"],
    2 * actin_radius + reaction_distance
)

add_pointed_growth_reaction(
    system, 
    rates["pointed_growth_ATP"], 
    rates["pointed_growth_ADP"],
    2 * actin_radius+reaction_distance
)
add_pointed_shrink_reaction(system)

add_barbed_growth_reaction(
    system, 
    rates["barbed_growth_ATP"], 
    rates["barbed_growth_ADP"],
    2 * actin_radius + reaction_distance
)
add_barbed_shrink_reaction(system)

add_hydrolyze_reaction(system)
add_actin_nucleotide_exchange_reaction(system)
add_arp23_nucleotide_exchange_reaction(system)

add_arp23_bind_reaction(
    system, 
    rates["arp_bind_ATP"], 
    rates["arp_bind_ADP"],
    actin_radius + arp23_radius + reaction_distance
)
add_arp23_unbind_reaction(system)

add_nucleate_branch_reaction(
    system, 
    rates["barbed_growth_branch_ATP"], 
    rates["barbed_growth_branch_ADP"],
    arp23_radius + actin_radius + reaction_distance
)
add_debranch_reaction(system)

add_cap_bind_reaction(
    system, 
    rates["cap_bind"], 
    cap_radius + actin_radius + reaction_distance
)
add_cap_unbind_reaction(system)


# SIMULATION ------------------------------------------------------------------

simulation = system.simulation("CPU")
simulation.kernel_configuration.n_threads = cpus

# set initial conditions
# add_linear_fibers(simulation, 15)
# add_branched_fiber(simulation)
# add_dimers(40, system.box_size.magnitude, simulation)
add_monomers(n_actin, system.box_size.magnitude, simulation)
add_arp23_dimers(n_arp23, system.box_size.magnitude, simulation)
# add_capping_protein(n_cap, system.box_size.magnitude, simulation)
# simulation.load_particles_from_checkpoint("checkpoints/checkpoint_100000.h5", n=0)

# Set strides and save topology and observables
simulation.output_file = "{}.h5".format(output_name)

if os.path.exists(simulation.output_file):
    os.remove(simulation.output_file)

simulation.observe.topologies(observe_stride)
simulation.observe.particles(observe_stride)
simulation.observe.reaction_counts(1)

simulation.record_trajectory(recording_stride)
checkpoint_path = "checkpoints/{}/".format(output_name)
if os.path.exists(checkpoint_path):
    rmtree(checkpoint_path)
simulation.make_checkpoints(
    checkpoint_stride, checkpoint_path, 0)
simulation.progress_output_stride = recording_stride


# RUN -------------------------------------------------------------------------

print("total simulated time = {} us".format(total_steps * timestep / 10**3))
simulation.run(total_steps, timestep)

# VISUALIZE -------------------------------------------------------------------

ActinVisualization.visualize_actin(simulation.output_file, box_size, [])
