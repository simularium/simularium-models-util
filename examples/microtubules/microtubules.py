import numpy as np
import readdy
import os
from shutil import rmtree

from simularium_models_util import ReaddyUtil
from simularium_models_util.actin import *

# Ref: http://jcb.rupress.org/content/jcb/217/8/2691/F7.large.jpg

print("using ReaDDy {}".format(readdy.__version__))

# PARAMETERS ------------------------------------------------------------------

# parameters to adjust per run

verbose = set_verbose(False)

output_name = "microtubules"
box_size = set_box_size(150.) #nm
total_steps = int(1e7)

tubulin_concentration = 100. # uM

rate_factor = 5e3
growth_rate_factor = 300.
shrink_rate_factor = 10.
attach_rate_factor = 1e5
detach_rate_factor = 1e4
hydrolyze_rate_factor = 1.
GTP_rate_factor = 20.

grow_reaction_distance = 1. #nm
attach_reaction_distance = 1.7 #nm

# intrinsic model parameters

cpus = 4
temperature_C = 37.
temperature_K = temperature_C + 273.15
eta = 8.1 #cP, viscosity in cytoplasm

n_tubulin = ReaddyUtil.calculate_nParticles(tubulin_concentration, box_size)
tubulin_radius = 2. #nm
tubulin_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
    tubulin_radius, eta, temperature_K) #nm^2/s

# TODO research actual rates
rates = {}
rates["protofilament_growth_GTP"] = (
    rate_factor * growth_rate_factor * 1e-8) #1/ns
rates["protofilament_growth_GDP"] = (
    rate_factor * growth_rate_factor / GTP_rate_factor * 1e-8) #1/ns
rates["protofilament_shrink_GTP"] = (
    rate_factor * shrink_rate_factor / GTP_rate_factor * 3.5e-9) #1/ns
rates["protofilament_shrink_GDP"] = (
    rate_factor * shrink_rate_factor * 3.5e-9) #1/ns
rates["ring_attach_GTP"] = rate_factor * attach_rate_factor * 2.5e-8 #1/ns
rates["ring_attach_GDP"] = (
    rate_factor * attach_rate_factor / GTP_rate_factor * 2.5e-8) #1/ns
rates["ring_detach_GTP"] = (
    rate_factor * detach_rate_factor / GTP_rate_factor * 3.5e-9) #1/ns
rates["ring_detach_GDP"] = rate_factor * detach_rate_factor * 3.5e-9 #1/ns
rates["hydrolyze"] = rate_factor * hydrolyze_rate_factor * 5e-9 #1/ns
set_rates(rates)

recording_stride = max(int(total_steps / 10**3), 1)
observe_stride = max(int(total_steps / 10**2), 1)
checkpoint_stride = max(int(total_steps / 10**1), 1)
timestep = 0.1 #ns


# REACTION-DIFFUSION SYSTEM ---------------------------------------------------

system = readdy.ReactionDiffusionSystem([box_size]*3)
system.temperature = temperature_K


# PARTICLES & TOPOLOGIES ------------------------------------------------------

'''
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
'''

system.topologies.add_type("Dimer")
system.add_topology_species("tubulinA#free", tubulin_diffCoeff)
system.add_topology_species("tubulinB#free", tubulin_diffCoeff)

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

tube_tubulin_types = ["tubulinA#GTP_", "tubulinA#GDP_",
                      "tubulinB#GTP_", "tubulinB#GDP_"]
bent_tubulin_types = ["tubulinA#GTP_bent_", "tubulinA#GDP_bent_",
                      "tubulinB#GTP_bent_", "tubulinB#GDP_bent_"]
all_tubulin_types = tube_tubulin_types + bent_tubulin_types
for tubulin_type in all_tubulin_types:
    add_polymer_topology_species(tubulin_type, tubulin_diffCoeff, system)

site_types = ["site#out", "site#1", "site#1_GTP", "site#1_GDP", "site#1_detach",
              "site#2", "site#2_GTP", "site#2_GDP", "site#2_detach", "site#3",
              "site#4", "site#4_GTP", "site#4_GDP", "site#new"]
system.add_species("site#remove", 0)
for site_type in site_types:
    system.add_topology_species(site_type, tubulin_diffCoeff)


# CONSTRAINTS -----------------------------------------------------------------

force_constant = 75.
multiplier = 1.2
util = ReaddyUtil()

# bonds
add_bonds_between_tubulins(all_tubulin_types, multiplier*force_constant, system, util)
add_tubulin_site_bonds(all_tubulin_types, site_types, force_constant, system, util)
add_bent_site_bonds(force_constant, system, util)

# angles
add_angles_between_tubulins(
    [tube_tubulin_types, bent_tubulin_types, all_tubulin_types],
    multiplier*force_constant, system, util)
add_tubulin_site_angles(all_tubulin_types, force_constant, system, util)
add_bent_site_angles(all_tubulin_types, force_constant, system, util)
add_edge_site_angles(all_tubulin_types, force_constant, system, util)

# repulsions
add_polymer_repulsion(all_tubulin_types, force_constant, 4.2, system, util)


# REACTIONS -------------------------------------------------------------------

add_growth_reaction(system, rates["protofilament_growth_GTP"],
    rates["protofilament_growth_GDP"], grow_reaction_distance)
add_shrink_reaction(system)

add_attach_reaction(system, rates["ring_attach_GTP"],
    rates["ring_attach_GDP"], attach_reaction_distance)
add_detach_reaction(system)

add_hydrolyze_reaction(system)

system.reactions.add("Cleanup_Sites: site#remove ->", rate=1e30)


# SIMULATION ------------------------------------------------------------------

simulation = system.simulation("CPU")
simulation.kernel_configuration.n_threads = cpus

add_microtubule(32, 6, 6, np.array([0., 0., -40.]), simulation)
add_tubulin_dimers(simulation, n_tubulin)
# simulation.load_particles_from_checkpoint(
#     "checkpoints/microtubules/checkpoint_500000.h5", n=0)

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

MicrotubulesVisualization.visualize_microtubules(simulation.output_file, box_size, [])
