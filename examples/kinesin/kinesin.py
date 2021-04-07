import numpy as np
import readdy
import os
import random
import sys
from shutil import rmtree

from simularium_models_util import ReaddyUtil
from simularium_models_util.kinesin import *

# Ref: https://www.nature.com/articles/nchembio.2028

print("using ReaDDy {}".format(readdy.__version__))

# PARAMETERS ------------------------------------------------------------------

# parameters to adjust per run

verbose = set_verbose(True)

output_name = "kinesin"
box_size = set_box_size(300.) #nm
total_steps = int(1e3)

motor_diffusion_factor = 1.
hips_diffusion_factor = 1.
cargo_diffusion_factor = 1. #35.
rate_factor = 1.

reaction_distance = 0.

# intrinsic model parameters

cpus = 4
temperature_C = 37.
temperature_K = temperature_C + 273.15
eta = 8.1 #cP, viscosity in cytoplasm

motor_radius = 2. #nm
motor_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
    motor_radius, eta, temperature_K) #nm^2/s

hips_radius = 1. #nm
hips_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
    hips_radius, eta, temperature_K) #nm^2/s

cargo_radius = 15. #nm
cargo_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
    cargo_radius, eta, temperature_K) #nm^2/s
    
tubulin_radius = 2. #nm
tubulin_diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
    tubulin_radius, eta, temperature_K) #nm^2/s

# TODO use actual rates from Tomishige
rates = {}
rates["motor_bind_tubulin"] = rate_factor * 5. #1/ns
rates["motor_bind_ATP"] = rate_factor * 5. #1/ns
rates["motor_release_tubulin"] = rate_factor * 1.8e-5 #1/ns
set_rates(rates)

recording_stride = max(int(total_steps / 10**3), 1)
observe_stride = max(int(total_steps / 10**2), 1)
checkpoint_stride = max(int(total_steps / 10**1), 1)
timestep = 0.05 #ns


# REACTION-DIFFUSION SYSTEM ---------------------------------------------------

system = readdy.ReactionDiffusionSystem([box_size]*3)
system.temperature = temperature_K


# PARTICLES & TOPOLOGIES ------------------------------------------------------

system.topologies.add_type("Kinesin")
system.topologies.add_type("Microtubule-Kinesin#ADP-ATP")
system.topologies.add_type("Microtubule-Kinesin#ADP-apo")
system.topologies.add_type("Microtubule-Kinesin#ATP-ATP")
system.topologies.add_type("Microtubule-Kinesin#ATP-apo")
system.topologies.add_type("Microtubule-Kinesin#apo-apo")
system.topologies.add_type("Microtubule-Kinesin#Binding")
system.topologies.add_type("Microtubule-Kinesin#Releasing")
system.add_topology_species("hips", hips_diffusion_factor * hips_diffCoeff)
system.add_topology_species("cargo", cargo_diffusion_factor * cargo_diffCoeff)
motor_types = ["motor#ADP", "motor#ATP", "motor#apo", "motor#new"]
for motor_type in motor_types:
    system.add_topology_species(
        motor_type, motor_diffusion_factor * motor_diffCoeff)

system.topologies.add_type("Microtubule")
tubulin_types = ["tubulinA#", "tubulinB#", "tubulinB#bound_"]
for tubulin_type in tubulin_types:
    add_polymer_topology_species(tubulin_type, tubulin_diffCoeff, system)


# CONSTRAINTS -----------------------------------------------------------------

force_constant = 400.
microtubule_force_constant = 280.
util = ReaddyUtil()

add_kinesin_bonds_and_repulsions(
    motor_types, force_constant, system, util)
add_kinesin_angles_and_dihedrals(
    tubulin_types[-1:], force_constant, system, util)

add_tubulin_bonds_and_repulsions(tubulin_types, microtubule_force_constant, system, util)
add_angles_between_tubulins(tubulin_types, microtubule_force_constant, system, util)

add_motor_tubulin_interactions(
    motor_types, tubulin_types[-2:], tubulin_types,
    force_constant, system, util)


# REACTIONS -------------------------------------------------------------------

add_motor_bind_tubulin_reaction(
    system, rates["motor_bind_tubulin"], reaction_distance)
add_motor_bind_ATP_reaction(system)
add_motor_release_tubulin_reaction(system)


# SIMULATION ------------------------------------------------------------------

simulation = system.simulation("CPU")
simulation.kernel_configuration.n_threads = cpus

add_microtubule(24, np.array([0., 0., 0.]), simulation)
add_kinesin(np.array([0., 14., -30.]), simulation)
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

KinesinVisualization.visualize_kinesin(simulation.output_file, box_size, [])
