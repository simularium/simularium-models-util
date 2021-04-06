import numpy as np
import readdy
import math
import random
import copy
import matplotlib.pyplot as plt

from ..common import ReaddyUtil

'''
print debug statements?
'''
verbose = False
def set_verbose(v):

    global verbose
    verbose = v
    return v

'''
set the box_size
'''
box_size = 0.
def set_box_size(s):

    global box_size
    box_size = s
    return s

'''
set keep_concentrations_constant
'''
keep_concentrations_constant = False
def set_keep_concentrations_constant(x):

    global keep_concentrations_constant
    keep_concentrations_constant = x
    return x

'''
set reaction rates
'''
rates = {}
def set_rates(r):

    global rates
    rates = r

'''
get the next tubulin neighbor in the branch from site named direction
'''
def get_neighboring_tubulin(topology, vertex, polymer_offsets):

    return ReaddyUtil.get_neighbor_of_type(
        topology, vertex,
        polymer_indices_to_string(increment_polymer_indices(
            get_polymer_indices(
                topology.particle_type_of_vertex(vertex)), polymer_offsets)),
        False
    )

'''
clamp offsets so y polymer offset is incremented
if new x polymer index is not in [1,3]
'''
def clamp_polymer_offsets(polymer_index_x, polymer_offsets):

    if len(polymer_offsets) < 2:
        return polymer_offsets

    offsets = copy.deepcopy(polymer_offsets)
    if offsets[0] != 0:
        if polymer_index_x + offsets[0] < 1:
            offsets[1] -= 1
        elif polymer_index_x + offsets[0] > 3:
            offsets[1] += 1

    return offsets

'''
get the x and y polymer index for a particle
'''
def polymer_indices_to_string(polymer_indices):

    return "_{}_{}".format(polymer_indices[0], polymer_indices[1])

'''
increment the x and y polymer index for a particle
'''
def increment_polymer_indices(polymer_indices, polymer_offsets):

    polymer_offsets = clamp_polymer_offsets(
        polymer_indices[0], polymer_offsets)
    x = ReaddyUtil.calculate_polymer_number(
        polymer_indices[0], polymer_offsets[0])
    y = ReaddyUtil.calculate_polymer_number(
        polymer_indices[1], polymer_offsets[1])

    return [x, y]

'''
get the x and y polymer index for a particle
'''
def get_polymer_indices(particle_type):

    if "tubulin" not in particle_type:
        return []

    flag_string = particle_type[particle_type.index("#")+1:]
    flags = flag_string.split("_")

    if len(flags) <= 1:
        return []

    x = int(flags[-2])
    y = int(flags[-1])

    return [x, y]

'''
gets a list of all polymer numbers
("type1_1", "type1_2", "type1_3", "type2_1", ... "type3_3")
    for type particle_type

    returns list of types
'''
def get_all_polymer_tubulin_types(particle_type):

    result = []
    for x in range(1,4):
        for y in range(1,4):

            result.append("{}{}_{}".format(particle_type, x, y))

    return result

'''
adds topology species for all polymer numbers
("type1_1", "type1_2", "type1_3", "type2_1", ... "type3_3")
    for type particle_type
    with diffusion coefficient diffCoeff [nm^2/s]
'''
def add_polymer_topology_species(particle_type, diffCoeff, system):

    types = get_all_polymer_tubulin_types(particle_type)

    for t in types:
        system.add_topology_species(t, diffCoeff)

'''
creates a list of types with 2D polymer numbers
    for each type in particle types
    at polymer number x_y
    with polymer_offsets [dx, dy] both in [-1, 1]

    returns list of types
'''
def get_types_with_polymer_numbers(particle_types, x, y, polymer_offsets):

    types = []
    for t in particle_types:
        types.append("{}{}_{}".format(t,
            ReaddyUtil.calculate_polymer_number(x, polymer_offsets[0]),
            ReaddyUtil.calculate_polymer_number(y, polymer_offsets[1]))
        if len(polymer_offsets) > 0 else t)
    return types

'''
adds a cosine dihedral between all polymer numbers
    with offsets polymer_offsets
    of types particle_types
    with force constant force_const
    and angle [radians]
'''
def add_polymer_dihedral(particle_types1, polymer_offsets1,
                         particle_types2, polymer_offsets2,
                         particle_types3, polymer_offsets3,
                         particle_types4, polymer_offsets4,
                         force_const, angle, system):

    for x in range(1,4):
        for y in range(1,4):

            offsets1 = clamp_polymer_offsets(x, polymer_offsets1)
            offsets2 = clamp_polymer_offsets(x, polymer_offsets2)
            offsets3 = clamp_polymer_offsets(x, polymer_offsets3)
            offsets4 = clamp_polymer_offsets(x, polymer_offsets4)

            ReaddyUtil.add_dihedral(
                get_types_with_polymer_numbers(particle_types1, x, y, offsets1),
                get_types_with_polymer_numbers(particle_types2, x, y, offsets2),
                get_types_with_polymer_numbers(particle_types3, x, y, offsets3),
                get_types_with_polymer_numbers(particle_types4, x, y, offsets4),
                force_const, angle, system
            )

'''
get lists of positions and types for particles in a microtubule
    with n_filaments protofilaments
    and n_rings rings
    and radius [nm]
'''
def get_microtubule_positions_and_types(n_filaments, n_rings, radius):

    positions = []
    types = []
    i = 0
    for filament in range(n_filaments):

        tube_angle = (n_filaments - filament) * 2.*np.pi/n_filaments + np.pi/2.
        normal = ReaddyUtil.normalize(
            np.array([math.cos(tube_angle), math.sin(tube_angle), 0.]))
        tangent = np.array([0., 0., 1.])
        side = ReaddyUtil.normalize(np.cross(normal, tangent))
        pos = radius * normal + (12./13. * filament - 2. * n_rings) * tangent

        for ring in range(n_rings):

            number1 = ring % 3 + 1
            number2 = (filament + math.floor(ring / 3)) % 3 + 1

            positions.append(copy.copy(pos))
            types.append("tubulin{}#{}_{}".format(
                "A" if ring % 2 == 0 else "B", number1, number2))

            if verbose:
                print("{} ({})".format("tubulin{}#{}_{}".format(
                    "A" if ring % 2 == 0 else "B", number1, number2), i))

            pos += 4. * tangent
            i += 1

    return positions, types

'''
add edges to a microtubule topology
    with n_filaments protofilaments
    and n_rings rings
'''
def add_edges(microtubule, n_filaments, n_rings):

    total_particles = n_filaments * n_rings
    for filament in range(n_filaments):
        for ring in range(n_rings):

            i = filament * n_rings + ring

            # bond along filament
            if ring < n_rings-1:

                i_filament = i + 1
                microtubule.get_graph().add_edge(i, i_filament)

            # bond along ring (as long as not in + end overhang)
            if not (filament == n_filaments-1 and ring > n_rings-4):

                i_ring = i + n_rings
                if filament == n_filaments-1:
                    i_ring -= total_particles-3

                microtubule.get_graph().add_edge(i, i_ring)

'''
add seed microtubule to the simulation
    and n_rings rings
    and position_offset
'''
def add_microtubule(n_rings, position_offset, simulation):

    n_filaments = 13
    frayed_angle = np.deg2rad(10.)

    positions, types = get_microtubule_positions_and_types(
        n_filaments, n_rings, 10.86)
    microtubule = simulation.add_topology(
        "Microtubule", types, positions + position_offset)
    add_edges(microtubule, n_filaments, n_rings)

'''
add a kinesin to the simulation
'''
def add_kinesin(position_offset, simulation):

    positions = np.array([
        [0., 3., 0.],
        [0., 3., -5.],
        [0., 0., 4.],
        # [0., 30., 0.]
    ])

    types = [
        "hips",
        "motor#ADP",
        "motor#ADP",
        # "cargo"
    ]

    kinesin = simulation.add_topology(
        "Kinesin", types, positions + position_offset)

    for i in range(1,3):
        kinesin.get_graph().add_edge(0, i)

'''
change the state of a motor and update the kinesin state to match
    for a dictionary of types and radii [nm]

    returns dictionary mapping all types to radii
'''
def set_kinesin_state(topology, recipe, from_motor_state, to_motor_state):

    motors = ReaddyUtil.get_vertices_of_type(topology, "motor", False)
    if len(motors) < 2:
        if verbose:
            print("failed to find 2 motors, found {}".format(len(motors)))
        return None

    motor_types = [topology.particle_type_of_vertex(motors[0]),
                   topology.particle_type_of_vertex(motors[1])]

    other_state = ""
    motors_in_to_state = []
    for i in range(2):
        if from_motor_state in motor_types[i]:
            motors_in_to_state.append(motors[i])
        else:
            other_state = motor_types[i][motor_types[i].index('#')+1:]

    if len(motors_in_to_state) < 1:
        if verbose:
            print("failed to find a motor in state {}".format(from_motor_state))
        return None

    if len(motors_in_to_state) > 1:
        motor_to_set = random.choice(motors_in_to_state)
        other_state = from_motor_state
    else:
        motor_to_set = motors_in_to_state[0]

    recipe.change_particle_type(motor_to_set, "motor#{}".format(to_motor_state))

    if "ADP" in to_motor_state and "ADP" in other_state:
        recipe.change_topology_type("Microtubule-Kinesin#Releasing")
    else:
        new_states = [to_motor_state, other_state]
        new_states.sort()
        recipe.change_topology_type("Microtubule-Kinesin#{}-{}".format(
            new_states[0], new_states[1]))

    return motor_to_set

'''
bind a kinesin motor in ADP state to a free tubulinB
'''
def reaction_function_motor_bind_tubulin(topology):

    if verbose:
        print("(bind tubulin)")

    recipe = readdy.StructuralReactionRecipe(topology)

    motor = set_kinesin_state(topology, recipe, "new", "apo")
    if motor is None:
        raise Error('failed to find motor(s)')

    tubulin = ReaddyUtil.get_neighbor_of_type(
        topology, motor, "tubulinB#bound", False)
    if tubulin is None:
        raise Error('failed to find bound tubulin')
    print("bound to tubulin {}".format(ReaddyUtil.vertex_to_string(topology, tubulin)))
    for neighbor in tubulin:
        print(ReaddyUtil.vertex_to_string(topology, neighbor.get()))

    if verbose:
        print("bind tubulin ------------------------------------------------")
        print("{} ++ {}".format(
            ReaddyUtil.vertex_to_string(topology, motor),
            ReaddyUtil.vertex_to_string(topology, tubulin)))

    tubulin_pos = ReaddyUtil.get_vertex_position(topology, tubulin)
    # TODO calculate position offset from tubulin neighbors
    recipe.change_particle_position(motor, tubulin_pos + (0., 4., 0.))

    return recipe

'''
set bound apo motor's state to ATP (and implicitly simulate ATP binding)
'''
def reaction_function_motor_bind_ATP(topology):

    if verbose:
        print("(bind ATP)")

    recipe = readdy.StructuralReactionRecipe(topology)

    motor = set_kinesin_state(topology, recipe, "apo", "ATP")
    if motor is None:
        raise Error('failed to find motor(s)')

    if verbose:
        print("bind ATP --------------------------------------------------")
        print("{}".format(
            ReaddyUtil.vertex_to_string(topology, motor)))

    return recipe

'''
release a bound motor from tubulin
'''
def reaction_function_motor_release_tubulin(topology):

    if verbose:
        print("(release tubulin)")

    recipe = readdy.StructuralReactionRecipe(topology)

    motor = set_kinesin_state(topology, recipe, "ATP", "ADP")
    if motor is None:
        raise GrowError('failed to find motor(s)')

    tubulin = ReaddyUtil.get_neighbor_of_type(
        topology, motor, "tubulinB#bound", False)
    if tubulin is None:
        raise Error('failed to find bound tubulin')

    if verbose:
        print("release tubulin -----------------------------------------------")
        print("{} -X- {}".format(
            ReaddyUtil.vertex_to_string(topology, motor),
            ReaddyUtil.vertex_to_string(topology, tubulin)))

    # ReaddyUtil.set_flags(topology, recipe, tubulin, [], ["bound"]) # TODO fix bug
    # workaround
    pt = topology.particle_type_of_vertex(tubulin)
    recipe.change_particle_type(tubulin, "tubulinB#{}".format(pt[-3:]))

    removed, message = ReaddyUtil.try_remove_edge(topology, recipe, motor, tubulin)
    if verbose and not removed:
        print(message)

    return recipe

'''
cleanup after releasing a bound motor from tubulin
'''
def reaction_function_cleanup_release_tubulin(topology):

    recipe = readdy.StructuralReactionRecipe(topology)

    motors = ReaddyUtil.get_vertices_of_type(topology, "motor", False)
    if len(motors) > 0:
        recipe.change_topology_type("Kinesin")
    else:
        recipe.change_topology_type("Microtubule")

    if verbose:
        print("cleaned up release tubulin")

    return recipe

'''
rate function for a motor binding ATP
'''
def rate_function_motor_bind_ATP(topology):

    return rates["motor_bind_ATP"]

'''
rate function for a bound motor releasing from tubulin
'''
def rate_function_motor_release_tubulin(topology):

    return rates["motor_release_tubulin"]

class Error(Exception):
    pass

'''
add bonds between tubulins
'''
def add_kinesin_bonds_and_repulsions(motor_types, force_constant, system, util):

    necklinker_force_constant = 0.002 * force_constant
    util.add_bond(["hips"], motor_types, force_constant, 2., system)
    util.add_bond(["hips"], ["cargo"], force_constant, 30., system)

    util.add_repulsion(motor_types, motor_types, force_constant, 4., system)
    util.add_repulsion(["hips"], motor_types, force_constant, 2., system)
    util.add_repulsion(["hips"], ["cargo"], force_constant, 30., system)

'''
add bonds between tubulins
'''
def add_tubulin_bonds_and_repulsions(tubulin_types, force_constant, system, util):

    util.add_polymer_bond_2D( # bonds between protofilaments
        tubulin_types, [0, 0],
        tubulin_types, [0, -1],
        force_constant, 5.2, system
    )
    util.add_polymer_bond_2D( # bonds between rings
        tubulin_types, [0, 0],
        tubulin_types, [-1, 0],
        force_constant, 4., system
    )
    all_tubulin_types = []
    for t in range(len(tubulin_types)):
        all_tubulin_types += get_all_polymer_tubulin_types(tubulin_types[t])
    util.add_repulsion(all_tubulin_types, all_tubulin_types, force_constant, 4., system)

'''
add kinesin angles
'''
def add_kinesin_angles_and_dihedrals(tubulin_types, force_constant, system, util):
    
    # angles from tubulins to bound motor
    util.add_polymer_angle_2D(
        tubulin_types, [-1, 0],
        ["tubulinB#bound_"], [0, 0],
        ["motor#apo", "motor#ATP", "motor#ADP"], [],
        1e32, 0., system
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
    # add_polymer_dihedral(
    #     tubulin_types, [-1, 0],
    #     ["tubulinB#bound_"], [0, 0],
    #     ["motor#ATP"], [],
    #     ["hips"], [],
    #     1.5 * force_constant, np.pi * 17./18., system
    # )
    # add_polymer_dihedral(
    #     tubulin_types, [1, 0],
    #     ["tubulinB#bound_"], [0, 0],
    #     ["motor#ATP"], [],
    #     ["hips"], [],
    #     1.5 * force_constant, np.pi / 18., system
    # )
    # add_polymer_dihedral(
    #     tubulin_types, [0, -1],
    #     ["tubulinB#bound_"], [0, 0],
    #     ["motor#ATP"], [],
    #     ["hips"], [],
    #     1.5 * force_constant, 1.79, system
    # )
    # add_polymer_dihedral(
    #     tubulin_types, [0, 1],
    #     ["tubulinB#bound_"], [0, 0],
    #     ["motor#ATP"], [],
    #     ["hips"], [],
    #     1.5 * force_constant, 1.44, system
    # )

    # # dihedral from bound tubulin to free motor
    # add_polymer_dihedral(
    #     ["tubulinB#bound_"], [0, 0],
    #     ["motor#ATP"], [],
    #     ["hips"], [],
    #     ["motor#ADP"], [],
    #     0.5 * force_constant, np.pi * 4./9., system
    # )

'''
add angles between tubulins
'''
def add_angles_between_tubulins(tubulin_types, force_constant, system, util):

    util.add_polymer_angle_2D(
        tubulin_types, [0, 1],
        tubulin_types, [0, 0],
        tubulin_types, [-1, 0],
        force_constant, 1.75, system
    )
    util.add_polymer_angle_2D(
        tubulin_types, [0, 1],
        tubulin_types, [0, 0],
        tubulin_types, [1, 0],
        force_constant, 1.40, system
    )
    util.add_polymer_angle_2D(
        tubulin_types, [0, -1],
        tubulin_types, [0, 0],
        tubulin_types, [-1, 0],
        force_constant, 1.40, system
    )
    util.add_polymer_angle_2D(
        tubulin_types, [0, -1],
        tubulin_types, [0, 0],
        tubulin_types, [1, 0],
        force_constant, 1.75, system
    )
    util.add_polymer_angle_2D(
        tubulin_types, [-1, 0],
        tubulin_types, [0, 0],
        tubulin_types, [1, 0],
        force_constant, np.pi, system
    )
    util.add_polymer_angle_2D(
        tubulin_types, [0, -1],
        tubulin_types, [0, 0],
        tubulin_types, [0, 1],
        force_constant, 2.67, system
    )

'''
add repulsions between motors and tubulins
'''
def add_motor_tubulin_interactions(
    motor_types, bound_tubulin_types, tubulin_types, force_constant, system, util):

    bound_types = []
    for t in bound_tubulin_types:
        bound_types += get_all_polymer_tubulin_types(t)

    all_types = []
    for t in tubulin_types:
        all_types += get_all_polymer_tubulin_types(t)

    util.add_bond(motor_types, bound_types, force_constant, 4., system)
    util.add_repulsion(motor_types, all_types, force_constant, 3., system)

'''
bind a kinesin motor in ADP state to a free tubulinB
'''
def add_motor_bind_tubulin_reaction(system, rate, reaction_distance):

    # spatial reactions
    polymer_numbers = get_all_polymer_tubulin_types("")
    kinesin_states = ["ADP-apo", "ADP-ATP"]
    i = 1
    for n in polymer_numbers:
        # first motor binding
        system.topologies.add_spatial_reaction(
            "Bind_Tubulin#ADP-ADP{}: Kinesin(motor#ADP) + Microtubule(tubulinB#{}) -> \
            Microtubule-Kinesin#Binding(motor#new--tubulinB#bound_{})".format(i, n, n),
            rate=rate, radius=4.+reaction_distance
        )
        # # second motor binding
        # for s in kinesin_states:
        #     system.topologies.add_spatial_reaction(
        #         "Bind_Tubulin#{}{}: {}".format(s, i,
        #         "Microtubule-Kinesin#{}(motor#ADP) + {}".format(s,
        #         "Microtubule-Kinesin#{}(tubulinB#{}) -> {}".format(s, n,
        #         "Microtubule-Kinesin#Binding(motor#new--{}".format(
        #         "tubulinB#bound_{}) [self=true]".format(n))))),
        #         rate=rate, radius=4.+reaction_distance
        #     )
        i += 1

    # structural reaction
    system.topologies.add_structural_reaction(
        "Finish_Bind_Tubulin",
        topology_type="Microtubule-Kinesin#Binding",
        reaction_function=reaction_function_motor_bind_tubulin,
        rate_function=ReaddyUtil.rate_function_infinity
    )

'''
set bound apo motor's state to ATP (and implicitly simulate ATP binding)
'''
def add_motor_bind_ATP_reaction(system):

    kinesin_states = ["ADP-apo", "ATP-apo", "apo-apo"]
    for state in kinesin_states:
        system.topologies.add_structural_reaction(
            "Bind_ATP#{}".format(state),
            topology_type="Microtubule-Kinesin#{}".format(state),
            reaction_function=reaction_function_motor_bind_ATP,
            rate_function=rate_function_motor_bind_ATP
        )

'''
release a bound motor from tubulin
'''
def add_motor_release_tubulin_reaction(system):

    kinesin_states = ["ADP-ATP", "ATP-ATP", "ATP-apo"]
    for state in kinesin_states:
        system.topologies.add_structural_reaction(
            "Release_Tubulin#{}".format(state),
            topology_type="Microtubule-Kinesin#{}".format(state),
            reaction_function=reaction_function_motor_release_tubulin,
            rate_function=rate_function_motor_release_tubulin
        )

    system.topologies.add_structural_reaction(
        "Cleanup_Release_Tubulin",
        topology_type="Microtubule-Kinesin#Releasing",
        reaction_function=reaction_function_cleanup_release_tubulin,
        rate_function=ReaddyUtil.rate_function_infinity
    )

'''
report necklinker lengths
'''
necklinker_lengths = []
def reaction_function_report_necklinker_length(topology):

    global necklinker_lengths

    recipe = readdy.StructuralReactionRecipe(topology)
    
    v_hips = ReaddyUtil.get_vertex_of_type(topology, "hips", True)
    v_motors = ReaddyUtil.get_neighbors_of_type(topology, v_hips, "motor", False)

    if v_hips is None or len(v_motors) < 2:
        print("failed to find hips and motors")
        return recipe

    pos_hips = ReaddyUtil.get_vertex_position(topology, v_hips)
    pos_motor1 = ReaddyUtil.get_vertex_position(topology, v_motors[0])
    pos_motor2 = ReaddyUtil.get_vertex_position(topology, v_motors[1])

    necklinker_length1 = np.linalg.norm(pos_motor1 - pos_hips)
    necklinker_length2 = np.linalg.norm(pos_motor2 - pos_hips)

    necklinker_lengths.append(necklinker_length1)
    necklinker_lengths.append(necklinker_length2)

    # print("necklinker lengths = {} and {} nm".format(necklinker_length1, necklinker_length2))

    return recipe

'''
add reporter for necklinker length
'''
def add_necklinker_length_reporter(system):

    system.topologies.add_structural_reaction(
        "Report_Necklinker_Length",
        topology_type="Kinesin",
        reaction_function=reaction_function_report_necklinker_length,
        rate_function=ReaddyUtil.rate_function_infinity
    )

'''
plot necklinker lengths
'''
def plot_necklinker_lengths():

    fig, ax = plt.subplots(1, 1)
    ax.hist(necklinker_lengths, density=1, bins=10) 
    plt.savefig("necklinker_lengths.png", dpi=220)
    plt.close()

    print(
        "necklinker lengths: median = {}, mean = {}, std = {}".format(
            np.median(necklinker_lengths), 
            np.mean(necklinker_lengths), 
            np.std(necklinker_lengths)
        )
    )
