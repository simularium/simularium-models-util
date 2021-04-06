
# coding: utf-8

# In[1]:


import numpy as np
import readdy
import os
import math
import random
import time
import sys
from decimal import Decimal

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


readdy.__version__


# ## Load the Trajectory

# In[3]:


traj = readdy.Trajectory("cluster_runs/actin34_0.h5")

times, topology_records = traj.read_observable_topologies()
times, types, ids, positions = traj.read_observable_particles()
reaction_times, reactions = traj.read_observable_reaction_counts()


# In[4]:


box_size = 300. #nm
min_time = 0
max_time = 101
observe_stride = 5e5
time_inc = 1

total_sim_time = int(1e6) * 0.005 * 1e-9 # seconds

print(len(times))
print(len(reaction_times))


# In[5]:


# ----------------------------------------------------------- General Processing / Helpers

'''
get all the edges at the given time index as (particle1 id, particle2 id)
'''
def get_edges(observe_time):
    
    result = []
    for top in topology_records[observe_time]:
        for e1, e2 in top.edges:
            if e1 <= e2:
                ix1 = top.particles[e1]
                ix2 = top.particles[e2]
                result.append((ix1, ix2))
                
    return result


'''
get a dictionary with 
keys = particle id
values = (particle type, list of neighbor particle ids, particle position)
for each particle at the given time index
'''
def get_particle_info_by_id(observe_time):
    
    edges = get_edges(observe_time)
    
    result = {}
    for p in range(len(ids[observe_time])):
        
        p_id = ids[observe_time][p]
        p_type = traj.species_name(types[observe_time][p])
        p_pos = positions[observe_time][p]
        
        neighbor_ids = []
        for edge in edges:
            if p_id == edge[0]:
                neighbor_ids.append(edge[1])
            elif p_id == edge[1]:
                neighbor_ids.append(edge[0])
            
        result[p_id] = (p_type, neighbor_ids, np.array([p_pos[0], p_pos[1], p_pos[2]]))
        
    return result


# ### Shape the data for calculating observables

# In[6]:


particle_info = []
for t in range(len(times)):
    if t >= min_time and t <= max_time and t % time_inc == 0:
        
        particle_info.append(get_particle_info_by_id(t))
    
        sys.stdout.write('\r')
        p = 100. * (t + 1) / float(max_time - min_time)
        sys.stdout.write("[{}{}] {}%".format('='*int(round(p)), ' '*int(100. - round(p)), round(10. * p) / 10.))
        sys.stdout.flush()


# ## Calculate Observables

# In[7]:


# ----------------------------------------------------------- Helpers

'''
normalize a numpy vector
'''
def normalize(v):
    
    return v / np.linalg.norm(v)


'''
check if any of a 3D vector's components are NaN
'''
def vector_is_invalid(v):
    
    return math.isnan(v[0]) or math.isnan(v[1]) or math.isnan(v[2])


'''
get the angle between two vectors in degrees
'''
def get_angle_between_vectors(v1, v2):
    
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2), -1., 1.)))


'''
get the number of topologies of a given type at the given time index
'''
def get_count_of_topologies_of_type(topology_type, observe_time):
    
    result = 0
    for top in topology_records[observe_time]:
        if traj.topology_type_name(top.type) == topology_type:
            result += 1
            
    return result
        

'''
get a list of ids for all the particles with particle type in the given list of types
'''
def get_ids_for_types(particle_info_frame, vertex_types):
    
    result = []
    for p_id in particle_info_frame:
        if particle_info_frame[p_id][0] in vertex_types:
            result.append(p_id)
            
    return result
   

'''
get the id for the first neighbor with one of the neighbor_types
'''
def get_id_for_neighbor_of_types(particle_info_frame, particle_id, neighbor_types, exclude_ids):
    
    for n_id in particle_info_frame[particle_id][1]:
        
        if n_id in exclude_ids:
            continue
            
        nt = particle_info_frame[n_id][0]
        
        if nt in neighbor_types:
            return n_id
        
    return None
        
'''
get ids for a chain of particles with neighbor_types from the particle with particle_id
if chain_length = 0, return entire chain
'''
def get_chain_of_types(particle_info_frame, particle_id, neighbor_types, chain_length, last_particle_id, result):
    
    n_id = get_id_for_neighbor_of_types(particle_info_frame, particle_id, neighbor_types, [last_particle_id])
    
    if n_id is None:
        return result
    
    result.append(n_id)
    
    if chain_length == 1:
        return result
    
    return get_chain_of_types(particle_info_frame, n_id, neighbor_types, 
                              chain_length-1 if chain_length > 0 else 0, 
                              particle_id, result)

'''
get ids for the first actins at the pointed end of a branch
'''
def get_first_branch_actin_ids(particle_info_frame):
    
    branch_actin_types = ["actin#branch_1", "actin#branch_2", "actin#branch_3", 
                          "actin#branch_ATP_1", "actin#branch_ATP_2", "actin#branch_ATP_3", 
                          "actin#branch_barbed_1", "actin#branch_barbed_2", "actin#branch_barbed_3", 
                          "actin#branch_barbed_ATP_1", "actin#branch_barbed_ATP_2", "actin#branch_barbed_ATP_3"]
    
    return get_ids_for_types(particle_info_frame, branch_actin_types)


'''
get a list of filaments, each filament is a list of the actin ids in the filament 
in order from pointed to barbed end
'''
def get_actin_ids_by_filament(particle_info_frame):
    
    result = []
    
    pointed_types = ["actin#pointed_1", "actin#pointed_2", "actin#pointed_3", 
                     "actin#pointed_ATP_1", "actin#pointed_ATP_2", "actin#pointed_ATP_3"]
    actin_types = ["actin#1", "actin#2", "actin#3", "actin#ATP_1", "actin#ATP_2", "actin#ATP_3",
                   "actin#barbed_1", "actin#barbed_2", "actin#barbed_3", 
                   "actin#barbed_ATP_1", "actin#barbed_ATP_2", "actin#barbed_ATP_3"]
    
    start_actin_ids = (get_ids_for_types(particle_info_frame, pointed_types) 
                       + get_first_branch_actin_ids(particle_info_frame))
    
    for start_actin_id in start_actin_ids:
        result.append(get_chain_of_types(particle_info_frame, start_actin_id, actin_types, 
                                         0, None, [start_actin_id]))
        
    return result


'''
format subplots
'''
def setup_plot(total_plots, title, figure, index):
    
    subplot = figure.add_subplot(int(math.ceil(total_plots / 2.)), 2, index + 1)
    subplot.title.set_text(title)
    return subplot, index + 1


# ----------------------------------------------------------- [filamentous actin] / [total actin] vs time

'''
get a list of the ratio of actin in filaments to total actin over time
'''
def get_ratio_of_filamentous_to_total_actin():
    
    free_actin_types = ["actin#free", "actin#free_ATP"]
    filamentous_actin_types = ["actin#1", "actin#2", "actin#3", "actin#ATP_1", "actin#ATP_2", "actin#ATP_3", 
                               "actin#pointed_1", "actin#pointed_2", "actin#pointed_3", 
                               "actin#pointed_ATP_1", "actin#pointed_ATP_2", "actin#pointed_ATP_3",
                               "actin#barbed_1", "actin#barbed_2", "actin#barbed_3", 
                               "actin#barbed_ATP_1", "actin#barbed_ATP_2", "actin#barbed_ATP_3", 
                               "actin#branch_1", "actin#branch_2", "actin#branch_3", 
                               "actin#branch_ATP_1", "actin#branch_ATP_2", "actin#branch_ATP_3", 
                               "actin#branch_barbed_1", "actin#branch_barbed_2", "actin#branch_barbed_3", 
                               "actin#branch_barbed_ATP_1", "actin#branch_barbed_ATP_2", 
                               "actin#branch_barbed_ATP_3"]
    result = []
    for t in range(len(particle_info)):
            
        free_actin = len(get_ids_for_types(particle_info[t], free_actin_types))
        filamentous_actin = len(get_ids_for_types(particle_info[t], filamentous_actin_types))

        if free_actin + filamentous_actin > 0:
            result.append(filamentous_actin / float(free_actin + filamentous_actin))
        else:
            result.append(0)
      
    return result


'''
plot [filamentous actin] / [total actin] vs time 
'''
def observe_ratio_of_filamentous_to_total_actin(figure):
    
    global current_figure
    result = get_ratio_of_filamentous_to_total_actin()
    subplot, current_figure = setup_plot("[filamentous actin] / [total actin] vs time", figure, current_figure)
    subplot.plot(times[min_time:max_time+1:time_inc], result)

# ------------------------------------------------ [ATP-actin] / [total actin] in filaments vs time

'''
get a list of the ratio of ATP-actin to total actin in filaments over time
'''
def get_ratio_of_ATP_actin_to_total_actin():
    
    ATP_actin_types = ["actin#ATP_1", "actin#ATP_2", "actin#ATP_3", 
                       "actin#pointed_ATP_1", "actin#pointed_ATP_2", "actin#pointed_ATP_3",
                       "actin#barbed_ATP_1", "actin#barbed_ATP_2", "actin#barbed_ATP_3", 
                       "actin#branch_ATP_1", "actin#branch_ATP_2", "actin#branch_ATP_3", 
                       "actin#branch_barbed_ATP_1", "actin#branch_barbed_ATP_2", "actin#branch_barbed_ATP_3"]
    ADP_actin_types = ["actin#1", "actin#2", "actin#3", "actin#pointed_1", "actin#pointed_2", "actin#pointed_3", 
                       "actin#barbed_1", "actin#barbed_2", "actin#barbed_3", 
                       "actin#branch_1", "actin#branch_2", "actin#branch_3", 
                       "actin#branch_barbed_1", "actin#branch_barbed_2", "actin#branch_barbed_3"]
    result = []
    for t in range(len(particle_info)):
            
        ATP_actin = len(get_ids_for_types(particle_info[t], ATP_actin_types))
        ADP_actin = len(get_ids_for_types(particle_info[t], ADP_actin_types))

        if ADP_actin + ATP_actin > 0:
            result.append(ATP_actin / float(ADP_actin + ATP_actin))
        else:
            result.append(1.)
        
    return result


'''
plot [ATP-actin] / [total actin] in filaments vs time
'''
def observe_ratio_of_ATP_actin_to_total_actin(figure):
    
    global current_figure
    result = get_ratio_of_ATP_actin_to_total_actin()
    subplot, current_figure = setup_plot("[ATP-actin] / [total actin] in filaments vs time", 
                                         figure, current_figure)
    subplot.plot(times[min_time:max_time+1:time_inc], result)
    

# ------------------------------------------------ [daughter filament actin] / [filamentous actin] vs time 

'''
get a list of mother filaments at each time point, each filament is a list of ids for 
actins in the chain, mother filaments = (pointed end --> barbed end)
'''
def get_mother_filaments():
    
    result = []
    actin_types = ["actin#1", "actin#2", "actin#3", "actin#ATP_1", "actin#ATP_2", "actin#ATP_3",
                   "actin#pointed_1", "actin#pointed_2", "actin#pointed_3", 
                   "actin#pointed_ATP_1", "actin#pointed_ATP_2", "actin#pointed_ATP_3",
                   "actin#barbed_1", "actin#barbed_2", "actin#barbed_3", 
                   "actin#barbed_ATP_1", "actin#barbed_ATP_2", "actin#barbed_ATP_3", 
                   "actin#branch_1", "actin#branch_2", "actin#branch_3", 
                   "actin#branch_ATP_1", "actin#branch_ATP_2", "actin#branch_ATP_3", 
                   "actin#branch_barbed_1", "actin#branch_barbed_2", "actin#branch_barbed_3", 
                   "actin#branch_barbed_ATP_1", "actin#branch_barbed_ATP_2", "actin#branch_barbed_ATP_3"]
    pointed_types = ["actin#pointed_1", "actin#pointed_2", "actin#pointed_3", 
                     "actin#pointed_ATP_1", "actin#pointed_ATP_2", "actin#pointed_ATP_3"]
    
    for t in range(len(particle_info)):
        
        result.append([])
        pointed_end_ids = get_ids_for_types(particle_info[t], pointed_types)
        for pointed_end_id in pointed_end_ids:
            result[t].append(get_chain_of_types(particle_info[t], pointed_end_id, actin_types, 
                                                0, None, [pointed_end_id]))
        
    return result


mother_filaments = get_mother_filaments()


'''
get a list of daughter filaments at each time point, each filament is a list of ids for 
actins in the chain, daughter filaments = (arp2/3 --> barbed end)
'''
def get_daughter_filaments():
    
    result = []
    actin_types = ["actin#1", "actin#2", "actin#3", "actin#ATP_1", "actin#ATP_2", "actin#ATP_3",
                   "actin#pointed_1", "actin#pointed_2", "actin#pointed_3", 
                   "actin#pointed_ATP_1", "actin#pointed_ATP_2", "actin#pointed_ATP_3",
                   "actin#barbed_1", "actin#barbed_2", "actin#barbed_3", 
                   "actin#barbed_ATP_1", "actin#barbed_ATP_2", "actin#barbed_ATP_3", 
                   "actin#branch_1", "actin#branch_2", "actin#branch_3", 
                   "actin#branch_ATP_1", "actin#branch_ATP_2", "actin#branch_ATP_3", 
                   "actin#branch_barbed_1", "actin#branch_barbed_2", "actin#branch_barbed_3", 
                   "actin#branch_barbed_ATP_1", "actin#branch_barbed_ATP_2", "actin#branch_barbed_ATP_3"]

    for t in range(len(particle_info)):
    
        result.append([])
        first_actin_ids = get_first_branch_actin_ids(particle_info[t])

        for first_actin_id in first_actin_ids:

            result[t].append(get_chain_of_types(particle_info[t], first_actin_id, actin_types, 
                                                0, None, [first_actin_id]))
        
    return result


daughter_filaments = get_daughter_filaments()


'''
plot [daughter filament actin] / [filamentous actin] vs time 
'''
def observe_ratio_of_daughter_filament_actin_to_total_filamentous_actin(figure):
    
    global current_figure
    result = []
    for t in range(len(particle_info)):
        
        mother_actin = 0
        for mother_filament in mother_filaments[t]:
            mother_actin += len(mother_filament)
            
        daughter_actin = 0
        for daughter_filament in daughter_filaments[t]:
            daughter_actin += len(daughter_filament)
        
        if mother_actin + daughter_actin > 0:
            result.append(daughter_actin / float(mother_actin + daughter_actin))
        else:
            result.append(0)
    
    subplot, current_figure = setup_plot("[daughter filament actin] / [filamentous actin] vs time", 
                                         figure, current_figure)
    subplot.plot(times[min_time:max_time+1:time_inc], result)


# ----------------------------------------------------------- length of mother filaments (distribution vs time)
    
'''
plot length of mother filaments
'''
def observe_mother_filament_lengths(figure):
    
    global current_figure
    result = []
    result_times = []
    for t in range(len(particle_info)):
        
        for filament in mother_filaments[t]:
            result.append(len(filament))
            result_times.append(t)
       
    if len(result) > 0:
        subplot, current_figure = setup_plot("length of mother filaments", figure, current_figure)
        subplot.scatter(result_times, result)
    
    
# ----------------------------------------------------------- length of daughter filaments (distribution vs time)

'''
plot length of daughter filaments
'''
def observe_daughter_filament_lengths(figure):
    
    global current_figure
    result = []
    result_times = []
    for t in range(len(particle_info)):
        
        for filament in daughter_filaments[t]:
            result.append(len(filament))
            result_times.append(t)
        
    if len(result) > 0:
        subplot, current_figure = setup_plot("length of daughter filaments", figure, current_figure)
        subplot.scatter(result_times, result)

    
# ----------------------------------------------------------- [bound arp2/3] / [total arp2/3] vs time

'''
get a list of the ratio of bound to total arp2/3 complexes over time
'''
def get_ratio_of_bound_to_total_arp23():
    
    result = []
    for t in range(len(particle_info)):
        
        bound_arp23 = 0
        free_arp23 = 0
        arp3_ids = get_ids_for_types(particle_info[t], ["arp3", "arp3#branched"])
        for arp3_id in arp3_ids:
            if len(particle_info[t][arp3_id][1]) > 1:
                bound_arp23 += 1
            if len(particle_info[t][arp3_id][1]) <= 1:
                free_arp23 += 1

        if free_arp23 + bound_arp23 > 0:
            result.append(bound_arp23 / float(free_arp23 + bound_arp23))
        else:
            result.append(0)
        
    return result


'''
plot [bound arp2/3] / [total arp2/3] vs time 
'''
def observe_ratio_of_bound_to_total_arp23(figure):
    
    global current_figure
    result = get_ratio_of_bound_to_total_arp23()
    subplot, current_figure = setup_plot("[bound arp2/3] / [total arp2/3] vs time", figure, current_figure)
    subplot.plot(times[min_time:max_time+1:time_inc], result)

    
# ----------------------------------------------------------- [capped ends] / [total ends] vs time

'''
get a list of the ratio of barbed ends capped with capping protein to all barbed ends
'''
def get_ratio_of_capped_ends_to_total_ends():
    
    capped_end_types = ["cap#bound"]
    growing_end_types = ["actin#barbed_1", "actin#barbed_2", "actin#barbed_3", 
                         "actin#barbed_ATP_1", "actin#barbed_ATP_2", "actin#barbed_ATP_3", 
                         "actin#branch_barbed_1", "actin#branch_barbed_2", "actin#branch_barbed_3", 
                         "actin#branch_barbed_ATP_1", "actin#branch_barbed_ATP_2", "actin#branch_barbed_ATP_3"]
    
    result = []
    for t in range(len(particle_info)):
        
        capped_ends = len(get_ids_for_types(particle_info[t], capped_end_types))
        growing_ends = len(get_ids_for_types(particle_info[t], growing_end_types))

        if growing_ends + capped_ends > 0:
            result.append(capped_ends / float(growing_ends + capped_ends))
        else:
            result.append(0)
        
    return result


'''
plot [capped ends] / [total ends] vs time
'''
def observe_ratio_of_capped_ends_to_total_ends(figure):
    
    global current_figure
    result = get_ratio_of_capped_ends_to_total_ends()
    subplot, current_figure = setup_plot("[capped ends] / [total ends] vs time", figure, current_figure)
    subplot.plot(times[min_time:max_time+1:time_inc], result)
    
    
# ----------------------------------------------------------- branch angles (histogram)

'''
orthonormalize and cross the vectors to an actin's two neighbor actins to get a basis local to the actin,
positions = [previous actin position, this actin position, next actin position]
'''
def get_actin_orientation(positions):

    v1 = normalize(positions[0] - positions[1])
    v2 = normalize(positions[2] - positions[1])
    v2 = normalize(v2 - (np.dot(v1, v2) / np.dot(v1, v1)) * v1)
    v3 = np.cross(v2, v1)

    return np.matrix([[v1[0], v2[0], v3[0]],
                      [v1[1], v2[1], v3[1]],
                      [v1[2], v2[2], v3[2]]])


# from unity positions
initial_positions = [np.array([0.,        0.,         0.]), 
                     np.array([1.307011, -2.901852,  -2.849991]), 
                     np.array([0.7042618, 0.2231636, -5.699982])]
vector_to_axis = np.array([0.8138123, -1.358344, -2.849991]) - initial_positions[1]
intial_orientation = get_actin_orientation(initial_positions)


'''
get the difference in the actin's current orientation 
compared to the initial orientation as a rotation matrix
'''
def get_actin_rotation(particle_info_frame, actin_ids):
    
    prev_position = particle_info_frame[actin_ids[0]][2]
    position = particle_info_frame[actin_ids[1]][2]
    next_position = particle_info_frame[actin_ids[2]][2]
    
    if (np.linalg.norm(prev_position - position) > box_size / 2 or
        np.linalg.norm(next_position - position) > box_size): # make sure we didn't cross periodic boundary
        return None
    
    current_orientation = get_actin_orientation([prev_position, position, next_position])
    
    return np.matmul(current_orientation, np.linalg.inv(intial_orientation))


'''
get the position on the filament axis closest to an actin
actin_ids = [previous actin id, this actin id, next actin id]
'''
def get_axis_position_for_actin(particle_info_frame, actin_ids):
    
    rotation = get_actin_rotation(particle_info_frame, actin_ids)
    if rotation is None:
        return None
        
    vector_to_axis_local = np.squeeze(np.array(np.dot(rotation, vector_to_axis)))
    
    return particle_info_frame[actin_ids[1]][2] + vector_to_axis_local

'''
get the type number for an actin plus the given offset in range [-1, 1]
(i.e. return 3 for type = "actin#ATP_1" and offset = -1)
'''
def get_actin_number(vertex_type, offset):
    
    if not "actin" in vertex_type:
        raise Error("Failed to get actin number: {} is not actin".format(vertex_type))
    
    n = int(vertex_type[-1:]) + offset
    if n > 3:
        n -= 3
    if n < 1:
        n += 3    
    
    return n

'''
for each branch point, get list of ids for (in order):
- [0,1,2,3] 4 actins after branch on main filament (toward barbed end, ordered from pointed end toward barbed end)
- [4,5,6,7] first 4 actins on branch (ordered from pointed end toward barbed end)
'''    
def get_branch_ids(particle_info_frame, verbose):
    
    arp3_ids = get_ids_for_types(particle_info_frame, ["arp3#branched"])
    actin_types = ["actin#1", "actin#2", "actin#3", "actin#ATP_1", "actin#ATP_2", "actin#ATP_3",
                   "actin#pointed_1", "actin#pointed_2", "actin#pointed_3", 
                   "actin#pointed_ATP_1", "actin#pointed_ATP_2", "actin#pointed_ATP_3",
                   "actin#barbed_1", "actin#barbed_2", "actin#barbed_3", 
                   "actin#barbed_ATP_1", "actin#barbed_ATP_2", "actin#barbed_ATP_3"]
    branch_actin_types = ["actin#branch_1", "actin#branch_2", "actin#branch_3", 
                   "actin#branch_ATP_1", "actin#branch_ATP_2", "actin#branch_ATP_3", 
                   "actin#branch_barbed_1", "actin#branch_barbed_2", "actin#branch_barbed_3", 
                   "actin#branch_barbed_ATP_1", "actin#branch_barbed_ATP_2", "actin#branch_barbed_ATP_3"]
    
    result = []
    for arp3_id in arp3_ids:
            
        actin1_id = get_id_for_neighbor_of_types(particle_info_frame, arp3_id, branch_actin_types, [])
        if actin1_id is None:
            if verbose:
                print("couldn't parse branch point: failed to find actin1_id")
            continue
            
        actin_arp3_id = get_id_for_neighbor_of_types(particle_info_frame, arp3_id, actin_types, [actin1_id])
        if actin_arp3_id is None:
            if verbose:
                print("couldn't parse branch point: failed to find actin_arp3_id")
            continue
            
        branch_actins = get_chain_of_types(particle_info_frame, actin1_id, actin_types, 3, arp3_id, [actin1_id])  
        if len(branch_actins) < 4:
            if verbose:
                print("couldn't parse branch point: only found {} branch actins".format(len(branch_actins)))
            continue
            
        n = get_actin_number(particle_info_frame[actin_arp3_id][0], -1)
        actin_arp2_types = ["actin#{}".format(n), "actin#ATP_{}".format(n),
                            "actin#pointed_{}".format(n), "actin#pointed_ATP_{}".format(n),
                            "actin#branch_{}".format(n), "actin#branch_ATP_{}".format(n)]
            
        actin_arp2_id = get_id_for_neighbor_of_types(particle_info_frame, actin_arp3_id, actin_arp2_types, [])
        if actin_arp2_id is None:
            if verbose:
                print("couldn't parse branch point: failed to find actin_arp2_id")
            continue
        
        main_actins = get_chain_of_types(particle_info_frame, actin_arp3_id, actin_types, 2, actin_arp2_id, [actin_arp2_id, actin_arp3_id])
        if len(main_actins) < 4:
            if verbose:
                print("couldn't parse branch point: only found {} main actins".format(len(main_actins)))
            continue
        
        result.append(main_actins + branch_actins)
        
    return result


'''
get the angle between mother and daughter filament at each branch point
'''
def get_branch_angles(particle_info_frame):
    
    branch_ids = get_branch_ids(particle_info_frame, False)
    result = []
    for branch in branch_ids:
        
        main_pos1 = get_axis_position_for_actin(particle_info_frame, [branch[0], branch[1], branch[2]])
        if main_pos1 is None or vector_is_invalid(main_pos1):
            continue
        
        main_pos2 = get_axis_position_for_actin(particle_info_frame, [branch[1], branch[2], branch[3]])
        if main_pos2 is None or vector_is_invalid(main_pos2):
            continue
        
        v_main = normalize(main_pos2 - main_pos1)
        
        branch_pos1 = get_axis_position_for_actin(particle_info_frame, [branch[4], branch[5], branch[6]])
        if branch_pos1 is None or vector_is_invalid(branch_pos1):
            continue
            
        branch_pos2 = get_axis_position_for_actin(particle_info_frame, [branch[5], branch[6], branch[7]])
        if branch_pos2 is None or vector_is_invalid(branch_pos2):
            continue
            
        v_branch = normalize(branch_pos2 - branch_pos1)
        
        result.append(get_angle_between_vectors(v_main, v_branch))
        
    return result
        
    
'''
plot branch angles
'''
def observe_branching_angles(figure):
    
    global current_figure
    result = []
    for t in range(len(particle_info)):
        result += get_branch_angles(particle_info[t])
    
    if len(result) > 0:
        subplot, current_figure = setup_plot("branch angles", figure, current_figure)
        subplot.hist(result, density=1, bins=10) 
        print("BRANCH ANGLE ----------\nideal angle = 95.33\nmedian angle = {}\nmean angle = {}\nstd = {}\n".format(
            np.median(result), np.mean(result), np.std(result)))
        
        
# ----------------------------------------------------------- pitch of actin helix (histogram)

'''
calculate the pitch of the helix between two actins
actin_ids = [previous actin id, this actin id, next actin id] for each of the two actins
'''
def calculate_pitch(particle_info_frame, actin1_ids, actin2_ids):
    
    actin1_pos = particle_info_frame[actin1_ids[1]][2]
    actin1_axis_pos = get_axis_position_for_actin(particle_info_frame, actin1_ids)
    if actin1_axis_pos is None or vector_is_invalid(actin1_axis_pos):
        return None
            
    v1 = normalize(actin1_axis_pos - actin1_pos)
    
    actin2_pos = particle_info_frame[actin2_ids[1]][2]
    actin2_axis_pos = get_axis_position_for_actin(particle_info_frame, actin2_ids)
    if actin2_axis_pos is None or vector_is_invalid(actin2_axis_pos):
        return None
    
    v2 = normalize(actin2_axis_pos - actin2_pos)
    
    length = np.linalg.norm(actin2_axis_pos - actin1_axis_pos)
    if length > box_size / 2: # make sure we didn't cross periodic boundary
        return None
    
    angle = get_angle_between_vectors(v1, v2)
    
    return (360. / angle) * length
    

'''
get the pitch of the short and long helix between all actins on each filament
'''
def get_helix_pitches(particle_info_frame):
    
    short_pitches = []
    long_pitches = []
    
    filaments = get_actin_ids_by_filament(particle_info_frame)
    
    for actin_ids in filaments:
        
        for i in range(1, len(actin_ids)-3):
        
            short_pitch = calculate_pitch(particle_info_frame,
                                          [actin_ids[i-1], actin_ids[i], actin_ids[i+1]], 
                                          [actin_ids[i], actin_ids[i+1], actin_ids[i+2]])
            if short_pitch is not None:
                short_pitches.append(short_pitch)
                
            long_pitch = calculate_pitch(particle_info_frame, 
                                         [actin_ids[i-1], actin_ids[i], actin_ids[i+1]],
                                         [actin_ids[i+1], actin_ids[i+2], actin_ids[i+3]])
            if long_pitch is not None:
                long_pitches.append(long_pitch)
    
    return (short_pitches, long_pitches)   


'''
plot short and long helix pitch
ideal = short: 5.9nm, long: 72nm (Ref: http://www.jbc.org/content/266/1/1.full.pdf)
'''
def observe_helix_pitches(figure):
    
    global current_figure
    short_pitches = []
    long_pitches = []
    for t in range(len(particle_info)):
        
        pitches = get_helix_pitches(particle_info[t])
        short_pitches += pitches[0]
        long_pitches += pitches[1]
    
    if len(short_pitches) > 0:
        subplot, current_figure = setup_plot("short helix pitch", figure, current_figure)
        subplot.hist(short_pitches, density=1, bins=20) 
        print("HELIX PITCH ----------\nideal short pitch = 5.9\nmedian short pitch = {}\nmean short pitch = {}\nstd = {}".format(
            np.median(short_pitches), np.mean(short_pitches), np.std(short_pitches)))
    
    if len(long_pitches) > 0:
        subplot, current_figure = setup_plot("long helix pitch", figure, current_figure)
        subplot.hist(long_pitches, density=1, bins=20) 
        print("ideal long pitch = 72\nmedian long pitch = {}\nmean long pitch = {}\nstd = {}\n".format(
            np.median(long_pitches), np.mean(long_pitches), np.std(long_pitches)))


# ----------------------------------------------------------- filament straightness (histogram)

'''
use singular value decomposition (first PCA component) to calculate a best fit vector along the filament
'''
def calculate_line(positions, length):
    
    center = np.mean(positions, axis=0)
    uu, dd, vv = np.linalg.svd(positions - center)
    
    return np.array([center - (length / 2.) * vv[0], center + (length / 2.) * vv[0]])


'''
get the point on the line closest to the given point
'''
def get_closest_point_on_line(line, point):
    
    lineDir = normalize(line[1] - line[0])
    v = point - line[0]
    d = np.dot(v, lineDir)
    
    return line[0] + d * lineDir
    

'''
get the distance from each actin axis position to the ideal axis position 
if the filament axis was a straight line
'''
def get_distance_from_straight(particle_info_frame):
    
    result = []
    filaments = get_actin_ids_by_filament(particle_info_frame)
    
    for actin_ids in filaments:
        
        positions = []
        last_pos = particle_info_frame[actin_ids[0]][2]
        
        for i in range(1,len(actin_ids)-1): 
            axis_pos = get_axis_position_for_actin(particle_info_frame, 
                                                   [actin_ids[i-1], actin_ids[i], actin_ids[i+1]])
            if axis_pos is None or vector_is_invalid(axis_pos):
                break
            
            if np.linalg.norm(axis_pos - last_pos) < box_size / 2.: # make sure we didn't cross periodic boundary
                positions.append(axis_pos)
                
            last_pos = axis_pos
        
        if len(positions) > 2:
        
            axis = calculate_line(np.squeeze(np.array(positions)), box_size)

            for pos in positions:
                line_pos = get_closest_point_on_line(axis, pos)
                result.append(np.linalg.norm(line_pos - pos))
        
    return result


'''
plot filament straightness
'''    
def observe_filament_straightness(figure):
    
    global current_figure
    result = []
    for t in range(len(particle_info)):
        result += get_distance_from_straight(particle_info[t])
          
    if len(result) > 0:
        
        subplot, current_figure = setup_plot("filament straightness (distance from ideal in nm)", 
                                             figure, current_figure)
        subplot.hist(result, density=1, bins=20)

        far = 0
        for d in result:
            if d > 2.5:
                far += 1
        
        print("STRAIGHTNESS ----------\n{}% of distances are over 2.5 nm\nmedian distance = {}\nmean distance = {}\nstd = {}\n".format(
            round(100 * far / float(len(result))), np.median(result), np.mean(result), np.std(result)))
    


# ### Run the Calculations (excluding Reaction rates)

# In[8]:


total_plots = 11
current_figure = 0
f = plt.figure(figsize=(15,25))

# [filamentous actin] / [total actin] vs time
observe_ratio_of_filamentous_to_total_actin(f)

# [filamentous ATP-actin] / [total filamentous actin] vs time
observe_ratio_of_ATP_actin_to_total_actin(f)

# [daughter filament actin] / [filamentous actin] vs time
observe_ratio_of_daughter_filament_actin_to_total_filamentous_actin(f)

# length of mother filaments (distribution vs time)
observe_mother_filament_lengths(f)

# length of daughter filaments (distribution vs time)
observe_daughter_filament_lengths(f)

# [bound arp2/3] / [total arp2/3] vs time
observe_ratio_of_bound_to_total_arp23(f)

# [capped ends] / [total ends] vs time
observe_ratio_of_capped_ends_to_total_ends(f)

# branch angles (histogram)
observe_branching_angles(f)

# pitch of actin helix (histogram)
observe_helix_pitches(f)

# filament bending (histogram)
observe_filament_straightness(f)

f.show()


# ### Calculate Reaction Rates

# In[ ]:


# ----------------------------------------------------------- actual reaction rates

# '''
# get the number of times a reaction with reaction_name was counted in the reaction count observable
# '''
# def get_reaction_count(reaction_name):
    
#     if reaction_name in reactions["reactions"]:
#         reaction_type = "reactions"
#     elif reaction_name in reactions["structural_topology_reactions"]:
#         reaction_type = "structural_topology_reactions"
#     elif reaction_name in reactions["spatial_topology_reactions"]:
#         reaction_type = "spatial_topology_reactions"
#     else:
#         print("couldn't find reaction named {}".format(reaction_name))
#         return 0
    
#     return int(round(np.sum(reactions[reaction_type][reaction_name])))

# '''
# add and subtract counts of observed reactions to get a total rate for a process
# '''
# def get_total_reaction_count(add_reaction_names, subtract_reaction_names):
    
#     result = 0
    
#     for reaction_name in add_reaction_names:
#         result += get_reaction_count(reaction_name)
#     a = result
#     for reaction_name in subtract_reaction_names:
#         result += -get_reaction_count(reaction_name)
    
#     return result

# '''
# format a rate in scientific notation
# '''
# def format_rate(rate):
    
#     if rate < sys.float_info.epsilon:
#         return "---"
#     else:
#         return "{} {}".format('%.2E' % Decimal(rate), "s⁻¹")
    
# '''
# print the observed rates
# '''
# def observe_reaction_rates():
    
#     print("ACTUAL RATES --------")

#     r = get_total_reaction_count(["Dimerize"], [])
#     print("dimerize = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Reverse_Dimerize"], ["Fail_Reverse_Dimerize"])
#     print("rev dimerize = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Trimerize1", "Trimerize2", "Trimerize3"], [])
#     print("trimerize = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Reverse_Trimerize"], ["Fail_Reverse_Trimerize"])
#     print("rev trimerize = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Pointed_Growth_ATP11", "Pointed_Growth_ATP12", "Pointed_Growth_ATP13", 
#                                   "Pointed_Growth_ATP21", "Pointed_Growth_ATP22", "Pointed_Growth_ATP23"], [])
#     print("pointed growth ATP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Pointed_Growth_ADP11", "Pointed_Growth_ADP12", "Pointed_Growth_ADP13", 
#                                   "Pointed_Growth_ADP21", "Pointed_Growth_ADP22", "Pointed_Growth_ADP23"], [])
#     print("pointed growth ADP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Pointed_Shrink_ATP"], ["Fail_Pointed_Shrink_ATP"])
#     print("pointed shrink ATP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Pointed_Shrink_ADP"], ["Fail_Pointed_Shrink_ADP"])
#     print("pointed shrink ADP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Barbed_Growth_ATP11", "Barbed_Growth_ATP12", "Barbed_Growth_ATP13", 
#                                   "Barbed_Growth_ATP21", "Barbed_Growth_ATP22", "Barbed_Growth_ATP23", 
#                                   "Barbed_Growth_Nucleate_ATP1", "Barbed_Growth_Nucleate_ATP2", 
#                                   "Barbed_Growth_Nucleate_ATP3", "Barbed_Growth_Branch_ATP"], [])
#     print("barbed growth ATP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Barbed_Growth_ADP11", "Barbed_Growth_ADP12", "Barbed_Growth_ADP13", 
#                                   "Barbed_Growth_ADP21", "Barbed_Growth_ADP22", "Barbed_Growth_ADP23", 
#                                   "Barbed_Growth_Nucleate_ADP1", "Barbed_Growth_Nucleate_ADP2", 
#                                   "Barbed_Growth_Nucleate_ADP3", "Barbed_Growth_Branch_ADP"], [])
#     print("barbed growth ADP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Barbed_Shrink_ATP"], ["Fail_Barbed_Shrink_ATP"])
#     print("barbed shrink ATP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Barbed_Shrink_ADP"], ["Fail_Barbed_Shrink_ADP"])
#     print("barbed shrink ADP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Hydrolysis_Actin"], ["Fail_Hydrolysis_Actin"])
#     print("hydrolyze actin = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Hydrolysis_Arp"], ["Fail_Hydrolysis_Arp"])
#     print("hydrolyze arp2/3 = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Nucleotide_Exchange_Actin"], [])
#     print("nucleotide exchange actin = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Nucleotide_Exchange_Arp"], [])
#     print("nucleotide exchange arp = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Arp_Bind_ATP11", "Arp_Bind_ATP12", "Arp_Bind_ATP13", 
#                                   "Arp_Bind_ATP21", "Arp_Bind_ATP22", "Arp_Bind_ATP23"], ["Fail_Arp_Bind_ATP"])
#     print("arp2/3 bind ATP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Arp_Bind_ADP11", "Arp_Bind_ADP12", "Arp_Bind_ADP13", 
#                                   "Arp_Bind_ADP21", "Arp_Bind_ADP22", "Arp_Bind_ADP23"], ["Fail_Arp_Bind_ADP"])
#     print("arp2/3 bind ADP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Debranch_ATP"], ["Fail_Debranch_ATP"])
#     print("debranch ATP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Debranch_ADP"], ["Fail_Debranch_ADP"])
#     print("debranch ADP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Arp_Unbind_ATP"], ["Fail_Arp_Unbind_ATP"])
#     print("arp2 unbind ATP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Arp_Unbind_ADP"], ["Fail_Arp_Unbind_ADP"])
#     print("arp2 unbind ADP = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Cap_Bind11", "Cap_Bind12", "Cap_Bind13", 
#                                   "Cap_Bind21", "Cap_Bind22", "Cap_Bind23"], [])
#     print("cap bind = {} ({})".format(format_rate(r / total_sim_time), r))
#     r = get_total_reaction_count(["Cap_Unbind"], ["Fail_Cap_Unbind"])
#     print("cap unbind = {} ({})".format(format_rate(r / total_sim_time), r))
    
    
# total reaction name : (readdy reactions to add, readdy reactions to subtract)
readdy_reactions = {
    "dimerize" : (["Dimerize"], []),
    "rev dimerize" : (["Reverse_Dimerize"], ["Fail_Reverse_Dimerize"]),
    "trimerize" : (["Trimerize1", "Trimerize2", "Trimerize3"], []),
    "rev trimerize" : (["Reverse_Trimerize"], ["Fail_Reverse_Trimerize"]),
    "pointed growth ATP" : (["Pointed_Growth_ATP11", "Pointed_Growth_ATP12", "Pointed_Growth_ATP13", 
                             "Pointed_Growth_ATP21", "Pointed_Growth_ATP22", "Pointed_Growth_ATP23"], []),
    "pointed growth ADP" : (["Pointed_Growth_ADP11", "Pointed_Growth_ADP12", "Pointed_Growth_ADP13", 
                             "Pointed_Growth_ADP21", "Pointed_Growth_ADP22", "Pointed_Growth_ADP23"], []),
    "pointed shrink ATP" : (["Pointed_Shrink_ATP"], ["Fail_Pointed_Shrink_ATP"]),
    "pointed shrink ADP" : (["Pointed_Shrink_ADP"], ["Fail_Pointed_Shrink_ADP"]),
    "barbed growth ATP" : (["Barbed_Growth_ATP11", "Barbed_Growth_ATP12", "Barbed_Growth_ATP13", 
                            "Barbed_Growth_ATP21", "Barbed_Growth_ATP22", "Barbed_Growth_ATP23", 
                            "Barbed_Growth_Nucleate_ATP1", "Barbed_Growth_Nucleate_ATP2", 
                            "Barbed_Growth_Nucleate_ATP3", "Barbed_Growth_Branch_ATP"], []),
    "barbed growth ADP" : (["Barbed_Growth_ADP11", "Barbed_Growth_ADP12", "Barbed_Growth_ADP13", 
                            "Barbed_Growth_ADP21", "Barbed_Growth_ADP22", "Barbed_Growth_ADP23", 
                            "Barbed_Growth_Nucleate_ADP1", "Barbed_Growth_Nucleate_ADP2", 
                            "Barbed_Growth_Nucleate_ADP3", "Barbed_Growth_Branch_ADP"], []),
    "barbed shrink ATP" : (["Barbed_Shrink_ATP"], ["Fail_Barbed_Shrink_ATP"]),
    "barbed shrink ADP" : (["Barbed_Shrink_ADP"], ["Fail_Barbed_Shrink_ADP"]),
    "hydrolyze actin" : (["Hydrolysis_Actin"], ["Fail_Hydrolysis_Actin"])
#     "hydrolyze arp" : (["Hydrolysis_Arp"], ["Fail_Hydrolysis_Arp"]),
#     "nucleotide exchange actin" : (["Nucleotide_Exchange_Actin"], []),
#     "nucleotide exchange arp" : (["Nucleotide_Exchange_Arp"], []),
#     "arp2/3 bind ATP" : (["Arp_Bind_ATP11", "Arp_Bind_ATP12", "Arp_Bind_ATP13", 
#                           "Arp_Bind_ATP21", "Arp_Bind_ATP22", "Arp_Bind_ATP23"], ["Fail_Arp_Bind_ATP"]),
#     "arp2/3 bind ADP" : (["Arp_Bind_ADP11", "Arp_Bind_ADP12", "Arp_Bind_ADP13", 
#                           "Arp_Bind_ADP21", "Arp_Bind_ADP22", "Arp_Bind_ADP23"], ["Fail_Arp_Bind_ADP"]),
#     "debranch ATP" : (["Debranch_ATP"], ["Fail_Debranch_ATP"]),
#     "debranch ADP" : (["Debranch_ADP"], ["Fail_Debranch_ADP"]),
#     "arp2 unbind ATP" : (["Arp_Unbind_ATP"], ["Fail_Arp_Unbind_ATP"]),
#     "arp2 unbind ADP" : (["Arp_Unbind_ADP"], ["Fail_Arp_Unbind_ADP"]),
#     "cap bind" : (["Cap_Bind11", "Cap_Bind12", "Cap_Bind13", "Cap_Bind21", "Cap_Bind22", "Cap_Bind23"], []),
#     "cap unbind" : (["Cap_Unbind"], ["Fail_Cap_Unbind"])
}      
        
'''
get the type of ReaDDy reaction for a ReaDDy reaction name
'''
def get_reaction_type(readdy_reaction_name):
    
    if readdy_reaction_name in reactions["reactions"]:
        return "reactions"
    
    if readdy_reaction_name in reactions["structural_topology_reactions"]:
        return "structural_topology_reactions"
    
    if readdy_reaction_name in reactions["spatial_topology_reactions"]:
        return "spatial_topology_reactions"
    
    print("couldn't find reaction named {}".format(readdy_reaction_name))
    return None

'''
get the number of times a ReaDDy reaction has happened by each time step
'''
def get_readdy_reaction_over_time(readdy_reaction_name, result, multiplier, stride):
        
    reaction_type = get_reaction_type(readdy_reaction_name)
    if reaction_type is None:
        return result

    count = 0
    for t in range(len(reactions[reaction_type][readdy_reaction_name])):
        
        count += multiplier * reactions[reaction_type][readdy_reaction_name][t]
        if t % stride == 0:
            
            i = int(math.floor(t / float(stride)))
            if len(result) < i + 1:
                result.append(0)
            
            result[i] += count
            
    return result

'''
get the number of times a set of ReaDDy reactions has happened by each time step
'''
def get_total_reaction_over_time(total_reaction_name, stride):
    
    reactions = readdy_reactions[total_reaction_name]
    result = []
    
    for reaction_name in reactions[0]:
        result = get_readdy_reaction_over_time(reaction_name, result, 1, round(stride))
    
    for reaction_name in reactions[1]:
        result = get_readdy_reaction_over_time(reaction_name, result, -1, round(stride))
    
    return result


'''
calculate the number of times each reaction has happened by each time step
'''
def get_all_reactions_over_time(steps):
    
    result = {}
    i = 0
    for r in readdy_reactions:
        
        result[r] = get_total_reaction_over_time(r, len(reaction_times) / float(steps))
        
        sys.stdout.write('\r')
        p = 100. * (i + 1) / float(len(readdy_reactions))
        sys.stdout.write("[{}{}] {}%".format('='*int(round(p)), ' '*int(100. - round(p)), round(10. * p) / 10.))
        sys.stdout.flush()
        i += 1
        
    return result


'''
calculates the concentration for a species 
    with number of particles n
    in cube container with dimensions box_size [nm]

    returns concentration [uM]
''' 
def calculate_concentration(n):
    
    return n / (1e-30 * 6.022e23 * np.power(box_size, 3.))


'''
calculate the concentration of free actin at each step to plot against reaction rates
'''
def get_free_actin_concentration_over_time(steps):
    
    stride = round(max_time / float(steps))
    free_actin_types = ["actin#free", "actin#free_ATP"]
    result = []
    for t in range(len(particle_info)):
        if t % stride == 0:
            
            result.append(calculate_concentration(len(get_ids_for_types(particle_info[t], free_actin_types))))
    
    return result


'''
plot reaction rates vs time
'''
def observe_reactions_over_time(reactions_over_time):
    
    total_plots = 24
    current_figure = 0
    figure = plt.figure(figsize=(15,50))
    
    t = []
    for reaction_name in reactions_over_time:
        
        if len(t) < 1:
            t = list(range(len(reactions_over_time[reaction_name])))
    
        subplot, current_figure = setup_plot(total_plots, "{} vs time".format(reaction_name), figure, current_figure)
        subplot.plot(t, reactions_over_time[reaction_name])


'''
plot reaction rates vs time for a combination of reactions
'''
def observe_reactions_over_time(reactions_over_time, add_reaction_names, subtract_reaction_names):
    
    total_plots = 1
    current_figure = 0
    figure = plt.figure(figsize=(15,5))
    
    result = []
    for reaction_name in add_reaction_names:
        
        if len(result) < 1:
            result = [0] * len(reactions_over_time[reaction_name])
            
        result = np.add(result, reactions_over_time[reaction_name])
        
    for reaction_name in subtract_reaction_names:
            
        result = np.subtract(result, reactions_over_time[reaction_name])
    
    t = list(range(len(reactions_over_time[reaction_name])))
    
    subplot, current_figure = setup_plot(total_plots, "reaction combo vs time", figure, current_figure)
    subplot.plot(t, result)
    
    return result


'''
plot reaction rates vs actin concentration
'''
def observe_reactions_vs_actin_concentration(reactions_over_time):
    
    total_plots = 24
    current_figure = 0
    figure = plt.figure(figsize=(15,50))
    
    c = []
    for reaction_name in reactions_over_time:
        
        if len(c) < 1:
            c = get_free_actin_concentration_over_time(len(reactions_over_time[reaction_name]))
    
        subplot, current_figure = setup_plot(total_plots, "{} vs [actin]".format(reaction_name), figure, current_figure)
        subplot.plot(c, reactions_over_time[reaction_name])


# In[ ]:


res = get_all_reactions_over_time(100)


# In[ ]:


'''
format subplots
'''
def setup_plot(total_plots, title, figure, index):
    
    subplot = figure.add_subplot(int(math.ceil(total_plots / 2.)), 2, index + 1)
    subplot.title.set_text(title)
    return subplot, index + 1


# In[ ]:


combo = observe_reactions_over_time(res, ["dimerize"], ["rev dimerize"])


# In[ ]:


combo = observe_reactions_over_time(res, ["trimerize"], ["rev trimerize"])


# In[ ]:


combo = observe_reactions_over_time(res, ["barbed growth ATP"], ["barbed shrink ATP"])


# In[ ]:


combo = observe_reactions_over_time(res, ["barbed growth ADP"], ["barbed shrink ADP"])


# In[ ]:


combo = observe_reactions_over_time(res, ["pointed growth ATP"], ["pointed shrink ATP"])


# In[ ]:


observe_reactions_over_time(res)


# In[ ]:


print(res["trimerize"])


# In[ ]:


time_per_step = 0.005 * 1e-9 * 5e7 / 100 # timestep * convert to s * total steps / steps in reaction observations

total_reaction_name = "trimerize"
last_linear_index = 11
init_rate = res[total_reaction_name][last_linear_index] / ((last_linear_index + 1) * time_per_step)
print("{} init rate = {} s-1".format(total_reaction_name, init_rate))


# ## Print outputs for Results UI

# In[15]:


timestep = 0.005 #ns
recording_stride = 1e4

'''
print time in ns
'''
def print_time():
    
    s = "["
    for t in times[min_time:max_time+1:time_inc]:
        s += "{},\n".format(t * timestep * recording_stride)
        
    print("{}]".format(s[:-2]))

    
'''
print [filamentous actin] / [total actin]
'''
def print_ratio_of_filamentous_to_total_actin():
    
    result = get_ratio_of_filamentous_to_total_actin()
    
    s = "["
    for r in result:
        s += "{},\n".format(r)
        
    print("{}]".format(s[:-2]))


'''
print [ATP-actin] / [total actin] in filaments
'''
def print_ratio_of_ATP_actin_to_total_actin():
    
    result = get_ratio_of_ATP_actin_to_total_actin()
    
    s = "["
    for r in result:
        s += "{},\n".format(r)
        
    print("{}]".format(s[:-2]))


'''
print [daughter filament actin] / [filamentous actin]
'''
def print_ratio_of_daughter_filament_actin_to_total_filamentous_actin():
    
    result = []
    for t in range(len(particle_info)):
        
        mother_actin = 0
        for mother_filament in mother_filaments[t]:
            mother_actin += len(mother_filament)
            
        daughter_actin = 0
        for daughter_filament in daughter_filaments[t]:
            daughter_actin += len(daughter_filament)
        
        if mother_actin + daughter_actin > 0:
            result.append(daughter_actin / float(mother_actin + daughter_actin))
        else:
            result.append(0)
    
    s = "["
    for r in result:
        s += "{},\n".format(r)
        
    print("{}]".format(s[:-2]))

    
'''
print length of mother filaments
'''
def print_mother_filament_lengths():
    
    result = []
    for t in range(len(particle_info)):
        
        lengths = []
        for filament in mother_filaments[t]:
            lengths.append(len(filament))
        
        result.append(lengths)
    
    s = "[\n    "
    for r in result:
        s += "[\n        "
        for l in r:
            s += "{},\n        ".format(l)
        s = "{}\n    ],\n    ".format(s[:-10])
        
    print("{}\n]".format(s[:-6]))


'''
print length of daughter filaments
'''
def print_daughter_filament_lengths():
    
    result = []
    for t in range(len(particle_info)):
        
        lengths = []
        for filament in daughter_filaments[t]:
            lengths.append(len(filament))
        
        result.append(lengths)
    
    s = "[\n    "
    for r in result:
        s += "[\n        "
        for l in r:
            s += "{},\n        ".format(l)
        s = "{}\n    ],\n    ".format(s[:-10])
        
    print("{}\n]".format(s[:-6]))


'''
print [bound arp2/3] / [total arp2/3]
'''
def print_ratio_of_bound_to_total_arp23():
    
    result = get_ratio_of_bound_to_total_arp23()
    
    s = "["
    for r in result:
        s += "{},\n".format(r)
        
    print("{}]".format(s[:-2]))


'''
print [capped ends] / [total ends]
'''
def print_ratio_of_capped_ends_to_total_ends():
    
    result = get_ratio_of_capped_ends_to_total_ends()
    
    s = "["
    for r in result:
        s += "{},\n".format(r)
        
    print("{}]".format(s[:-2]))
        
    
'''
print branch angles
'''
def print_branching_angles():
    
    result = []
    for t in range(len(particle_info)):
        result += get_branch_angles(particle_info[t])
    
    s = "["
    for r in result:
        s += "{},\n".format(r)
        
    print("{}]".format(s[:-2]))


'''
print short helix pitch
'''
def print_short_helix_pitches():
    
    result = []
    for t in range(len(particle_info)):
        
        pitches = get_helix_pitches(particle_info[t])
        result += pitches[0]
    
    s = "["
    for r in result:
        s += "{},\n".format(r)
        
    print("{}]".format(s[:-2]))


'''
print long helix pitch
'''
def print_long_helix_pitches():
    
    result = []
    for t in range(len(particle_info)):
        
        pitches = get_helix_pitches(particle_info[t])
        result += pitches[1]
    
    s = "["
    for r in result:
        s += "{},\n".format(r)
        
    print("{}]".format(s[:-2]))


'''
print filament straightness
'''    
def print_filament_straightness():
    
    result = []
    for t in range(len(particle_info)):
        result += get_distance_from_straight(particle_info[t])
    
    s = "["
    for r in result:
        s += "{},\n".format(r)
        
    print("{}]".format(s[:-2]))
    
    
        
# print_time()
# print_ratio_of_filamentous_to_total_actin()
# print_ratio_of_ATP_actin_to_total_actin()
# print_ratio_of_daughter_filament_actin_to_total_filamentous_actin()
print_mother_filament_lengths()
# print_daughter_filament_lengths()
# print_ratio_of_bound_to_total_arp23()
# print_ratio_of_capped_ends_to_total_ends()
# print_branching_angles()
# print_short_helix_pitches()
# print_long_helix_pitches()
# print_filament_straightness()


# ## Visualize

# In[36]:


# Optionally re-assign particle sizes for visualization 

actin_radius = 2. #nm
arp23_radius = 0. #nm
cap_radius = 0. #nm

traj.convert_to_xyz(particle_radii={
    "actin#free": 0,
    "actin#free_ATP": 0,
    "actin#new": actin_radius,
    "actin#new_ATP": actin_radius,
    "actin#1": actin_radius,
    "actin#2": actin_radius,
    "actin#3": actin_radius,
    "actin#ATP_1": actin_radius,
    "actin#ATP_2": actin_radius,
    "actin#ATP_3": actin_radius,
    "actin#pointed_1": actin_radius,
    "actin#pointed_2": actin_radius,
    "actin#pointed_3": actin_radius,
    "actin#pointed_ATP_1": actin_radius,
    "actin#pointed_ATP_2": actin_radius,
    "actin#pointed_ATP_3": actin_radius,
    "actin#barbed_1": actin_radius,
    "actin#barbed_2": actin_radius,
    "actin#barbed_3": actin_radius,
    "actin#barbed_ATP_1": actin_radius,
    "actin#barbed_ATP_2": actin_radius,
    "actin#barbed_ATP_3": actin_radius,
    "actin#branch_1": actin_radius,
    "actin#branch_2": actin_radius,
    "actin#branch_3": actin_radius,
    "actin#branch_ATP_1": actin_radius,
    "actin#branch_ATP_2": actin_radius,
    "actin#branch_ATP_3": actin_radius,
    "actin#branch_barbed_1": actin_radius,
    "actin#branch_barbed_2": actin_radius,
    "actin#branch_barbed_3": actin_radius,
    "actin#branch_barbed_ATP_1": actin_radius,
    "actin#branch_barbed_ATP_2": actin_radius,
    "actin#branch_barbed_ATP_3": actin_radius,
    "arp2": arp23_radius,
    "arp2#ATP": arp23_radius,
    "arp2#new": arp23_radius,
    "arp2#new_ATP": arp23_radius,
    "arp3": arp23_radius,
    "arp3#branched": arp23_radius,
    "cap": 0,
    "cap#bound": cap_radius
})

