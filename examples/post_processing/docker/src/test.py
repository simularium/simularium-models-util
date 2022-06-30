from simularium_models_util.visualization import MicrotubulesVisualization
import numpy as np

# h5_path = "outputs/scan_1_conc1x_rxn1x.h5"
# h5_path = "outputs/test_conc1x_rxn1x.h5"
h5_path = "outputs/microtubules_scan_growth_attach_20220418_x100_growth_x100_attach_1.h5"
# box_size = np.array([150.0, 150.0, 250.0])
box_size = np.array([300.0]*3)
stride = 1
pickle_file_path = "/mnt/c/Users/saurabh.mogre/OneDrive - Allen Institute/Projects/Simularium/simularium-models-util/examples/microtubules/"+h5_path+".dat"
sim_steps = 2e7
viz_steps = max(int(sim_steps / 1000.0), 1)
scaled_time_step_us = 0.1 * 1e-3 * viz_steps
plots = MicrotubulesVisualization.generate_plots(
    h5_path,
    box_size=box_size,
    stride=stride,
    save_pickle_file=True,
    # pickle_file_path=pickle_file_path,
    )
MicrotubulesVisualization.visualize_microtubules(
                h5_path,
                box_size=box_size, 
                scaled_time_step_us = scaled_time_step_us,
                plots=plots,
            )