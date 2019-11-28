from VibroSim_WelderModel import contact_model

# Uncomment this and comment out below to disable GPU use
#gpu_context_device_queue=None


# Uncomment these to enable GPU use
gpu_device_priority_list_str="[('NVIDIA CUDA','Quadro GP100'), ('Intel(R) OpenCL HD Graphics','Intel(R) Gen9 HD Graphics NEO'), ('Portable Computing Language', 'pthread-AMD EPYC 7351P 16-Core Processor')]"
gpu_context_device_queue = contact_model.select_gpu_device(gpu_device_priority_list_str)




gpu_precision="single"

specimen_model_file="example_specimen_model.csv.bz2"
max_t = 0.2 # s
mass_of_welder_and_slider = 2.0 # kg
pneumatic_force = 100.0 # N

welder_elec_ampl= 1.8e7*5.0 # (Volts?)
welder_elec_freq=19890.0 # Hz

# Al
nu=.33 # Poisson's ratio
E=68.9e9 # Pa

welder_spring_constant = 5000 # N/m

R_contact=25.4e-3 # m

specimen_dict = contact_model.load_specimen_model(specimen_model_file)


motiontable = contact_model.contact_model(specimen_dict,
                                          max_t,
                                          mass_of_welder_and_slider,
                                          pneumatic_force,
                                          welder_elec_ampl,
                                          E,
                                          nu,
                                          welder_spring_constant=welder_spring_constant,
                                          R_contact=R_contact,
                                          welder_elec_freq=welder_elec_freq,
                                          gpu_context_device_queue=gpu_context_device_queue,
                                          gpu_precision=dc_gpu_precision_str)


plotdict = contact_model.plot_contact(motiontable)

pl.show()
