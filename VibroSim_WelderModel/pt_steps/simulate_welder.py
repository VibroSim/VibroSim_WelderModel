import os
import os.path
import sys

from dc_value import numericunitsvalue as numericunitsv
from dc_value import hrefvalue as hrefv

from VibroSim_WelderModel import contact_model

def run(dc_dest_href,
        dc_measident_str,
        dc_dynamicmodel_href,
        dc_max_t_numericunits,
        dc_mass_of_welder_and_slider_numericunits,
        dc_pneumatic_force_numericunits,
        dc_welder_elec_ampl_float,
        dc_YoungsModulus_numericunits,
        dc_PoissonsRatio_float,
        dc_welder_spring_constant_numericunits = numericunitsv(contact_model.default_welder_spring_constant,"N/m"),
        dc_R_contact_numericunits = numericunitsv(contact_model.default_R_contact,"m"),
        dc_welder_elec_freq_numericunits = numericunitsv(contact_model.default_welder_elec_freq,"Hz"),
        dc_contact_model_timestep_numericunits = numericunitsv(contact_model.default_dt,"s"),
        dc_gpu_device_priority_list_str = "" # string containing python-style list of tuples of quoted strings with (platform name, device name) in priority order e.g. "[('NVIDIA CUDA','Quadro GP100'), ('Intel(R) OpenCL HD Graphics','Intel(R) Gen9 HD Graphics NEO'), ('Portable Computing Language', 'pthread-AMD EPYC 7351P 16-Core Processor')]". These names are shown under "Device Name" by the clinfo command. if "" is provided then acceleration will not be used. 
        dc_gpu_precision_str = contact_model.default_gpu_precision):
    
    specimen_dict = contact_model.load_specimen_model(dc_dynamicmodel_href.getpath())

    
    if dc_gpu_device_priority_list_str != "":

        gpu_device_priority_list = ast.literal_eval(dc_gpu_device_priority_list_str)
        platforms=cl.get_platforms()
        platforms_byname = { platform.name: platform for platform in platforms }

        device = None
        
        for (gpu_platform_name,gpu_device_name) in gpu_device_priority_list:
            
            if gpu_platform_name in platforms_byname:
                platform = platforms_byname[platformname]

                devices=platform.get_devices()
                devices_byname = { device.name: device for device in devices }

                if gpu_device_name in devices_byname:
                    device=devices_byname[devicename]
                    break
                pass
            pass
        
        if device is None:
            raise ValueError("No OpenCL devices found matching any entry in priority list %s. Use clinfo command to find platform name and device name. Priority list must be entered using list notation: \"[ ('platform1_name','device1_name'), ('platform2_name','device2_name') ]\"")
        context = cl.Context(devices=[device])
        queue = cl.CommandQueue(context)
        gpu_context_device_queue = (device,context,queue)
        pass
    else:
        gpu_context_device_queue=None
        pass
    
    motiontable = contact_model.contact_model(specimen_dict,
                                              dc_max_t_numericunits.value("s"),
                                              dc_mass_of_welder_and_slider_numericunits.value("kg"),
                                              dc_pneumatic_force_numericunits.value("N")
                                              dc_welder_elec_ampl_float,
                                              dc_YoungsModulus_numericunits.value("Pa"),
                                              dc_PoissonsRatio_float,
                                              welder_spring_constant=dc_welder_spring_constant_numericunits.value("N/m"),
                                              R_contact=dc_R_contact_numericunits.value("m"),
                                              welder_elec_freq=dc_welder_elec_freq_numericunits.value("Hz"),
                                              gpu_context_device_queue=gpu_context_device_queue,
                                              gpu_precision=dc_gpu_precision_str)


    # Save motiontable CSV and add to return dictionary
    motiontable_href = hrefv(quote("%s_motiontable.csv.bz2" % (dc_measident_str)),dc_dest_href)
    write_motiontable(motiontable,motiontable_href.getpath())
    ret = {
        "dc:motion": motiontable_href,
    }
    
    return ret 
