import sys
import os
import os.path
import copy
import collections
import ast



import numpy
import numpy as np
import pandas as pd


from matplotlib import pyplot as pl
#import pyximport
#pyximport.install()
from . import convolution
from .convolution import impulse_response,convolution_evaluation


#
#
# Inputs:
# Welder model: Vector A of complex amplitude
#               Vector alpha of complex exponents
#               time delay (syn_time_delay)
#               initial_displacement (should be -rho*c_A) because motion positive toward vibrometer
# Welder model predicts response to an impulse force away from the vibrometer
# with amplitude 1 N*s

# Also need a model for the pneumatic cylinder behaviour
# That provides the restoring force
# and the long-time behavior where the 1 N-s impulse
# accelerate the entire welder. 


#
#
# Welder & Specimen model
# Represent as impulse response, possibly followed by continued ringing
# of one or more decaying sinusoids (i.e. exp(-alpha*t)


class dummy(object):
    pass

pkgpath = sys.modules[dummy.__module__].__file__
pkgdir=os.path.split(pkgpath)[0]

# All welder responses now loaded from welder_modeling_out.py
welder_model_output_dir = pkgdir

# Welder model provides globals welder_tip_tip_resp, welder_tip_elec_resp, welder_elec_tip_resp,
# and welder_elec_elec_resp
exec(open(os.path.join(welder_model_output_dir,"welder_modeling_out.py")).read(),globals())


## Temporarily zero out welder_tip_tip_resp for convergence testing
#welder_tip_tip_resp.h[:]=0.0
#welder_tip_tip_resp.h[:10]=-1e-28
#welder_tip_tip_resp.h[1]=-1e-28
#welder_tip_tip_resp.A[:]=0.0

default_welder_spring_constant=5000 # N/m -- bounciness of seals in welder pneumatic cylinder
default_R_contact=25.4e-3 # Hertzian contact parameter: 1 inch radius                  
default_welder_elec_freq=19890.0 # Frequency, Hz
default_dt=1e-6 # Time step, seconds

default_gpu_precision="single" # gpu_precision must be "single" or "double"

contact_F_nominal = 100.0 # N -- nominal contact force used to decide to neglect imaginary parts order of magnitude smaller
G_nominal = contact_F_nominal**(1.0/3.0)


#specimen_model_output_dir = '.'
#specimen_model_fname = "specimen_modeling_out.py"
#specimen_model_fname = "cantilever_modeling_out.py"
#specimen_model_fname = "cantilever_modeling_out_moredamping.py"
#specimen_model_fname = "cantilever_modeling_out_2019_10_01.py"
#specimen_model_fname = "cantilever_modeling_out_2019-11-09-finerseg.py.xz"

## response of specimen, evaluated at welder contact point
#specimen_dataframe = pd.read_csv(specimen_model_fname)
#specimen_resp =

#specimen_model_fpath=os.path.join(specimen_model_output_dir,specimen_model_fname)


def select_gpu_device(priority_list_str):
    """Based on a priority list string of the form:
     [
       ('NVIDIA CUDA','Quadro GP100'), 
       ('Intel(R) OpenCL HD Graphics','Intel(R) Gen9 HD Graphics NEO'), 
       ('Portable Computing Language','pthread-AMD EPYC 7351P 16-Core Processor')
     ]
    select the first of these devices found and return (context,device,queue).
    If priority_list_str=="" then None will be returned. 
    Otherwise a ValueError or other error will be raised if the 
    string is not parseable or if the device is not found. 
    
    Each entry in the list is (Platform Name, Device Name)
    The available platforms and devices can be found by looking
    at the output of the "clinfo" command. 
    """
    
    if priority_list_str != "":
        import pyopencl as cl
        gpu_device_priority_list = ast.literal_eval(priority_list_str)
        platforms=cl.get_platforms()
        platforms_byname = { platform.name: platform for platform in platforms }

        device = None
        
        for (gpu_platform_name,gpu_device_name) in gpu_device_priority_list:
            
            if gpu_platform_name in platforms_byname:
                platform = platforms_byname[gpu_platform_name]

                devices=platform.get_devices()
                devices_byname = { device.name: device for device in devices }

                if gpu_device_name in devices_byname:
                    device=devices_byname[gpu_device_name]
                    break
                pass
            pass
        
        if device is None:
            raise ValueError("No OpenCL devices found matching any entry in priority list %s. Use clinfo command to find platform name and device name. Priority list must be entered using list notation: \"[ ('platform1_name','device1_name'), ('platform2_name','device2_name') ]\"")
        context = cl.Context(devices=[device])
        queue = cl.CommandQueue(context)
        gpu_context_device_queue = (context,device,queue)
        pass
    else:
        gpu_context_device_queue=None
        pass
    return gpu_context_device_queue

def load_specimen_model(specimen_model_filepath):
    import pandas as pd
    specimen_dataframe = pd.read_csv(specimen_model_filepath,index_col="Time(s)")
    
    #dt=specimen_dataframe["Time(s)"][1]-specimen_dataframe["Time(s)"][0]
    dt=specimen_dataframe.index[1]-specimen_dataframe.index[0]

    assert(specimen_dataframe.index[0]==0.0)
    
    specimen_dict=collections.OrderedDict()    
    for column in specimen_dataframe.columns:
        if column=="Time(s)":
            continue
        specimen_dict[column.split("(")[0]]=convolution.impulse_response(
            h=np.array(specimen_dataframe[column]),
            dt=dt,
            t0=0.0,
            A=np.array((0,),dtype='d'),
            D=0.0,
            alpha=np.array((1.0,),dtype='d'))
        
        pass

    return specimen_dict # keys are "specimen_resp", etc.; values are impulse_response instances


def contact_model(specimen_dict,
                  t0_t1, # Excitation start time (seconds)
                  t2_t3, # Excitation end time (seconds)
                  t4, # Time to calculate out to (seconds)
                  mass_of_welder_and_slider, # mass, kg
                  pneumatic_force, # Force, N
                  welder_elec_ampl, # Amplitude (au...? should be volts)
                  specimen_E, specimen_nu, # Specimen elastic params
                  welder_spring_constant=default_welder_spring_constant, # N/m -- bounciness of seals in welder pneumatic cylinder
                  R_contact=default_R_contact, # Hertzian contact parameter: 1 inch radius                  
                  welder_elec_freq=default_welder_elec_freq, # Frequency, Hz
                  dt=default_dt, # Time step, seconds
                  gpu_context_device_queue=None,                  
                  gpu_precision=default_gpu_precision):

    # specimen_model defines specimen_resp, specimen_mobility, specimen_laser,
    # specimen_crackcenternormalstrain, and specimen_crackcentershearstrain
    
    ## temporarily hardwire gpu_precision to double
    #gpu_precision="double"
    #gpu_context_device_queue=None # Temporarily disable GPU

    #specimen_dict=load_specimen_model(specimen_model_fpath)

    # IDEA: Add small uniform real part to specimen_resp in frequency domain
    # To keep the phase away from the edge.... FAILED

    # IDEA: Add damping to Hertzian contact spring

    # Resample specimen data to desired dt value
    for key in specimen_dict:
        
        specimen_dict[key].resample(dt)
        pass
    
    specimen_resp=specimen_dict["specimen_resp"]
    del specimen_dict["specimen_resp"] # Remove specimen_resp from specimen_dict so convolutions with entries in specimen_dict will not be redundant
    if "specimen_mobility" in specimen_dict:
        del specimen_dict["specimen_mobility"] # Don't care about predicting specimen contact velocity
        pass 
    
    ## Temporarily zero out specimen_resp
    #specimen_resp.h[:]=0.0

    assert(welder_elec_tip_resp.h[0]==0.0) # response of tip to electrical excitation MUST be delayed
    assert(welder_elec_elec_resp.h[0]==0.0) # electrical response to electrical excitation MUST be delayed
    assert(welder_tip_elec_resp.h[0]==0.0) # electrical response to tip impulse MUST be delayed
    
    
    welder_tip_tip_resp_local=copy.deepcopy(welder_tip_tip_resp)
    welder_tip_elec_resp_local=copy.deepcopy(welder_tip_elec_resp)
    welder_elec_tip_resp_local=copy.deepcopy(welder_elec_tip_resp)
    welder_elec_elec_resp_local=copy.deepcopy(welder_elec_elec_resp)
    
    # resample all responses to desired timestep
    welder_tip_tip_resp_local.resample(dt)
    welder_tip_elec_resp_local.resample(dt)
    welder_tip_elec_resp_local.h[0]=0.0 # resampling can mess this up
    welder_elec_tip_resp_local.resample(dt)
    welder_elec_tip_resp_local.h[0]=0.0 # resampling can mess tihs up
    welder_elec_elec_resp_local.resample(dt)
    welder_elec_elec_resp_local.h[0]=0.0 # resampling can mess this up
    

    # welder_tip_tip_resp_local.h[0] is negative... specimen_resp.h[0] is positive,
    # (criteria for solving for contact force, below)
    assert(welder_tip_tip_resp_local.h[1] < 0.0)
    #assert(specimen_resp.h[0] > 0.0)
    

    # Connect specimen and welder models at contact
    #  * Sum of forces at contact = 0 (contact massless)
    #  * No overlap between specimen and welder.
    
    # z positive towards specimen
    welder_tip_z = 0
    specimen_z = 0
    
    #max_t = 0.1
    #max_t = 0.3
    #max_t=0.2
    trange = np.arange(t4/dt)*dt
    
    #mass_of_welder_and_slider=2.0   # mass, kg !!!*** Need to properly measure
    #pneumatic_force = 300 # Force, N  ***!!! Not necessarily representative
    # NOTE: If welder_spring_constant is nonzero, the pneumatic force ( in
    # equilbrium) be split between contact load and spring!!!
    

    # NOTE: with welder_overall_dashpot = 0
    # get resonance (~170Hz as of this writing), which
    # matches (1/(2pi))*sqrt(k/m) for
    # m = mass_of_welder_and_slider = 2 and
    # k = 1.0/(np.sum(specimen_resp.eval(np.arange(trange.shape[0])).real)*dt) = 2.5e6 N/m
    # represents the effective DC stiffness of the specimen
    #  (we should consider the welder and contact stiffnesses as well probably,
    #  but they are much stiffer!) 
    # For critically damped, set damping ratio = c/(2sqrt(m*k)) = 1
    # so c = 2*sqrt(m*k) = 4472 for critically damped.
    # Probably want significantly underdamped, so set c=400 N/(m/s)
    welder_overall_dashpot = 1000 # dashpot coefficient simulating absorption of pneumatic cylinder
    
    # ... at welder 50% amplitude setting , expect open circuit velocity of roughly 80 um p-p at 20 kHz. this corresponds to a welder_elec_ampl of ~1.8e7 (UNITS?)
    #welder_elec_ampl= 1.8e7*1.0
    
    #... double it!
    #welder_elec_ampl= 1.8e7*5.0
    
    #welder_elec_freq=19890.0
    
    # Plot welder open-circuit behavior with:
    #
    #welder_elec_input = welder_elec_ampl*np.cos(2*np.pi*welder_elec_freq*trange)
    #welder_elec_tip_conv = convolution_evaluation.blank_from_imp_resp(welder_elec_tip_resp)
    #welder_tip_z = np.zeros(trange.shape,dtype='d')
    #for tcnt in range(trange.shape[0]):
    #    welder_tip_z[tcnt]=welder_elec_tip_conv.step(welder_elec_input[tcnt])
    #    pass
    #pl.plot(trange,welder_tip_z)
    
    
    # NOTE: This spring represents the limited contact zone,
    # and helps to prevent system resonances involving
    # high Q, high-mobility resonances like the 20 kHz welder motion
    #kspring = 120e9*(np.pi*12e-3**2/4.0)/(1e-3)*.1 # *100)  # contact stiffness, N/m
    # NOTE: kspring replaced by Hertzian contact model
    # Hertzian contact parameters
    #R_contact=25.4e-3 # 1 inch radius
    # Welder material (Ti)
    nu1=.342
    E1=113.8e9
    
    ## Al
    #nu2=.33
    #E2=68.9e9
    

    Estar = 1.0/( (1.0-nu1**2.0)/E1 + (1.0-specimen_nu**2.0)/specimen_E)


    # This spring represents the bounciness of the seals in the pneumatic cylinder... based on 20 ms resonant period, and assuming 2kg mass,
    # f=sqrt(k/m) = 1/.02
    # f*sqrt(m) = sqrt(k)
    # k = f^2*m = (1/.02)^2 * 2.0 = 5000 N/m 
    # ... if you set this to 0, then you get instead pure
    # pneumatic cylinder behavior
    #welder_spring_constant = 5000
    

    welder_overall_velocity=0.0 # positive towards specimen
    #welder_overall_pos=-.5e-3 # half-mm distance initially # positive towards specimen
    #welder_overall_pos=0e-3 # no distance initially # positive towards specimen
    
    # This code is for starting from equilibrium displacement
    # Next line is correct for pneumatic cylinder but no spring
    #welder_overall_pos = convolution_evaluation.quiescent_value(pneumatic_force,specimen_resp.A,specimen_resp.alpha,specimen_resp.dt,specimen_resp.h).real - convolution_evaluation.quiescent_value(pneumatic_force,welder_tip_tip_resp_local.A,welder_tip_tip_resp_local.alpha,welder_tip_tip_resp_local.dt,welder_tip_tip_resp_local.h).real

    # overall_pos = specimen_quiescent_coeff*(pneumatic-welder_spring_constant*overall_pos) - welder_quiescent_coeff*(pneumatic-welder_spring_constant*overall_pos)
    # overall_pos = specimen_quiescent_coeff*pneumatic-specimen_quescent_coeff * welder_spring_constant*overall_pos - welder_quiescent_coeff*pneumatic + welder_quiescent_coeff*welder_spring_constant*overall_pos
    # overall_pos*(1.0 + specimen_quiescent_coeff*welder_spring_constant - welder_quiescent_coeff*welder_spring_constant) = (specimen_quiescent_coeff - welder_quiescent_coeff)*pneumatic
    # overall_pos = (specimen_quiescent_coeff - welder_quiescent_coeff)*pneumatic/(1.0 + specimen_quiescent_coeff*welder_spring_constant - welder_quiescent_coeff*welder_spring_constant)

    # Hertzian equilibrium displacement:
    # Displacement = (9/(16Estar^2R))^(1/3) * contact_F^(2/3)
    initial_contact_displacement = (9.0/(16.0*R_contact*Estar**2.0))**(1.0/3.0) * pneumatic_force**(2.0/3.0)
    

    welder_overall_pos = (convolution_evaluation.quiescent_value(1.0,specimen_resp.A,specimen_resp.alpha,specimen_resp.dt,specimen_resp.h).real - convolution_evaluation.quiescent_value(1.0,welder_tip_tip_resp_local.A,welder_tip_tip_resp_local.alpha,welder_tip_tip_resp_local.dt,welder_tip_tip_resp_local.h).real)*pneumatic_force/(1.0 + convolution_evaluation.quiescent_value(1.0,specimen_resp.A,specimen_resp.alpha,specimen_resp.dt,specimen_resp.h).real*welder_spring_constant - convolution_evaluation.quiescent_value(1.0,welder_tip_tip_resp_local.A,welder_tip_tip_resp_local.alpha,welder_tip_tip_resp_local.dt,welder_tip_tip_resp_local.h).real*welder_spring_constant) + initial_contact_displacement


    


    # Define convolutions of the various responses
    # This is for zero initial displacement
    #specimen_conv = convolution_evaluation.blank_from_imp_resp(specimen_resp)
    # This is for equilbrium initial displacement

    specimen_conv = convolution_evaluation.quiescent_from_imp_resp(specimen_resp,pneumatic_force-welder_overall_pos*welder_spring_constant,gpu_context_device_queue=gpu_context_device_queue,gpu_precision=gpu_precision)


    # specimen_dict_conv gets convolution evaluations setup just like
    # specimen_conv for all the other characteristics of interest
    # (laser point velocity, crack normal stress, etc.)
    specimen_dict_conv=collections.OrderedDict()
    specimen_dict_history=collections.OrderedDict()
    for specimen_motion_characteristic in specimen_dict:
        specimen_dict_conv[specimen_motion_characteristic] = convolution_evaluation.quiescent_from_imp_resp(specimen_dict[specimen_motion_characteristic],pneumatic_force-welder_overall_pos*welder_spring_constant,gpu_context_device_queue=gpu_context_device_queue,gpu_precision=gpu_precision)

        specimen_dict_history[specimen_motion_characteristic]=np.zeros(trange.shape[0],dtype='d')

        
        pass
    
    

    assert(dt==specimen_resp.dt)
    
    
    # This is for zero initial displacement
    #welder_tip_tip_conv = convolution_evaluation.blank_from_imp_resp(welder_tip_tip_resp_local)
    # This is for equilbrium initial displacement
    welder_tip_tip_conv = convolution_evaluation.quiescent_from_imp_resp(welder_tip_tip_resp_local,pneumatic_force)
    assert(dt==welder_tip_tip_resp_local.dt)


    welder_tip_elec_conv = convolution_evaluation.blank_from_imp_resp(welder_tip_elec_resp_local)
    assert(dt==welder_tip_elec_resp_local.dt)
    
    welder_elec_tip_conv = convolution_evaluation.blank_from_imp_resp(welder_elec_tip_resp_local)
    assert(dt==welder_elec_tip_resp_local.dt)
    assert(welder_elec_tip_resp_local.h[0]==0.0) # response of tip to electrical excitation MUST be delayed
    
    welder_elec_elec_conv = convolution_evaluation.blank_from_imp_resp(welder_elec_elec_resp_local)
    assert(dt==welder_elec_elec_resp_local.dt)
    assert(welder_elec_elec_resp_local.h[0]==0.0) # electrical response to electrical excitation MUST be delayed
    


    specimen_z_history=np.zeros(trange.shape[0],dtype='d')
    welder_tip_z_history=np.zeros(trange.shape[0],dtype='d')
    contact_F_history=np.zeros(trange.shape[0],dtype='d')
    welder_overall_velocity_history=np.zeros(trange.shape[0],dtype='d')

    welder_tip_z=welder_tip_tip_conv.evaluate() + welder_elec_tip_conv.evaluate() + welder_overall_pos

    specimen_z=specimen_conv.evaluate()
    
    
    last_overlap = welder_tip_z - specimen_z

    for tcnt in range(trange.shape[0]):
        if tcnt % 10000 == 0:
            print("tcnt=%d/%d" % (tcnt,trange.shape[0]))
            pass
    


        #if tcnt==42890:
        #    raise ValueError("Debug!")
        
        #if tcnt==3953: 
        #    import pdb
        #    pdb.set_trace()
        #    pass

        specimen_z=specimen_conv.step_without_instantaneous()
        
        welder_overall_pos += welder_overall_velocity*dt
        welder_tip_z=welder_tip_tip_conv.step_without_instantaneous() + welder_elec_tip_conv.step_without_instantaneous() + welder_overall_pos
        
    
        
        welder_elec_resp_voltage = welder_tip_elec_conv.step_without_instantaneous() + welder_elec_elec_conv.step_without_instantaneous()
        
        # Determine electrical control input
        
        # Welder elec input  --- Could include welder controller behavior here (based on welder_elec_resp_voltage and its history)
        
        if trange[tcnt] >= t0_t1 and trange[tcnt] <= t2_t3:
            welder_elec_input = welder_elec_ampl*np.cos(2*np.pi*welder_elec_freq*(trange[tcnt]-t0_t1))
            pass
        else:
            welder_elec_input = 0.0
            pass

        # Determine contact force from overlap
    
        # evaluate overlap
        overlap = welder_tip_z - specimen_z  # overlap represents amount of overalp between welder and specimen if no force is applied in this step


        if overlap > 0:
            # Conditions
            # Define Fwelder positive compression into welder
            # Define Fspecimen positive compression into specimen
            # Fwelder = Fspecimen
            # zshift_welder = instantaneous_welder_displacement*Fwelder*dt    # instantaneous_welder_displacement from welder model corresponds to displacement resulting from a 1 N*s impulse. Therefore it can be interpreted as have units of meters/(N*s) It is negative because the positive contact_F pulse pushes the welder away from the vibrometer
            # zshift_specimen = specimen_response[0]*Fspecimen*dt # Due to 1 N*s impulse... Should be positive
            # Add in contact springiness in series with
            # surface springiness:
            #  zspring=contact_F/kspring
            # zshift_welder - zshift_specimen - zspring = -overlap (known)
            
            # Solve this sytem....
            # let contact_F = Fwelder = Fspecimen = Fspring
            # instantaneous_welder_displacement*contact_F*dt  - specimen_response[0]*contact_F*dt - contact_F/kspring = -overlap
            # F = -overlap/(dt*(instantaneous_welder_displacement-specimen_resp[0]) - (1/kspring))
            
            # New Hertzian contact model
            # instantaneous_welder_displacement*contact_F*dt  - specimen_response[0]*contact_F*dt - (9/(16Estar^2R))^(1/3) * contact_F^(2/3) = -nocontactforce_overlap
            
            # This is a cubic equation for contact_F
            # for a*F - b*F^(2/3) + c = 0
            # G = F^(1/3)
            # a*G^3 - b*G^2 + c = 0
            G = np.roots([np.real(welder_tip_tip_resp_local.h[0]*dt-specimen_resp.h[0]*dt),-(9.0/(16.0*R_contact*Estar**2.0))**(1.0/3.0),0.0,overlap])
            use_G = G[(G.real > 0) & (abs(G.imag) <= G_nominal*1e-8)].real
            if use_G.shape[0] != 1:
                raise ValueError("Bad Hertzian contact solution: %s" % (str(G)))
            
            contact_F=use_G[0]**3.0
        

            # Old simple spring contact model here:
            #contact_F = -overlap/(dt*(np.real(welder_tip_tip_resp_local.h[0]-specimen_resp.h[0])) - 1.0/kspring)  # welder_tip_tip_resp_local.h[0] is negative... specimen_resp.h[0] is positive, overlap is positive, so contact_F is positive for compressive force between
            # Welder and specimen
            # Accumulate immediate response to this force
            pass
        else:
            contact_F=0.0
            pass
        
        contact_F_history[tcnt]=contact_F
    
        
        # Need to convolve force history
        # displacement = integral( F(t-tau) * response(tau)) dtau
        # store force history Fhist(t)
        # At a particular time t, 
        # displacement = integral( F(t-tau) * response(tau)) dtau where tau >= 0
        
        #welder_tip_z = welder_tip_z + contact_F*welder_tip_tip_resp_local.h[0]*dt
        #specimen_z = specimen_z + contact_F*specimen_resp.h[0]*dt
        
        # Positive (compressive) contact_F gives positive instantaneous
        # contribution... specimen_z moves in +z direction (away from welder)
        specimen_z += specimen_conv.step_instantaneous_contribution(contact_F)

        for specimen_motion_characteristic in specimen_dict_conv:
            specimen_dict_history[specimen_motion_characteristic][tcnt] = specimen_dict_conv[specimen_motion_characteristic].step(contact_F)
            pass
        
        
        # Positive (compressive) contact_F gives negative instantaneous
        # contribution... welder tip moves in -z direction (away from specimen)
        # Also need to apply contribution of welder electric input (but
        # the instantaneous effect of this is zero
        welder_tip_z += welder_tip_tip_conv.step_instantaneous_contribution(contact_F) + welder_elec_tip_conv.step_instantaneous_contribution(welder_elec_input)
        
        # Apply the contribution to the electric response...
        # instantaneous effect is zero. 
        welder_elec_resp_voltage = welder_tip_elec_conv.step_instantaneous_contribution(contact_F) + welder_elec_elec_conv.step_instantaneous_contribution(welder_elec_input)
        assert(welder_elec_resp_voltage==0.0) # These should not have instantaneous responses!
        
    
        # Should verify that overlap has been reduced to approximately 0
        new_overlap = welder_tip_z - specimen_z
        #assert(new_overlap < 1e-12)  (new overlap is no longer small now that we have added contact stiffness) 
        
        specimen_z_history[tcnt]=specimen_z
        welder_tip_z_history[tcnt]=welder_tip_z
        
        # determine change in welder_overall_velocity
        # ***!!!! Should this have a lag ?
        # F = ma  -> a = F/m
        # v = vprior + a*dt
        # v = vprior + (F/m)*dt
        # Where F = pneumatic_force - contact_F - welder_overall_dashpot*welder_overall_velocity
        welder_spring_force = -welder_tip_z*welder_spring_constant + pneumatic_force    # use "pneumatic_force" parameter as spring preload
        welder_overall_velocity += ((welder_spring_force-contact_F - welder_overall_dashpot*welder_overall_velocity)/mass_of_welder_and_slider) * dt
        assert(np.imag(welder_overall_velocity)==0.0)
        
        welder_overall_velocity_history[tcnt]=welder_overall_velocity
        
        #if tcnt >= 80890 and contact_F > 0:
        #raise ValueError("Debug!")
        #    break
        last_overlap = new_overlap
        pass
    # !!! Need conservation of energy constraint !!!***
    # Would it help to make contact more compliant (don't require
    #  exactly zero overlap)? 
    
    
    motiontable = pd.DataFrame(index=pd.Float64Index(data=np.arange(trange.shape[0],dtype='d')*dt,dtype='d',name="Time(s)"))
    
    motiontable.insert(len(motiontable.columns),"specimen_z_history(m)",specimen_z_history)
    motiontable.insert(len(motiontable.columns),"welder_tip_z_history(m)",welder_tip_z_history)
    motiontable.insert(len(motiontable.columns),"contact_F_history(N)",contact_F_history)
    motiontable.insert(len(motiontable.columns),"welder_overall_velocity_history(m/s)",welder_overall_velocity_history)

    motiontable.insert(len(motiontable.columns),"welder_tip_tip_resp(m/(N*s))",welder_tip_tip_resp_local.eval(np.arange(trange.shape[0])).real)
    motiontable.insert(len(motiontable.columns),"specimen_resp(m/(N*s))",specimen_resp.eval(np.arange(trange.shape[0])).real)
    
    for specimen_motion_characteristic in specimen_dict_history:
        motiontable.insert(len(motiontable.columns),specimen_motion_characteristic,specimen_dict_history[specimen_motion_characteristic])
        pass
    
    return motiontable

def write_motiontable(motiontable,output_filename):
    if output_filename.endswith(".bz2"):
        motiontable.to_csv(output_filename,compression='bz2')
        pass
    else:
        motiontable.to_csv(output_filename)
        pass

    pass



def plot_contact(motiontable):

    trange = np.array(motiontable.index) # ["Time(s)"]
    dt=trange[1]-trange[0]

    difft = (trange[:-1]+trange[1:])/2.0

    max_t_plot = np.max(trange)
    
    frange = np.arange(trange.shape[0],dtype='d')/(trange.shape[0]*dt)
    frange[trange.shape[0]//2:] -= 1.0/dt
    frange[trange.shape[0]//2]=np.nan

    welder_tip_z_history = motiontable["welder_tip_z_history(m)"]
    specimen_z_history = motiontable["specimen_z_history(m)"]
    contact_F_history = motiontable["contact_F_history(N)"]
    
    
    impresp_plot=pl.figure()
    pl.clf()
    pl.title("Welder and specimen impulse response")
    pl.plot(trange*1e6,
            motiontable["welder_tip_tip_resp(m/(N*s))"],'-',
            trange*1e6,
            motiontable["specimen_resp(m/(N*s))"],'--')    
    pl.xlabel('time (us)')
    pl.legend(('Welder','Specimen'))
    pl.grid()
        
    velspec_plot=pl.figure()
    pl.clf()
    pl.title("Welder and specimen velocity spectrum")
    pl.plot(frange/1e3,
            np.abs(np.fft.fft(motiontable["welder_tip_tip_resp(m/(N*s))"])*dt*(2.0*np.pi*np.abs(frange))),'-',
            frange/1e3,
            np.abs(np.fft.fft(motiontable["specimen_resp(m/(N*s))"])*dt*(2.0*np.pi*np.abs(frange))),'--')
    pl.xlabel('Frequency (kHz)')
    pl.ylabel('Velocity spectrum (m/s/Hz)')
    pl.legend(('Welder','Specimen'))
    pl.axis((0,250,0,.012))
    pl.grid()
        
    phasespec_plot = pl.figure()
    # NOTE: Because welder velocity is negated
    # relative to force direction, we undo that by negating before
    # evaluating the angle, so it is clear whether the phase is in the
    # required (-pi/2,pi/2) range
    pl.clf()
    pl.title("Welder and specimen velocity phase spectrum")
    pl.plot(frange/1e3,
            np.angle(-np.fft.fft(motiontable["welder_tip_tip_resp(m/(N*s))"])*dt*((0+1j)*2.0*np.pi*np.abs(frange))),'-',
            frange/1e3,
            np.angle(np.fft.fft(motiontable["specimen_resp(m/(N*s))"])*dt*((0+1j)*2.0*np.pi*np.abs(frange))),'--')
    pl.xlabel('Frequency (kHz)')
    pl.ylabel('Phase angle (rad)')
    pl.legend(('Welder','Specimen'))
    pl.axis((0,250,-np.pi,np.pi))
    pl.grid()
    
    
    disp_plot = pl.figure()
    pl.clf()
    pl.plot(trange*1e3,specimen_z_history*1e6,'-',
            trange*1.e3,welder_tip_z_history*1e6,'-')
    pl.axis((0,max_t_plot*1.e3,min(np.min(specimen_z_history*1e6),np.min(welder_tip_z_history*1e6)),max(np.max(specimen_z_history*1e6),np.max(welder_tip_z_history*1e6))))
    pl.xlabel('Time (ms)')
    pl.ylabel('Displacement (um)')
    pl.title('Specimen and welder displacement')
    pl.legend(('Specimen','Welder'))
    pl.grid()
    

    
    dispzoom_plot = pl.figure()
    pl.clf()
    fig5_tstart = 5e-3
    fig5_tend = 6e-3
    pl.plot(trange*1e3,specimen_z_history*1e6,'-',
            trange*1.e3,welder_tip_z_history*1e6,'-')
    pl.axis((fig5_tstart*1e3,fig5_tend*1e3,min(np.min(specimen_z_history[(trange >= fig5_tstart) & (trange <= fig5_tend)]*1e6),np.min(welder_tip_z_history[(trange >= fig5_tstart) & (trange <= fig5_tend)]*1e6)),max(np.max(specimen_z_history[(trange >= fig5_tstart) & (trange <= fig5_tend)]*1e6),np.max(welder_tip_z_history[(trange >= fig5_tstart) & (trange <= fig5_tend)]*1e6))))
    pl.xlabel('Time (ms)')
    pl.ylabel('Displacement (um)')
    pl.title('Specimen and welder displacement')
    pl.legend(('Specimen','Welder'))
    pl.grid()

        
    contactforce_plot = pl.figure()
    pl.clf()
    # Impulse between .28*.33
    Impulse=contact_F_history[(trange >.28e-3) & (trange < .33e-3)].sum()*dt
    pl.plot(trange*1e3,contact_F_history)
    pl.axis((0,max_t_plot*1.e3,0,20000))
    pl.xlabel('Time (ms)')
    pl.ylabel('Force (N)')
    pl.title('Contact force (time domain')
    pl.grid()
        
        
    #pl.figure(5)
    #pl.clf()
    #pl.plot(trange,welder_overall_velocity_history)
        
    contactspectrum_plot = pl.figure()
    pl.clf()
    # Impulse between .28*.33
    pl.plot(frange/1e3,np.abs(np.fft.fft(contact_F_history)*dt))
    #pl.axis((0,max_t_plot*1.e3,0,20000))
    pl.xlabel('Frequency (kHz)')
    pl.ylabel('Force spectrum (N/Hz)')
    pl.title('Contact force (frequency domain')
    pl.grid()
        
        
    overlap_plot = pl.figure()
    pl.clf()
    pl.plot(trange*1.e3,(welder_tip_z_history-specimen_z_history)*1e6,'-')
    #pl.axis((0,55,-1000,500))
    pl.xlabel('Time (ms)')
    pl.ylabel('Displacement (um)')
    pl.title('Welder/specimen overlap')
    #pl.axis((0,1,-3,1))
    pl.grid()
        
    #pl.figure(7)
    #pl.clf()
    #difft = (trange[:-1]+trange[1:])/2.0
    #pl.plot(difft*1.e3,np.diff(welder_tip_z_history-specimen_z_history)/dt,'-')
    #pl.xlabel('Time (ms)')
    #pl.ylabel('Velocity (m/s)')
    #pl.grid()
    
    contactvel_plot = pl.figure()
    pl.clf()
    pl.plot(difft*1.e3,np.diff(specimen_z_history)/dt,'-')
    #pl.axis((0,55,-1000,500))
    pl.xlabel('Time (ms)')
    pl.ylabel('Velocity (m/s)')
    pl.title('Specimen contact surface velocity (synthetic vibrometer)')
    #pl.axis((0,1,-3,1))
    pl.grid()
    
    contactvelspec_plot = pl.figure()
    pl.clf()
    pl.plot(frange/1.e3,np.abs(np.fft.fft(specimen_z_history)*dt*(2.0*np.pi*frange)),'-')
    #pl.axis((0,55,-1000,500))
    pl.xlabel('Frequency (kHz)')
    pl.ylabel('Velocity spectrum (m/s)/Hz')
    pl.title('Specimen contact velocity spectrum (synthetic vibrometer)')
    #pl.axis((0,1,-3,1))
    pl.grid()
    

    plotdict = {
        "impulse_response": impresp_plot,
        "velocity_spectrum": velspec_plot,
        "phase_spectrum": phasespec_plot,
        "displacement": disp_plot,
        "displacement_zoom": dispzoom_plot,
        "contact_force": contactforce_plot,
        "contact_spectrum": contactspectrum_plot,
        "overlap": overlap_plot,
        "contact_velocity": contactvel_plot,
        "contact_velocity_spectrum": contactvelspec_plot
    }
    return plotdict
