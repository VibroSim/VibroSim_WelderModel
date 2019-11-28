# To use:
# import pyximport
# pyximport.install()
# import convolution
 

# OLD, DON'T DO THIS to rebuild: cythonize -i convolution.pyx

import copy
cimport numpy as np
import numpy as np
import scipy
import scipy.signal
 
#from libc.math cimport exp as exp_c
from libc.stdint cimport int64_t
#from libc.stdio cimport printf

cdef extern from "CL/opencl.h":
  ctypedef int cl_int
  ctypedef void *cl_command_queue
  ctypedef void *cl_mem
  ctypedef int cl_bool
  ctypedef unsigned cl_uint
  ctypedef void *cl_event

  cl_int clEnqueueWriteBuffer(cl_command_queue queue,
                              cl_mem buffer,
                              cl_bool blocking_write,
                              size_t offset,
                              size_t cb,
                              const void *ptr,
                              cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list,
                              cl_event *event)
                              
cdef extern from "complex.h" nogil:
    double complex cexp(double complex)
    pass

cdef extern from "convolution_c.h":
    cdef double rolled_inner_product_with_timereverse_c(double *history,long history_len,long rollshift_c,double *h,long h_len) 
    pass

def rolled_inner_product_with_timereverse(np.ndarray[np.float64_t,ndim=1,mode='c'] history,rollshift,np.ndarray[np.float64_t,ndim=1,mode='c'] h):
    cdef long rollshift_c = int(rollshift)
    cdef long history_len = history.shape[0]
    cdef long h_len = h.shape[0]
    
    return rolled_inner_product_with_timereverse_c(<double *>history.data,history_len,rollshift_c,<double *>h.data,h_len)

rolled_inner_product_ocl=r"""

// FLPT will be typedef'd to either float or double at runtime!
typedef int uint32_t;

void __kernel rolled_inner_product(__global FLPT *history,
                                   __global FLPT *hreverse,
                                   __global FLPT *accumulator,
                                  uint32_t iters_per_workitem,
                                  uint32_t n_total,
                                  uint32_t rollshift) 
{
  uint32_t workitem=get_global_id(0);
  uint32_t idx,iter;
  FLPT accum=0.0;

  for (iter=0,idx=workitem*iters_per_workitem;iter < iters_per_workitem && idx < n_total;iter++,idx++) {
    accum += history[(rollshift + idx) % n_total]*hreverse[idx];
  }
  accumulator[workitem] = accum;  
}

"""


class impulse_response(object):
    """ An impulse response with two different representations:
         * As a series of taps h(t) representing short-time behavior
         * As a sum of decaying exponential representing long-time
           behavior
        Interpretation shifts from the former to the latter once the
        former runs out """
    
    h = None # numpy vector representing impulse response over short time
             # If there is no separately defined short term response
             # then h should be length 1 and h[0] = sum(A) 
    dt = None # time step
    t0 = None # time of first sample
    A = None # Vector of complex amplitudes for long term response
    alpha = None # vector of complex exponents for long term response
    D = None # Vector of offsets for long term response
    
    # The variables are intepreted as follows:
    # h[0] = instantaneous impulse response (immediate motion)
    # for t=0..(h.shape[0]-1)*dt, the impulse response
    # is given by h
    #
    # For t=(h.shape[0]-1)*dt..infinity, the impulse response is
    # given (for scalar t) by np.inner(A,exp(-alpha*t))+D,
    # i.e. sum(A_i*exp(-alpha_i*t))+D

    # WARNING: Unless you are careful in how you set up h, A,
    # and alpha, the imaginary part may have different characteristics
    # between the two segments

    def __init__(self,**kwargs):
        for kwarg in kwargs:
            if hasattr(self,kwarg):
                setattr(self,kwarg,kwargs[kwarg])
                pass
            else:
                raise ValueError("Unknown attribute %s" % (kwarg))
            pass

        if self.D is None: # Default to zero offset
            self.D=0.0
            pass

        if self.t0 is None: # Default to zero initial time sample time
            self.t0=0.0
            pass

        if self.A.shape[0]==1 and self.A[0]==0.0 and self.alpha[0]==0.0:
            # dummy exponential decay entries... set alpha[0] to 1.0
            # so we don't divide by zero below
            self.alpha[0]=1.0
            pass
        
        pass

    def resample(self,new_dt):
        # resample to new_dt...

        assert(self.t0==0.0) # resampling where t0 != 0 not yet implemented/tested

        # BUG scipy.signal.resample not intended for transients...
        # should fix resampling function

        if new_dt==self.dt:
            return
        
        # Note: position of h elements is not quite exact
        # (subsample error) ... do not perform resampling repeatedly
        new_h_len = int(np.round(self.h.shape[0]*self.dt*1.0/new_dt))
        #self.h=scipy.signal.resample(self.h,new_h_len)

        # evaluate twice needed length and window and line up
        # beginning and end

        h_resampleable = self.eval(np.arange(self.h.shape[0]*2))
        # window down second half with cos^2
        h_resampleable[self.h.shape[0]:] *= np.cos(np.arange(self.h.shape[0],dtype='d')*np.pi/(2.0*self.h.shape[0]))**2.0
        # ramp second half to match first sample, to be consistent
        # with periodicity assumption of scipy.signal.resample()
        h_resampleable[self.h.shape[0]:] += np.arange(self.h.shape[0],dtype='d')*(h_resampleable[0]/self.h.shape[0])
        
        h_resampled=scipy.signal.resample(h_resampleable,new_h_len*2)

        self.h=np.ascontiguousarray(h_resampled[:new_h_len].astype(np.double))
        
        self.dt=new_dt
        
        pass
        
    def integrate(self):
        # Transform this impulse response into its integral
        assert(self.D==0.0) # Cannot represent integral of a non-zero constant
        self.h=np.cumsum(self.h)*self.dt
        self.t0=self.t0 + self.dt/2.0
        # integral from t=t1...t (A_i * exp(-alpha_i*t)) dt
        # let u = -alpha_i*t
        #    du = -alpha_i*dt
        # integral from t=t1...t (A_i * exp(u)) du/(-alpha_i)
        # -A_i/alpha_i  * integral from t=t1...t exp(u) du
        #  = -A_i/alpha_i (exp(-alpha_i*t) - exp(-alpha_i*t1)
        self.A = -self.A/self.alpha
        self.D = np.sum((self.A/self.alpha)*np.exp(-self.alpha*self.h.shape[0]*self.dt))
        pass

    def differentiate(self):
        # Transform this impulse response into its derivative
        self.D=0.0 # derivatve can have no constant term

        ## !!! No initial transient in derivative... now fixed
        #self.h=np.concatenate((np.array((0.0,),dtype='d'),np.diff(self.h)))/self.dt
        self.h=np.concatenate((np.diff(np.concatenate((np.array((0.0,),dtype='d'),self.h))),np.array((np.sum(self.A*np.exp(-self.alpha*self.h.shape[0]*self.dt))-self.h[-1],),dtype=np.complex)))/self.dt
        self.t0 = self.t0-self.dt/2.0
        # derivative (A_i * exp(-alpha_i*t))/dt
        # let u = -alpha_i*t
        #    du = -alpha_i*dt
        # = A_i * exp(u)*du/dt
        # = A_i * exp(-alpha_i*t) * (-alpha_i)
        # = -alpha_i * A_i * exp(-alpha_i*t) 
        
        self.A = -self.A*self.alpha
        pass

    def evalshift(self,plusonehalf_or_minusonehalf,tcnt):
        droponesample=False
        
        if plusonehalf_or_minusonehalf==0.5:
            assert(self.t0==0.5*self.dt or self.t0==-0.5*self.dt)
            if self.t0==-0.5*self.dt:
                droponesample=True
                pass
            pass        
        else:
            assert(plusonehalf_or_minusonehalf==-0.5)
            assert(self.t0==-0.5*self.dt)
            pass
        
        
        
        res=np.zeros(tcnt.shape,dtype='d')
        h_portion = (tcnt < self.h.shape[0]-droponesample)

        res[h_portion]=self.h[tcnt[h_portion]+droponesample]

        exp_portion = (tcnt >= self.h.shape[0]-droponesample)
        exp_portion_t = (plusonehalf_or_minusonehalf + tcnt[exp_portion])*self.dt

        # inner product
        res[exp_portion] = np.dot(self.A,np.exp(-self.alpha[:,np.newaxis]*exp_portion_t[np.newaxis,:])).real + self.D

        return res
    
    def eval(self,tcnt): # vectorized over tcnt; t intepreted as tcnt*self.dt
        assert(self.t0==0.0) # eval() inoperable if t0 is not 0
        res=np.zeros(tcnt.shape,dtype='d')
        h_portion = (tcnt >= 0) & (tcnt < self.h.shape[0])
        
        res[h_portion]=self.h[tcnt[h_portion]]

        exp_portion = (tcnt >= self.h.shape[0])
        exp_portion_t = tcnt[exp_portion]*self.dt

        # inner product
        res[exp_portion] = np.dot(self.A,np.exp(-self.alpha[:,np.newaxis]*exp_portion_t[np.newaxis,:])).real + self.D
        
        return res
    def _array_repr(self,array):
        return "numpy.array(%s,dtype=numpy.%s)" % (np.array2string(array,separator=',',suppress_small=False,threshold=np.inf,floatmode='unique'),repr(array.dtype))
    
    def representation(self):
        retval = r"""convolution.impulse_response(h=%s,
                                                  dt=%.20g,
                                                  t0=%.20g,
                                                  A=%s,
                                                  D=%.20g,
                                                  alpha=%s)
                  """ % (self._array_repr(self.h),
                         self.dt,
                         self.t0,
                         self._array_repr(self.A),
                         self.D,
                         self._array_repr(self.alpha))
        return retval
    
    @classmethod
    def from_component_estimator(cls,h,dt,blendwidth,ce_params_result):
        num_sinusoids=ce_params_result.shape[1]
        ir = cls(h=copy.deepcopy(h),
                   dt=dt,
                   A=np.concatenate((ce_params_result[0,:(num_sinusoids//2)],(0+1j)*ce_params_result[0,(num_sinusoids//2):])),
                   alpha=np.array((ce_params_result[1,:(num_sinusoids//2)]**2.0,ce_params_result[1,:(num_sinusoids//2)]**2.0),dtype='d').reshape(num_sinusoids) + (0+1j)*2.0*np.pi*np.array((ce_params_result[1,(num_sinusoids//2):],ce_params_result[1,(num_sinusoids//2):]),dtype='d').reshape(num_sinusoids))

        # blend latter part of h 

        htime = ir.h.shape[0]*dt
        blendstart_index = int((htime-blendwidth)//dt)
        blend_indexes = np.arange(blendstart_index,ir.h.shape[0])
        
        blend_t = blend_indexes*ir.dt
        # inner product
        exp_blend = np.dot(ir.A,np.exp(-ir.alpha[:,np.newaxis]*blend_t[np.newaxis,:]))
        h_blend=ir.h[blend_indexes]
        blend_factor=np.arange(blend_indexes.shape[0],dtype='d')/blend_indexes.shape[0]  # linear ramp from 0...1
        
        ir.h[blend_indexes] = exp_blend*blend_factor + h_blend*(1.0-blend_factor)
        
        return ir
        
    pass

class convolution_evaluation(object):
    # NOTE: Definitions below correspond to
    # interpretations BETWEEN calls to
    # step()... t corresponds to to
    # time of LAST call to step()

    # This evaluates, step-by-step the
    # convolution of some function F(t)
    # with a known impulse response given by imp_resp
    
    imp_resp=None # impulse_response object
    history=None # Circular history buffer representing the most
                 # recent n samples we are convolving with,
                 # where n is the length of imp_resp.h
    history_nextpos=None # Next position to use in history buffer
    last_exponential_innerproduct = None
                         # Length m array representing             
                         #   sum_tau=-infty..(t-n*dt) F(tau)A_j*exp(-alpha_j*(t-tau))*imp_resp.dt
                         # over j=1..m (no sum over j), where n = imp_resp.h.shape[0]
    real=None
    gpu_context_device_queue=None # Either None or a (pyopencl Context,pyopencl Queue) tuple
    gpu_precision=None # Must be specified as either "double" or "single" if gpu_context_device_queue is given

    gpu_program=None
    gpu_kernel=None
    
                         
    def __init__(self,**kwargs):
        for kwarg in kwargs:
            if hasattr(self,kwarg):
                setattr(self,kwarg,kwargs[kwarg])
                pass
            else:
                raise ValueError("Unknown attribute %s" % (kwarg))
            pass
        if self.real is None:
            self.real=True;
            pass
        assert(self.imp_resp.t0==0.0) # only designed/tested where t0==0.0

 
        if self.gpu_context_device_queue is not None:
            import pyopencl as cl
            
            assert(self.gpu_precision is not None) # gpu_precision must be specified if GPU context/queue are given

            if self.gpu_precision=="double":
                self.gpu_program=cl.Program(self.gpu_context_device_queue[0],"typedef double FLPT;\n"+rolled_inner_product_ocl)
                floattype = np.float64
                pass
            else:
                assert(self.gpu_precision=="single")
                self.gpu_program=cl.Program(self.gpu_context_device_queue[0],"typedef float FLPT;\n"+rolled_inner_product_ocl)
                floattype = np.float32
                pass

            self.history=np.ascontiguousarray(self.history.astype(floattype))

            self.gpu_program.build()
            self.gpu_kernel=cl.Kernel(self.gpu_program,"rolled_inner_product")
            self.gpu_kernel.set_scalar_arg_dtypes([None,None,None,np.uint32,np.uint32,np.uint32])

            self.gpu_work_group_size = self.gpu_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE,self.gpu_context_device_queue[1])
            work_group_size_multiple = self.gpu_kernel.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,self.gpu_context_device_queue[1])

            n = self.imp_resp.h.shape[0]
            assert(n==self.history.shape[0])
            self.gpu_workitems = self.gpu_work_group_size * work_group_size_multiple
            self.gpu_iters_per_workitem = int(np.ceil(n/self.gpu_workitems))

            print("work_group_size = %d; size_multiple=%d; n=%d; workitems=%d; iters_per_workitem=%d" % (self.gpu_work_group_size,work_group_size_multiple,n,self.gpu_workitems,self.gpu_iters_per_workitem)) 
            
            self.gpu_history_buffer=cl.Buffer(self.gpu_context_device_queue[0],cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR,hostbuf=self.history)

            self.gpu_hreverse_buffer=cl.Buffer(self.gpu_context_device_queue[0],cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR,hostbuf=np.ascontiguousarray(self.imp_resp.h[::-1].astype(floattype)))
            self.gpu_accumulator = np.zeros(self.gpu_workitems,dtype=floattype)
            self.gpu_accumulator_buffer=cl.Buffer(self.gpu_context_device_queue[0],cl.mem_flags.WRITE_ONLY,size=self.gpu_accumulator.nbytes)

            pass
        
        pass

    def gpu_update_history_element(self,elnum):
        # pyopencl doesn't provide access to clEnqueueWriteBuffer()
        # so we have to call it manually instead
        cdef np.ndarray[dtype=np.float32_t,ndim=1,mode='c'] floathist_array
        cdef np.ndarray[dtype=np.float64_t,ndim=1,mode='c'] doublehist_array
        cdef cl_command_queue c_queue
        cdef cl_mem c_mem
        cdef unsigned long long c_queue_int_ptr
        cdef unsigned long long c_history_buffer_int_ptr
        cdef double *double_ptr
        cdef float *float_ptr
        cdef unsigned elnum_c
        queue=self.gpu_context_device_queue[2]
        queue_int_ptr = queue.int_ptr

        #import pyopencl as cl
        #cl.enqueue_copy(queue,self.gpu_history_buffer,self.history,is_blocking=True)
        
        #print("gpu_update_history_element(%d)=%g" % (elnum,self.history[elnum]))
        
        #print("queue_int_ptr=0x%x" % (queue_int_ptr))        
        history_buffer_int_ptr = self.gpu_history_buffer.int_ptr
        
        #print("history_buffer_int_ptr=0x%x" % (history_buffer_int_ptr))

        c_queue_int_ptr = queue_int_ptr
        c_history_buffer_int_ptr = history_buffer_int_ptr
        
        c_queue = <void *>c_queue_int_ptr
        c_mem = <void *>c_history_buffer_int_ptr

        elnum_c = elnum
        if self.gpu_precision=="double":
            doublehist_array=self.history
            #print("elnum=%d" % (elnum))
            #print("doublehist_array.itemsize=%d" % (doublehist_array.itemsize))
            double_ptr = <double *>doublehist_array.data
            double_ptr = double_ptr + elnum_c # EnqueueWriteBuffer wants a pointer just to the new data item 
            clEnqueueWriteBuffer(c_queue,c_mem,1,elnum*doublehist_array.itemsize,doublehist_array.itemsize,double_ptr,0,NULL,NULL)
            pass
        else:
            floathist_array=self.history
            float_ptr = <float *>floathist_array.data
            float_ptr = float_ptr + elnum_c # EnqueueWriteBuffer wants a pointer just to the new data item 
            clEnqueueWriteBuffer(c_queue,c_mem,1,elnum*floathist_array.itemsize,floathist_array.itemsize,float_ptr,0,NULL,NULL)
            pass
        pass
    
        
    def stepwise_step(self,new_F):
        # perform step by doing first with no
        # instantaneous value, then by adding
        # in that value....
        #
        # Should be identical to just calling step()
        res1 = self.step_without_instantaneous()

        res2 = self.step_instantaneous_contribution(new_F)

        return res1+res2
        
    def step_without_instantaneous(self):  # new_t actually unnecessary 
        """ Call this to step forward by imp_res.dt
        where we don't yet know the instantaneous force
        applied in the current step. i.e. treat new_F as 0.0"""
        return self.step(0.0)

    def step_instantaneous_contribution(self,new_F): 
        """ Call this to add in an instantaneous 
        contribution to a prior call to step_without_instantaneous"""
        old_history_nextpos = (self.imp_resp.h.shape[0]+self.history_nextpos-1) % self.imp_resp.h.shape[0]

        self.history[old_history_nextpos] = new_F
        if self.gpu_context_device_queue is not None:
            self.gpu_update_history_element(old_history_nextpos)
            pass

        # return instantaneous contribution
        if self.real:
            return np.real(new_F*self.imp_resp.h[0]*self.imp_resp.dt)
        else:
            return new_F*self.imp_resp.h[0]*self.imp_resp.dt

        pass
    
    
    def step(self,new_F): # new_t actually unnecessary
        """ Call this to step forward by imp_res.dt
        If new_t - old_t is not essentially equal to 
        imp_res.dt, the behavior is undefined"""
        
        veryold_F = self.history[self.history_nextpos] # extract old F from h(t) portion for use in exponential portion
        
        # For h(t) portion, 
        # Convolution = sum_tau=(t-n*dt)..t F(tau)h(t-tau)*dt
        self.history[self.history_nextpos] = new_F

        if self.gpu_context_device_queue is not None:
            self.gpu_update_history_element(self.history_nextpos)
            pass
        
        # increment history_nextpos
        self.history_nextpos = (self.history_nextpos+1) % self.imp_resp.h.shape[0]
        
        #F_last_n_dt = np.roll(self.history,-self.history_nextpos)
        #res = np.inner(F_last_n_dt,self.imp_resp.h[::-1])*self.imp_resp.dt

        if self.gpu_context_device_queue is not None:
            import pyopencl as cl
            
            (context,device,queue) = self.gpu_context_device_queue
            n=self.history.shape[0]
            assert(self.gpu_workitems==self.gpu_accumulator.shape[0])
            
            innerprod_ev = self.gpu_kernel(queue,(self.gpu_workitems,),(self.gpu_work_group_size,),self.gpu_history_buffer,self.gpu_hreverse_buffer,self.gpu_accumulator_buffer,self.gpu_iters_per_workitem,n,self.history_nextpos);

            # copy result accumulator
            cl.enqueue_copy(queue,self.gpu_accumulator,self.gpu_accumulator_buffer,wait_for=(innerprod_ev,))
            queue.finish()

            res=np.sum(self.gpu_accumulator)*self.imp_resp.dt

            #res2 = rolled_inner_product_with_timereverse(self.history,self.history_nextpos,self.imp_resp.h)*self.imp_resp.dt

            #if res != res2:
            #    print("gpu_accum*dt = %s" % (str(self.gpu_accumulator[:n]*self.imp_resp.dt)))
            #    F_last_n_dt = np.roll(self.history,-self.history_nextpos)
            #    ref = (F_last_n_dt*self.imp_resp.h[::-1])*self.imp_resp.dt
            #    print("ref          = %s" % (str(ref)))
            #    print("ref.shape=%s; gpu_accum.shape=%s" % (str(ref.shape),str(self.gpu_accumulator.shape)))
            #    print("res=%g; res2=%g" % (res,res2))
            #    pass
            pass
        else:

            if self.history.shape[0] > 5000:
                res = rolled_inner_product_with_timereverse(self.history,self.history_nextpos,self.imp_resp.h)*self.imp_resp.dt
                pass
            else:
                F_last_n_dt = np.roll(self.history,-self.history_nextpos)
                res = np.inner(F_last_n_dt,self.imp_resp.h[::-1])*self.imp_resp.dt
                pass

            pass
        
        #print("terms=%s,factor1=%s,factor2=%s" % (str(F_last_n_dt*self.imp_resp.h[::-1]),str(F_last_n_dt),str(self.imp_resp.h[::-1])))
        # in quiescent equilibrium,
        # history=constant_force, so res = integral(constant_force*h)dt
        #print("res = %g" % (res))
        # exponential portion
        new_exponential_innerproduct = self.last_exponential_innerproduct*np.exp(-self.imp_resp.alpha*self.imp_resp.dt) + veryold_F*self.imp_resp.A*np.exp(-self.imp_resp.alpha*self.imp_resp.dt*self.imp_resp.h.shape[0])*self.imp_resp.dt

        # in quiescent equilibrium,
        # new_exponential_innerproduct = last_exponential_innerproduct
        # so exponential_innerproduct = exponential_innerproduct*np.exp(-self.imp_resp.alpha*self.imp_resp.dt) + constant_force*self.imp_resp.A*np.exp(-self.imp_resp.alpha*self.imp_resp.dt*self.imp_resp.h.shape[0])*self.imp_resp.dt
        # or exponential_innerproduct*(1 - np.exp(-self.imp_resp.alpha*self.imp_resp.dt)) = constant_force*self.imp_resp.A*np.exp(-self.imp_resp.alpha*self.imp_resp.dt*self.imp_resp.h.shape[0])*self.imp_resp.dt
        # or exponential_innerproduct = constant_force*self.imp_resp.A*np.exp(-self.imp_resp.alpha*self.imp_resp.dt*self.imp_resp.h.shape[0])*self.imp_resp.dt/(1 - np.exp(-self.imp_resp.alpha*self.imp_resp.dt))
        # or exponential_innerproduct = constant_force*self.imp_resp.A*np.exp(-self.imp_resp.alpha*self.imp_resp.dt*self.imp_resp.h.shape[0])*self.imp_resp.dt/(1 - np.exp(-self.imp_resp.alpha*self.imp_resp.dt))
        
        res+=np.sum(new_exponential_innerproduct)
        #print("res now %g" % (res))

        # in quiescent equilibrium,
        # res=integral(constant_force*h)dt + sum(exponential_innerproduct)
        #    =constant_force*(integral(h)dt + sum(A*exp(-alpha*dt*h.shape[0])/(1.0 - exp(-alpha*dt)))*dt

        self.last_exponential_innerproduct=new_exponential_innerproduct

        if self.real:
            return np.real(res)
        else:
            return res
        pass

    @classmethod
    def quiescent_exponential_innerproduct(cls,constant_force,A,alpha,dt,h):
        # see comment in step() above
        return constant_force*A*np.exp(-alpha*dt*h.shape[0])*dt/(1.0 - np.exp(-alpha*dt))

    @classmethod
    def quiescent_value(cls,constant_force,A,alpha,dt,h):
        transient_portion = np.sum(h*constant_force)*dt
        exponential_portion = np.sum(convolution_evaluation.quiescent_exponential_innerproduct(constant_force,A,alpha,dt,h))

        return transient_portion + exponential_portion

    @classmethod
    def blank_from_imp_resp(cls,imp_resp,real=True,gpu_context_device_queue=None,gpu_precision=None):
        return cls(imp_resp=imp_resp,
                   history=np.zeros(imp_resp.h.shape[0],dtype='d'),
                   history_nextpos=0,
                   last_exponential_innerproduct=np.zeros(imp_resp.A.shape[0],dtype='D'),
                   real=real,
                   gpu_context_device_queue=gpu_context_device_queue,
                   gpu_precision=gpu_precision)
    
    @classmethod
    def quiescent_from_imp_resp(cls,imp_resp,constant_force,real=True,gpu_context_device_queue=None,gpu_precision=None):
        return cls(imp_resp=imp_resp,
                   history=np.ones(imp_resp.h.shape[0],dtype='d')*constant_force,
                   history_nextpos=0,
                   last_exponential_innerproduct=cls.quiescent_exponential_innerproduct(constant_force,imp_resp.A,imp_resp.alpha,imp_resp.dt,imp_resp.h),
                   real=real,
                   gpu_context_device_queue=gpu_context_device_queue,
                   gpu_precision=gpu_precision)
    

    @classmethod
    def convolve_purepy(cls,imp_resp,fcn,initial_exponential_innerproduct=None,initial_history=None):
        # currently throws out stuff beyond the bounds of fcn
        #print("convolve_purepy_start")
        conv_eval=convolution_evaluation.blank_from_imp_resp(imp_resp)
        if initial_exponential_innerproduct is not None:
            conv_eval.last_exponential_innerproduct[:]=initial_exponential_innerproduct
            pass
        if initial_history is not None:
            conv_eval.history[:]=initial_history
            pass
        
        conv_evald = np.zeros(fcn.shape[0],dtype='d')

        for tidx in np.arange(fcn.shape[0]):
            #tpos=tidx*dt
            conv_evald[tidx] = conv_eval.step(fcn[tidx]).real
            pass
        return conv_evald

    @classmethod # WARNING: Known to be buggy when initial_exponential_innerproduct and initial_history are specified -- see contact_model_integral.py
    def convolve(cls,imp_resp,np.ndarray[np.float64_t,ndim=1,mode='c'] fcn,initial_exponential_innerproduct=None,initial_history=None):
        # currently throws out stuff beyond the bounds of fcn

        cdef np.ndarray[np.float64_t,ndim=1,mode="c"] history
        cdef np.ndarray[np.float64_t,ndim=1,mode="c"] h=np.ascontiguousarray(imp_resp.h,dtype='d')
        cdef double dt = imp_resp.dt
        cdef double veryold_F

        cdef np.ndarray[np.complex128_t,ndim=1,mode="c"] alpha=np.asarray(imp_resp.alpha,dtype='D')
        cdef np.ndarray[np.complex128_t,ndim=1,mode="c"] A=np.asarray(imp_resp.A,dtype='D')
        
        cdef int64_t history_nextpos=0

        cdef np.ndarray[np.complex128_t,ndim=1,mode="c"] last_exponential_innerproduct
        cdef np.ndarray[np.complex128_t,ndim=1,mode="c"] new_exponential_innerproduct=np.zeros(imp_resp.A.shape[0],dtype='D')
        cdef np.ndarray[np.float64_t,ndim=1,mode="c"] conv_evald = np.zeros(fcn.shape[0],dtype='d')

        cdef double *fcn_c = <np.float64_t *>fcn.data
        cdef double *history_c
        cdef double *h_c = <np.float64_t *>h.data
        cdef double complex *alpha_c = <np.complex128_t *>alpha.data 
        cdef double complex *A_c = <np.complex128_t *>A.data 
        cdef double complex *last_exponential_innerproduct_c
        cdef double complex *new_exponential_innerproduct_c = <np.complex128_t *>new_exponential_innerproduct.data 
        cdef double  *conv_evald_c = <np.float64_t *>conv_evald.data

        cdef int64_t nsteps=fcn.shape[0]
        cdef int64_t nh=imp_resp.h.shape[0]
        cdef int64_t nalpha=alpha.shape[0]
        cdef int64_t hpos
        cdef int64_t histpos
        cdef int64_t rolledhistpos
        cdef int64_t alphacnt

        cdef int64_t stepnum
        cdef double res

        #print("h=%s h.dtype=%s; h.flags=%s" % (str(h),str(h.dtype),str(h.flags)))
        #print("imp_resp.h.dtype=%s" % (str(imp_resp.h.dtype)))
        if initial_exponential_innerproduct is not None:
            last_exponential_innerproduct=np.array(initial_exponential_innerproduct,dtype='D')
            assert(last_exponential_innerproduct.shape[0]==imp_resp.A.shape[0])
            pass
        else:
            last_exponential_innerproduct=np.zeros(imp_resp.A.shape[0],dtype='D')
            pass
        
        if initial_history is not None:
            history=np.array(initial_history,dtype='d')
            assert(history.shape[0]==imp_resp.h.shape[0])
            pass
        else:
            history=np.zeros(imp_resp.h.shape[0],dtype='d')
            pass


        history_c = <np.float64_t *>history.data
        last_exponential_innerproduct_c = <np.complex128_t *>last_exponential_innerproduct.data
        
        assert(A.shape[0]==nalpha)
        
        with nogil:
            #printf("convolution\n")

            for stepnum in range(nsteps):
                #tpos=tidx*dt                
                #conv_evald[tidx] = conv_eval.step(fcn[tidx]).real

                veryold_F = history_c[history_nextpos] # extract old F from h(t) portion for use in exponential portion
                # Convolution = sum_tau=(t-n*dt)..t F(tau)h(t-tau)*dt
                history_c[history_nextpos] = fcn_c[stepnum]

                # increment history_nextpos
                history_nextpos = (history_nextpos+1) % nh

                
                #F_last_n_dt = np.roll(self.history,-self.history_nextpos)
                #res = np.inner(F_last_n_dt,self.imp_resp.h[::-1])*self.imp_resp.dt
                
                res=0.0
                for histpos in range(nh):
                    hpos = nh-histpos-1
                    rolledhistpos=histpos+history_nextpos
                    #printf("rolledhistpos=%ld\n",rolledhistpos)
                    if rolledhistpos >= nh:
                        rolledhistpos -= nh
                        pass
                    #printf("term = %g = %g * %g\n",history_c[rolledhistpos]*h_c[hpos],history_c[rolledhistpos],h_c[hpos])
                    res += history_c[rolledhistpos]*h_c[hpos]
                    pass
                res = res*dt

                #printf("res = %g\n",res)
                # exponential portion
                #new_exponential_innerproduct = self.last_exponential_innerproduct*np.exp(-self.imp_resp.alpha*self.imp_resp.dt) + veryold_F*self.imp_resp.A*np.exp(-self.imp_resp.alpha*self.imp_resp.dt*self.imp_resp.h.shape[0])*self.imp_resp.dt
                #res+=np.sum(new_exponential_innerproduct)

                for alphacnt in range(nalpha):
                    new_exponential_innerproduct_c[alphacnt] = last_exponential_innerproduct_c[alphacnt]*cexp(-alpha_c[alphacnt]*dt) + veryold_F*A_c[alphacnt]*cexp(-alpha_c[alphacnt]*dt*(<double>nh))*dt
                    res += new_exponential_innerproduct_c[alphacnt].real
                    pass
                
                #printf("res now %g\n",res)
                #self.last_exponential_innerproduct=new_exponential_innerproduct
                for alphacnt in range(nalpha):
                    last_exponential_innerproduct_c[alphacnt]=new_exponential_innerproduct_c[alphacnt]
                    pass
                
                conv_evald_c[stepnum]=res                
                
                pass
            pass
        return conv_evald

    pass



# ... if run as a script...
if __name__=="__main__":
    # Perform tests
    # ***!!! Primary copy of these have been moved to
    # convolution_tests.py because running a cython module as a script
    # is painful

    from matplotlib import pyplot as pl
    
    # ... These are based on an early cut of a welder model
    # ... We had
    A1_norm=-0.00248475
    alpha1 = (6.0+125437.51147253325j)
    alpha2 = (138.79999999999927+133856.3514656232j)
    A2_norm = -0.000591607
    # as the constant for the first 12.5 us
    syn_time_delay = 12.5e-6
    
    dt = 2e-6
    # The A1_norm and A2_norm were for a model
    # that included an extra exp(alpha*syn_time_delay) factor
    # ... we need to work this into A1 and A2
    A1_mod = A1_norm*np.exp(alpha1*syn_time_delay)
    A2_mod = A2_norm*np.exp(alpha2*syn_time_delay)

    h = np.ones(int(np.floor(syn_time_delay/dt)),dtype='D')*(A1_norm+A2_norm)

    #A1_mod=0.0
    #A2_mod=0.0
    
    imp_resp=impulse_response(h=h,
                              dt=dt,
                              A=np.array((A1_mod,A2_mod),dtype='D'),
                              alpha=np.array((alpha1,alpha2),dtype='D'))

    tcnt=np.arange(300000)
    t = tcnt*dt
    
    full_h = imp_resp.eval(tcnt)

    pl.figure(1)
    pl.clf()
    pl.plot(t,np.real(full_h),'-')

    # chirp: f = f0 + t*chirp_rate
    # i.e. cos(2*pi*(f0+t*chirp_rate)*t)

    f0=20.0 # Hz
    chirp_rate = 30000.0 # Hz/s
    chirp = np.cos(2.0*np.pi*(f0 + t*chirp_rate)*t)

    pl.figure(2)
    pl.clf()
    pl.plot(t,chirp,'-')
    
    # full convolution, by Numpy... very slow
    chirp_conv=dt*np.convolve(chirp,full_h.real,mode="full")[:tcnt.shape[0]]

    pl.figure(3)
    pl.clf()
    pl.plot(t,chirp_conv,'-')

    #neval=10000
    #neval=tcnt.shape[0]


    # Evaluate convolution using convolution_evaluation class
    
    # This is still not all that fast because it is Python iteration
    # but it is very amenable to optimization (i.e. Cython)
    tidx_poss = np.arange(tcnt.shape[0],dtype='d')*dt
    chirp_conv_eval=convolution_evaluation.blank_from_imp_resp(imp_resp)
    chirp_conv_eval_stepwise=convolution_evaluation.blank_from_imp_resp(imp_resp)
    chirp_conv_evald = np.zeros(tcnt.shape[0],dtype='d')
    chirp_conv_evald_stepwise = np.zeros(tcnt.shape[0],dtype='d')
    for tidx in tcnt:
        #tpos=tidx*dt
        chirp_conv_evald[tidx] = chirp_conv_eval.step(chirp[tidx]).real
        chirp_conv_evald_stepwise[tidx] = chirp_conv_eval_stepwise.stepwise_step(chirp[tidx]).real
        pass

    convolved = convolution_evaluation.convolve(imp_resp,chirp)
    pl.plot(tidx_poss,chirp_conv_evald,'-')

    # verify match
    assert(np.linalg.norm((chirp_conv_evald-chirp_conv)/np.linalg.norm(chirp_conv)) < 1e-8)
    assert(np.linalg.norm((chirp_conv_evald_stepwise-chirp_conv)/np.linalg.norm(chirp_conv)) < 1e-8)
    assert(np.linalg.norm((convolved-chirp_conv)/np.linalg.norm(chirp_conv)) < 1e-8)
    
    pl.show()
    pass
