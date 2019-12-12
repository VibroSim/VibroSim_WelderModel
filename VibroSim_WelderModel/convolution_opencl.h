#ifdef CONVOLUTION_ENABLE_OPENCL
#include <CL/opencl.h>
#include <assert.h>

// This is needed because pyopencl doesn't provide access to clEnqueueWriteBuffer() 
static void opencl_update_history_element(void *queue_ptr, void *buffer_ptr,unsigned indexoffset,unsigned elsize,void *hist_array)
{
  cl_command_queue queue;
  cl_mem buffer;
  unsigned offset;
  
  queue=(cl_command_queue)queue_ptr;
  buffer=(cl_mem)buffer_ptr;

  offset=indexoffset*elsize;
  clEnqueueWriteBuffer(queue,buffer,1,offset,elsize,((char *)buffer_ptr)+offset,0,NULL,NULL);
  
}

static int convolution_have_opencl(void)
{
  return 1;
}


#else // CONVOLUTION_ENABLE_OPENCL

void opencl_update_history_element(void *queue_ptr, void *buffer_ptr,unsigned indexoffset,unsigned elsize,void *hist_array)
{

  assert(0);
}

int convolution_have_opencl(void)
{
  return 0;
}
#endif
