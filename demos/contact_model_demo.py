

# Uncomment these and comment out below to disable GPU use
#gpu_context_device_queue=None
#gpu_precision=None

# Uncomment these and comment out above to enable GPU use
import pyopencl as cl
ctx=cl.create_some_context()
device=ctx.devices[0]
queue=cl.CommandQueue(ctx)
gpu_context_device_queue = (ctx,device,queue)
gpu_precision="single"
