import pyopencl as cl  
import pyopencl.array as pycl_array  
import numpy as np  
import os
import time
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


print("11111111111111111111111111111111111111")
context = cl.create_some_context() 
queue = cl.CommandQueue(context)


a = pycl_array.to_device(queue, np.random.rand(50000).astype(np.float32))
b = pycl_array.to_device(queue, np.random.rand(50000).astype(np.float32))  
c = pycl_array.empty_like(a) 

program = cl.Program(context, """
__kernel void sum(__global const float *a, __global const float *b, __global float *c)
{
  int i = get_global_id(0);
  c[i] = a[i] + b[i];
}""").build() 

start = time.clock()
program.sum(queue, a.shape, None, a.data, b.data, c.data) 
print(time.clock() - start)

print("a: {}".format(a))
print("b: {}".format(b))
print("c: {}".format(c))
