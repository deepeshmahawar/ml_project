import numpy as np  
import time


a1 = np.random.rand(50000).astype(np.float32)
b1 = np.random.rand(50000).astype(np.float32)
start = time.clock()
c1 = a1 + b1 
print(time.clock() - start)
print("a: {}".format(a1))
print("b: {}".format(b1))
print("c: {}".format(c1))
