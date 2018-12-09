import numpy as np
import methods.shuttle as sh

u = np.array([1.0,1.0])
m = np.array([2.0,2.0,2.0])
l = np.array([1.0,1.0])

A = np.array([	[2.0,1.0,0],
		[1.0,2,1.0],
		[0,1.0,2.0]])

f = np.array([3.0,7.0,5.0])
print(sh.solve(u,m,l,f))

print(np.linalg.solve(A,f))
