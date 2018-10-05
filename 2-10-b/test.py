import numpy as np

import gauss_method as gm
import helpers as hlp


A, f = hlp.get_system(3) 
ev = np.linalg.eigvals(A)
det = np.linalg.det(A)
inv = np.linalg.inv(A)
mu = np.linalg.norm(A)*np.linalg.norm(inv)		## frobenius norm
#print("mu = {0:.5g}\n\\lambda_max = {1:.5g}, \\lambda_min = {2:.5g}"
#	.format(mu, np.amax(ev), np.amin(ev)))

myu, myinv = gm.gauss_solve(A, f, full=True)
u = np.linalg.solve(A, f) 
print(myinv, '\n', inv)
print(myu, '\n', u)
print(np.linalg.norm(inv - myinv))
print(np.linalg.norm(u - myu))


