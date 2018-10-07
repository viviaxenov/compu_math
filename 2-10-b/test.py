import numpy as np

import gauss_method as gm
import zeydel_method as zm
import helpers as hlp

n = 90
eps=1e-10

A, f = hlp.get_system(n) 
u = np.linalg.solve(A, f) 
ev = np.linalg.eigvals(A)
det = np.linalg.det(A)
inv = np.linalg.inv(A)
mu = np.linalg.norm(A)*np.linalg.norm(inv)		## frobenius norm
							## 
print("System of Linear equations Au = f, where:\n")
print("A = ", A)
print("f = ", f)
print("Matrix properties:\n")
print("mu = {0:.5g}\n\\lambda_max = {1:.5g}, \\lambda_min = {2:.5g}"
	.format(mu, np.amax(ev), np.amin(ev)))
print("Reference solution (by numpy.linalg.solve):\n")
print(u)
g_u, g_inv = gm.solve(A, f, full=True)
print("Gauss method yeilded:\n")
print(g_u)
print("Norm of difference with the reference: {0:0.3e}".format(np.linalg.norm(u - g_u)))
z_u, res, delta, n = zm.solve(A, np.zeros(f.shape), f, eps=eps, full=True)
print("Zeydel method yeilded ({0:d} iterations):\n".format(n))
print(z_u)
print("Norm of difference with the reference: {0:0.3e}".format(np.linalg.norm(u - g_u)))
print("||Au - f|| = {0:0.3e}, ||u_k+1 - u_k|| = {1:0.3e}".format(np.linalg.norm(res), np.linalg.norm(delta)))




