import numpy as np

import methods.gauss_method as gm
import methods.zeydel_method as zm


def get_matrix(n : int = 9) -> np.ndarray:
	res = np.zeros([n + 1, n + 1])
	for i in range(n):
		res[i + 1][i] = 1.0
		res[i][i + 1] = 1.0
		res[i][i] = -2.0
	res[0][0] = 1.0	
	res[0][1] = 0.0
	for i in range(n + 1):
		res[n][i] = 2
	return res

def get_right_part(n: int = 9) -> np.ndarray:
	f = np.zeros([n + 1, 1])
	f[0] = 1
	for i in range(1, n):
		f[i] = 2.0/(i + 1)**2
	f[n] = -n/3.0
	return f

def get_system(n : int = 9):
	return get_matrix(n), get_right_part(n)  


n = 9
eps=1e-10

A, f = get_system(n) 
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



