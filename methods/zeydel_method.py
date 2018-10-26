import numpy as np

import methods.helpers as hlp

max_iter = 15000
def_eps = 1e-10

def _iteration(A : np.ndarray, u : np.ndarray, f : np.ndarray) -> np.ndarray:
	"""
	Does a Zeydel method iteration
	Args:
		A - equation matrix
		u_0 - initial approximation
		f - right part
	Returns:
		u_res - next step of iterational process
	Warning: this function doesn't check arguments' corretness!!! Please, use hlp.verify_args()
	"""
	n = A.shape[0]
	u_res = np.zeros(u.shape)
	for i in range(n):
		for j in range(i):
			u_res[i] += A[i, j]*u_res[j]		## u_i^k+1 += a_ij*u^k+1_j j < i
		for j in range(i + 1, n):
			u_res[i] += A[i, j]*u[j]		## u_i^k+1 += a_ij*u^k_j j > i
		u_res[i] -= f[i]
		u_res[i] *= -1/A[i, i]
	return u_res 

def residual(A : np.ndarray, u : np.ndarray, f : np.ndarray) -> np.float64:
	return np.linalg.norm(np.dot(A, u) - f)	

def solve(A : np.ndarray, u_0 : np.ndarray, f : np.ndarray, eps : np.float64=def_eps, full : bool=False): 
	"""
	Solves equation with Zeydel iterational method
	Args:
		A - equation matrix
		u_0 - initial approximation
		f - right part
	kwargs:
		eps - precision
		full - if full=True, returns additional data 
	Returns:
		u - solution of the equation (by default)
		if full=True, returns additional data such as residual A@u - f, difference between
			last iteration results u_cur - u_prev, number of iterations n
	Raises:
		ValueError - if shapes of A, u, f don't correspond
		ValueError - if number of iteration exceeds maximal (zm.max_iter)
	"""
	hlp.verify_args(A, f)
	input_shape = f.shape
	f = f.reshape([f.shape[0], 1])
	u_0 = u_0.reshape([u_0.shape[0], 1])
	if(A.shape[0] != u_0.shape[0]):
		raise ValueError("Matrix and solution vector should have same size; Having {0:d} and {1:d}"
					.format(A.shape[0], u.shape[0]))
	check = True
	i = 0
	u_cur = u_0
	while(check):		
		u_prev = u_cur
		u_cur = _iteration(A, u_cur, f)

		res = np.linalg.norm(np.dot(A, u_cur) - f)
		delta = np.linalg.norm(u_cur - u_prev)

		check = not(res <= eps and delta <= eps)	## iterations end criteria - residual and delta is lower
								## than required precison eps
		i += 1
		if(i >= max_iter):
			raise RuntimeError("Number of iterations exceeded limit {0:d}; Aborting"
					.format(i))
	if(full):
		res = np.dot(A, u_cur) - f
		delta = u_cur - u_prev
		u_cur = u_cur.reshape(input_shape)
		res = res.reshape(input_shape)
		delta = delta.reshape(input_shape)
		return u_cur, res, delta, i
	return u_cur.reshape(input_shape)
