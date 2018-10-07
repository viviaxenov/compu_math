import numpy as np

import helpers as hlp

def solve(A : np.ndarray, f : np.ndarray, 
		full=False, eps=1e-17): 				
	"""
	Solves equation with Gauss method (with choice of elemet along the column)
	Args:
		A - equation matrix
		f - right part
	kwargs:
		eps - precision (needed to compare values with zero)
		full - if full=True, returns additional data 
	Returns:
		u - solution of the equation (by default)
		if full=True, returns solution and inverse matrix 
	Raises:
		ValueError - if shapes of A, f don't correspond
		ValueError - if matrix is singular (e.g. algo fails to find a non-zero element in the column)
	"""
	hlp.verify_args(A, f)
	n = A.shape[0]
	ext = np.concatenate((A, f), axis=1)				## transforming the extended matrix we get solution
	if(full):
		ext = np.concatenate((ext, np.identity(n)), axis=1)	## get inv matrix by doing same transformations
									## with identity matrix
									
	for i in range(n - 1):						## straight cycle
		col = A[i:, i]						## Gauss method with element choice along the column
		k = np.argmax(np.abs(col))				## searching for max abs element which are lower
									## than the current
		k += i							## counting its index
		if((np.abs(A[k, i]) < eps)):
			continue
		ext[k, :], ext[i, :] = ext[i, :], ext[k, :].copy()	## swapping
		for j in range(i + 1, n):
			ext[j, :] -= (ext[j, i]/ext[i, i])*ext[i, :]

	if(np.abs(ext[n - 1, n -1]) < eps):
		raise ValueError('Singular matrix; Aborting') 

	for i in reversed(range(1, n)):					## backwards cycle
		for j in reversed(range(i)):
			ext[j, :] -= (ext[j,i]/ext[i,i])*ext[i, :]
	for i in range(n):
		ext[i,] /= ext[i,i]	
	u = ext[:, n]							## solution
	u = u.reshape(f.shape)
	if(full):
		inv = ext[:, n + 1 :]
		return u, inv
	return u
		
