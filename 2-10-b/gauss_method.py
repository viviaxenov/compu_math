import numpy as np

import helpers as hlp

def gauss_solve(A : np.ndarray, f : np.ndarray, full=False): 		## full=True => returning solution and inverse matrix
	hlp.verify_args(A, f)
	n = A.shape[0]
	inv = None							
	if(full):
		inv = np.identity(n)					## get inv matrix by doing same transformations
									## with identity matrix
									## straight cycle
	for i in range(n - 1):
		col = A[i:, i]						## Gauss method with element choice along the column
		k = np.argmax(np.abs(col))				## searching for max abs element which are lower
									## than the current
		k += i							## counting its index
		if((np.abs(A[k, i]) < 1e-15)):
			continue
		A[k, :], A[i, :] = A[i, :], A[k, :].copy()		## swapping
		f[k], f[i] = f[i], f[k]
		if(full):	
			inv[k, :], inv[i, :] = inv[i, :], inv[k, :].copy()	## swapping
		for j in range(i + 1, n):
			A[j, :] -= (A[j, i]/A[i, i])*A[i, :]
			A[j, :] -= (A[j, i]/A[i, i])*A[i, :]
			if(full):	
				f[j] -= (A[j, i]/A[i, i])*f[i]
 									## backwards cycle
	
