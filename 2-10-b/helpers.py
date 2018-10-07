import numpy as np


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

def verify_args(A : np.ndarray, f : np.ndarray) -> bool:
	shape = A.shape
	if(shape[0] != shape[1]):
		raise ValueError("Matrix should have shape [n,n]; Having [{0:d},{1:d}]".format(*shape))
	if(shape[0] != f.shape[0]):
		raise ValueError("Matrix and right part should have same size; Having {0:d} and {1:d}"
					.format(shape[0], f.shape[0]))
	return True
