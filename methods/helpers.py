import numpy as np

def verify_args(A : np.ndarray, f : np.ndarray) -> bool:
	shape = A.shape
	if(shape[0] != shape[1]):
		raise ValueError("Matrix should have shape [n,n]; Having [{0:d},{1:d}]".format(*shape))
	if(shape[0] != f.shape[0]):
		raise ValueError("Matrix and right part should have same size; Having {0:d} and {1:d}"
					.format(shape[0], f.shape[0]))
	return True
