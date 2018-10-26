import numpy as np

import methods.gauss_method as gm


def lin_fit(x : np.ndarray, y : np.ndarray) -> np.ndarray:
	"""
	Fits function y(x) with linear function a_1x + a_0 with least square 		method.
	Args:
		x - argument array
		y - function values array
	Returns:
		coefs - array [a_1 a_0]
	Raises:
		ValueError - if shapes of x, y don't correspond
	"""
	if(x.shape != y.shape):
		raise ValueError("x and y shapes don't correspond!")

	x = x.reshape(x.shape[0],1)
	A = np.ones_like(x)
	A = np.concatenate((x, A), axis=1)
	B = np.dot(A.T, A)
	F = np.dot(A.T, y)
	
	coefs = gm.solve(B, F)
	coefs = coefs.reshape([coefs.shape[0]])
	return coefs
	
	
