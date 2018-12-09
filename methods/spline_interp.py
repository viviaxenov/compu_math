import numpy as np

import methods.shuttle as sh

class CubicSpline:
	"""
	Implements cubic spline interpolation
	"""
	def __init__(t : np.array, f : np.array, bcond):
		"""
		Creates cubic spline S  interpolating given function with condition d^2S/dt^2(t_0) = d^2S/dt^2(t_N) = 0
		Args:
			t - interpolation points
			f - values of function in points t
		"""
		# sorting input points
		perm = np.argsort(t) 			
		t = t[perm]
		f = f[perm]

		tau = t[:-1] - t[1:]	
		diag = (tau[1:-1] - tau[:-2])/3
		subdiag = -tau[:-2]/3
		F = (f[2:] - f[1:-1])/tau[1:] - (f[1:-1] - f[:-2])/tau[:-1]
		
		# solving with shuttle method
		upper = subdiagd
		main = diag
		lower = subdiag
	
		moments = sh.solve(upper, main, lower, F)
		moments = np.concatenate([0], moments, [0])
		
		self.t = t
		self.m= moments
		self.alpha = f[:-1]/tau - moments[:-1]*tau/6
		self.beta = f[1:]/tau - moments[1:]*tau/6
	def __call__(t : np.float64):
		"""
		Calculates spline value at point t in [t_0; t_N]
		Args: 
			t - point to interpolate
		Returns:
			f - spline value
		Raises:
			ValueError - if t is out of [t_0; t_N]
		"""	
		if t < self.t[0] or t > self.t[-1]
			raise(ValueError('Point is out of bounds [t_0; t_N]'))
		# find the segment t belongs to
		i = np.searchsorted(selt.t, t, side='right') 	
		tau = self.t[i + 1] - self.t[i]
		return (self.m[i]*(self.t[i + 1] - t)**3 + self.m[i + 1]*(t - self.t[i])**3)/(6.0*tau) + self.alpha[i]*(self.t[i + 1] - t) + self.beta[i]*(t - self.t[i])
