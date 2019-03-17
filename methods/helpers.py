import numpy as np
import scipy as sp
import scipy.linalg

def verify_args(A : np.ndarray, f : np.ndarray) -> bool:
	shape = A.shape
	if(shape[0] != shape[1]):
		raise ValueError("Matrix should have shape [n,n]; Having [{0:d},{1:d}]".format(*shape))
	if(shape[0] != f.shape[0]):
		raise ValueError("Matrix and right part should have same size; Having {0:d} and {1:d}"
					.format(shape[0], f.shape[0]))
	return True


def linear_ode_analytical(A : np.ndarray, y_0 : np.ndarray):
	""""
    Gets analytical solution for ODE system dy/dt = Ay, y(0) = y_0, where A=const
    Args:
        A - right part matrix
        y_0 - initial condition
    Returns:
        u - function  : y(t)
    Warning: works only if A has basis of real eigenvalues. Other cases weren't considered
    """
	eigvals, eigvecs = sp.linalg.eig(A)

	def solution(t : np.ndarray) -> np.ndarray :
		consts = sp.linalg.solve(eigvecs, y_0)
		l = np.diag(eigvals)
		times = np.stack((t,t))
		exps = np.exp(l@times)
		consts = np.diag(consts)
		cords = consts@exps
		return eigvecs@cords
	return solution
