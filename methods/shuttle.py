import numpy as np

def solve(upper : np.array, main : np.array, lower : np.array, f : np.array) -> np.array:
	"""
	Solves linear system with three-diaginal matrix with shuttle method
	| m_1  u_1   0   ... ||s_1|   |F_1|
	| l_1  u_2  u_2  ... ||s_2|   |F_2|
	| ...  ...  ...  ... ||...| = |...|
	| ...  ... l_n-1 u_n ||s_n|   |F_n|
	Args:
		upper - upper diag
		main - main diag
		lower - lower diag
		f - right part
	Returns:
		s - solution of the system
	Raises:
		
	"""
	diag = np.copy(main)
	F = np.copy(f)
	N = diag.shape[0]
	for i in range(1, N):
		coef = lower[i - 1]/diag[i - 1]
		diag[i] -= upper[i - 1]*coef
		F[i] = F[i] - F[i - 1]*coef
	for i in reversed(range(N - 1)):
		coef =	upper[i]/diag[i + 1]
		F[i] -= F[i + 1]*coef
	return F/diag
