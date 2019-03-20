import numpy as np


def solve(upper : np.array, main : np.array, lower : np.array, f : np.array) -> np.array:
    """
    Solves linear system with three-diagonal matrix with shuttle method
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
        F[i] -= F[i - 1]*coef
    for i in reversed(range(N - 1)):
        coef = upper[i]/diag[i + 1]
        F[i] -= F[i + 1]*coef
    return F/diag

def solve_cyclic(upper : np.array, main : np.array, lower : np.array, f : np.array, ur: np.float64, bl: np.float64):
    diag = np.copy(main)
    F = np.copy(f)
    u = np.copy(upper)
    N = diag.shape[0]

    right_col = np.zeros_like(diag)
    right_col[0] = ur
    right_col[-2] = u[-1]
    right_col[-1] = diag[-1]

    left_col = np.zeros_like(diag)
    left_col[0] = diag[0]
    left_col[-1] = bl

    for i in range(1, N):
        coef = lower[i - 1]/diag[i - 1]

        diag[i] -= u[i - 1]*coef
        F[i] -= F[i - 1]*coef
        right_col[i] -= right_col[i - 1]*coef

    u[-1] = right_col[-2]

    for i in reversed(range(N - 1)):
        coef = u[i]/diag[i + 1]

        F[i] -= F[i + 1]*coef
        right_col[i] -= right_col[i + 1]*coef
        left_col[i] -= left_col[i + 1]*coef

    diag[0] = left_col[0]

    for i in range(1, N):
        coef = left_col[i]/left_col[0]
        right_col[i] -= coef*right_col[0]
        F[i] -= F[0]*coef

    diag[-1] = right_col[-1]
    for i in range(N - 1):
        coef = right_col[i]/right_col[-1]
        F[i] -= F[-1]*coef

    return F/diag
