import numpy as np
import matplotlib.pyplot as plt

def find_root(f, left: np.float64, right: np.float64, prec: np.float64 = 1e-7, max_iter: int = 1000):
    if(f(left)*f(right) > 0):
        raise ValueError("Same sign at left and right corner")

    iters = 0
    curr = (left + right)/2
    f_curr = f(curr)
    while (right - left)/2.0 > prec and iters < max_iter:
        if f(curr)*f(left) > 0:
            left = curr
        else:
            right = curr
        iters += 1
        curr = (left + right)/2
        f_curr = f(curr)
    if iters >= max_iter and (right - left)/2.0 > prec:
        raise RuntimeError(f'Root search did not converge after {iters} iterations')
    return curr

def get_TDM_det(f, lam: np.float64, n_steps: int):
    domain = np.linspace(0.0, 1.0, n_steps + 1, endpoint=True)
    h = domain[1] - domain[0]
    fk = f(domain)
    main = np.full_like(domain, h**2*lam - 2.0)
    main[0] = 1.0
    main[-1] = 1.0 - 0.5*h**2*lam

    upper = fk[:-1]/2*h + 1.0
    upper[0] = 0.0

    lower = -0.5*fk[1:]*h + 1.0
    lower[-1] = -1.0

    d_cur = 0.0
    dp = main[0]
    dpp = 1.0
    for i in range(1, main.shape[0]):
        d_cur = main[i]*dp - upper[i - 1]*lower[i - 1]*dpp
        dpp = dp
        dp = d_cur
    return d_cur




dom = np.linspace(0, 100.0, 200, endpoint=True)
dets = np.zeros_like(dom)

# for i in range(dom.shape[0]):
#     dets[i] = func(dom[i])
# plt.plot(dom, dets)
# plt.grid(True)
# plt.show()

# Model problem with fixed coefs

const = .0
f = lambda x: np.full_like(x, const)
n_steps = 10000


def func(lam):
    return get_TDM_det(f, lam, n_steps)


eigvals = np.array([(np.pi*(1.0/2.0 + np.float64(k)))**2 for k in range(4)])
locs = np.vstack((eigvals - .1, eigvals + .2)).T


print('+---------------------------------------------------------------------------------------------------------------')
print(f'Model problem solution (f(x) == {const})')
for ind, loc in enumerate(locs):
    root = find_root(func, loc[0], loc[1], prec=1e-9)
    print(f'{ind + 1} eigval. Numeric: {root:.5e}, Analytical: {eigvals[ind]:.5e}, Difference: {np.fabs(root - eigvals[ind]):.5e}')
print('+---------------------------------------------------------------------------------------------------------------\n\n')


f = lambda x: -2.0*x
n_steps_coarse = 2000

def func_coarse(lam):
    return get_TDM_det(f, lam, n_steps_coarse)
def func(lam):
    return get_TDM_det(f, lam, n_steps)


dom = np.linspace(.0, 140.0, 500, endpoint=True)

dets = np.zeros_like(dom)
for i in range(dom.shape[0]):
    dets[i] = func_coarse(dom[i])
fig, ax = plt.subplots(1,1)
ax.plot(dom, dets)
ax.grid()
ax.set_xlabel('$\\lambda$')
ax.set_ylabel('Определитель системы')
fig.suptitle('Примерное расположение корней')
plt.show()
locs = np.array(
            [[3.0, 4.0],
             [23.0, 24.0],
             [62., 64.],
             [122.0, 123.0]])




print('+---------------------------------------------------------------------------------------------------------------')
print(f'Varying coefs problem solution (f(x) == -2x)')
for ind, loc in enumerate(locs):
    root = find_root(func, loc[0], loc[1], prec=1e-9)
    print(f'{ind + 1} eigval. Localization [{loc[0]:.2f}:{loc[1]:.2f}], Numeric value {root:.10f}')
print('+---------------------------------------------------------------------------------------------------------------\n\n')
print(func(3.0))
print(func(4.0))

