import numpy as np
from scipy import linalg

l1_norm = lambda x, y: abs(x - y)
l2_norm = lambda x, y: np.sqrt((x - y) ** 2)

triangular_kernel = lambda x: (1 - abs(x)) * (abs(x) <= 1)
quadratic_kernel = lambda x: (1 - x ** 2) ** 2 * (abs(x) <= 1)
gaussian_kernel = lambda x: (2 * np.pi) ** (-1.0 / 2) * np.exp(-x ** 2 / 2.0)


def loo(x, y, i, gamma, h, kernel, metric):
    n = 0
    d = 0
    xi = x[i]
    hi = h[i]
    for j, (xj, yj, gj) in enumerate(zip(x, y, gamma)):
        if xi == xj: continue
        k = kernel(metric(xi, xj) / hi)
        n += yj * gj * k
        d += gj * k
    return n / d 

def lowess(x, y, k, kernel, metric):
    n = len(x)
    gamma = [1.0] * n
    h = [np.partition([metric(xi, xj) for xi in x if metric(xi, xj) > 0], k)[k] for xj in x]

    for _ in range(3):
        answers = [loo(x, y, i, gamma, h, kernel, metric) for i in range(n)]
        errs = [abs(answers[i] - y[i]) for i in range(n)]
        m = np.median(errs)
        mean_change = 0
        for i in range(n):
            new = kernel(errs[i] / (6 * m))
            mean_change += abs(new - gamma[i])
            gamma[i] = new 
        if mean_change / n < 1e-6:
            break
    
    ws = []
    for xj in x:
        n = 0
        d = 0
        for xi, yi, gi, hi in zip(x, y, gamma, h):
            k = kernel(metric(xi, xj) / hi)
            n += yi * gi * k
            d += gi * k
        ws.append(n / d)
    return np.array(ws)

def kernel_smoothing(x, y, k, kernel, metric):
    h = [np.partition([metric(xi, xj) for xi in x if metric(xi, xj) > 0], k)[k] for xj in x]
    ws = []
    for xj in x:
        ni, di = 0, 0
        for i, yi in enumerate(y):
            c = kernel(metric(x[i], xj) / h[i])
            ni += c * yi
            di += c
        ws.append(ni / di)
    return ws

class NonParametricRegression:
    def __init__(self,
                 params={
                     'method': 'lowess',
                     'metric': l2_norm,
                     'k': 5,
                     'kernel': quadratic_kernel
                 }):
        self._kernel = params['kernel']
        self._method = params['method']
        self._k = params['k']
        self._metric = params['metric']
        self._x, self._y = None, None
        self._pred = None

    def train(self, x, y):
        self._x = x
        self._y = y
        if self._method == 'ks':
            return kernel_smoothing(x, y, self._k, self._kernel, self._metric)
        elif self._method == 'lowess':
            return lowess(x, y, self._k, self._kernel, self._metric)