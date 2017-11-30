import numpy as np
from scipy import linalg

l1_norm = lambda x, y: abs(x - y)
l2_norm = lambda x, y: np.sqrt((x - y) ** 2)

triangular_kernel = lambda x: (1 - abs(x)) * (abs(x) <= 1)
quadratic_kernel = lambda x: (1 - x ** 2) ** 2 * (abs(x) <= 1)
gaussian_kernel = lambda x: (2 * np.pi) ** (-1.0 / 2) * np.exp(-x ** 2 / 2.0)


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


def lowess(x, y, k, kernel, metric):
    r = int(np.ceil(0.25 * len(x)))
    h = [np.sort(metric(x, x[i]))[r] for i in range(len(x))]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = kernel(w)
    yest = np.zeros(len(x))
    delta = np.ones(len(x))
    for iteration in range(k):
        for i in range(len(x)):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = kernel(delta)

    return yest


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