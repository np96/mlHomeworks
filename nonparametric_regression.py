import numpy as np

l1_norm = lambda x, y: abs(x - y)
l2_norm = lambda x, y: np.sqrt((x - y) ** 2)

triangular_kernel = lambda x: 1 - abs(x) if abs(x) <= 1 else 0
quadratic_kernel = lambda x : (1 - x ** 2) ** 2 if x ** 2 <= 1 else 0
gaussian_kernel = lambda x: np.exp(-2 * (x ** 2))

def kernel_smoothing(x, y, k, kernel, metric):
    h = [np.partition([metric(xi, xj) for xi in x if metric(xi, xj) > 0], k)[k] for xj in x]
    print(h)
    ws = []
    for xj in x:
        ni , di = 0, 0
        for i, yi in enumerate(y):
            c = kernel(metric(x[i], xj) / h[i])
            ni += c * yi
            di += c
        ws.append(ni / di)
    return ws

class NonParametricRegression:
    def __init__(self,
                 params={
                     'method':'ks',
                     'metric':l2_norm,
                     'k':5,
                     'kernel':gaussian_kernel
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
            return kernel_smoothing(x, y, self._k,self._kernel, self._metric)
        elif self._method == 'lowess':
            pass
