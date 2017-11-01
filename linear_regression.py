import numpy as np
import random
from numpy.linalg import inv, norm
from preprocessing import normalize


def gradient(x, y, w, t=0.0):
    grad = np.zeros(x[0].T.shape)
    for xi, yi in zip(x, y):
        grad += (np.dot(w, xi) - yi) * xi
    return grad + w * 2.0 * t


def gradient_descent(x, y, t):
    n = 10000
    rate = 0.5
    n, l = x.shape
    w = np.ones((1, l))
    for i in range(n):
        w_was = np.copy(w)
        w -= rate * gradient(x, y, w, t) / x.shape[0]
        if np.linalg.norm(w_was - w, 2) < 1e-8:
            break
    return w


def conjugate_gradients(x, y, n=5000, eps=1e-8):
    l = x.shape[1]
    H = [[sum(map(lambda a: a[i] * a[j], x)) for j in range(l)] for i in range(l)]
    H = np.array(H)
    w = (np.random.rand(l) - 0.5) * 1000
    omega, grad, s = 0, 0, 0
    for i in range(n):
        if i:
            omega = np.inner(grad, grad)
        grad = gradient(x, y, w)
        if i:
            omega = np.inner(grad, grad) / omega
        s = -grad + omega * s
        lam = np.inner(grad, s) / np.inner(np.dot(s, H), s)
        w_next = w - lam * s
        if norm(w_next - w, 2) < eps:
            return w_next
        w = w_next
    return w


def exact_solution(x, y):
    w = inv(np.dot(x.T, x))
    w = w.dot(x.T)
    w = w.dot(y)
    return w.T

def fit(ws, x):
    return np.inner(ws, x)


def L(ws, x, y):
    return (y - fit(ws, x)) ** 2

def Q(ws, xs, ys):
    q = 0
    for x, y in zip(xs, ys):
        q += L(ws, x, y)
    return (q / len(ys)) ** 0.5

def de(xs, ys, gen_size = 20, F = 1.2, p = 0.7, n = 600):
    gen = (np.random.rand(gen_size, xs.shape[1]) - 0.5) * 10000
    for _ in range(n):
        new_gen = gen.copy()
        for i in range(gen_size):
            vs = np.random.randint(0, gen_size, size=3)
            while len(np.unique(vs)) != 3 or i in vs:
                vs = np.random.randint(0, gen_size, size=3)
            
            v = gen[vs[0]] + F * (gen[vs[1]] - gen[vs[2]])
            for j in range(xs.shape[1]):
                if np.random.rand(1) > p:
                    v[j] = gen[i][j]
            if Q(v, xs, ys) < Q(gen[i], xs, ys):
                new_gen[i] = v
        gen = new_gen
        
    qs = np.array(list(map(lambda v: Q(v, xs, ys), gen)))
    return gen[qs.argmin()]


def evolution(train_x, train_y, generations=200, much=100):
    coeffs = (np.random.uniform(0, 1, (much, 3)) - 0.5) * 10000

    for i in range(generations):
        next_gen = []
        for a, b, c in coeffs:
            diff = 0
            for i, x in enumerate(train_x):
                one, shit1, shit2 = x
                predY = shit1 * a + shit2 * b + c
                diff += (predY - train_y[i]) ** 2
            next_gen += [[diff, [a, b, c]]]
        next_gen.sort()
        coeffs = list(map(lambda x: x[1], next_gen))
        best = coeffs[:len(coeffs) // 2]
        for i in range(len(coeffs) // 2, len(coeffs)):
            best1 = random.choice(best)
            best2 = random.choice(best)
            coeffs[i] = [(best1[i] + best2[i]) / 2 for i in range(3)]
    return np.array(coeffs[0])


class LinearRegression:
    def __init__(self, normalize=False, reg_score=0.0, method='gd'):
        self.normalize = normalize
        self.reg_score = reg_score
        self.method = method
        self.X = None
        self.Y = None
        self.weights = None

    def fit(self, x, y):
        shape = (x.shape[0], 1)
        self.X = x
        self.X = np.c_[np.ones(shape), x]
        self.Y = np.copy(y)
        self.weights = self.__solve__()
        return self.weights

    def __solve__(self):
        if self.normalize:
            normalize(self.X)
        if self.method == 'gd':
            return gradient_descent(self.X, self.Y, self.reg_score)
        if self.method == 'exact':
            return exact_solution(self.X, self.Y)
        if self.method == 'evolution':
            return de(self.X, self.Y)
        if self.method == 'conj':
            return conjugate_gradients(self.X, self.Y)
