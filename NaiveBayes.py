from preprocessing import keys
import numpy as np

def multinomial_train(x, y, alpha):
    ks = keys(x)
    alpha_v = alpha * len(ks)
    xy = list(zip(x, y))
    x_false = [e for (e, spam) in xy if not spam]
    x_true  = [e for (e, spam) in xy if spam]
    probs = {}
    sum_false = -np.log(alpha_v + sum([sum(e.values()) for e in x_false]))
    sum_true = -np.log(alpha_v + sum([sum(e.values()) for e in x_true]))
    probs[0] = dict.fromkeys(ks, alpha)
    probs[1] = dict.fromkeys(ks, alpha)
    for (i, d) in enumerate(x):
        for (k,v) in d.items():
            probs[y[i]][k] += v
    probs[0] = {k: np.log(v) + sum_false for (k, v) in probs[0].items() }
    probs[1] = {k: np.log(v) + sum_true for (k, v) in probs[1].items() }
    return probs

class MultinomialNaiveBayes:

    def __init__(self, alpha, border = 0.0):
        self._X = None
        self._y = None
        self._probs = None
        self._alpha = alpha
        self._border = border

    def fit(self, X, y):
        self._X = X
        self._y = y
        self._probs = multinomial_train(self._X, self._y, self._alpha)   

    def predict(self, x):
        res = [0, 0]
        for i in [0,1]:
            res[i] = sum((self._probs[i].get(k, self._alpha) * v for (k,v) in x.items()))
        return res[1] + self._border > res[0]


    def score(self, X, y):
        res = [0, 0]
        for (i, x) in enumerate(X):
            res[y[i]] += self.predict(x[i]) != y[i]
        return res
