import numpy as np
from preprocessing import dicts_to_np, bag_of_words, keys
from collections import defaultdict

def multinomial_train(x, y, alpha):
    ks = keys(x)
    alpha_v = alpha * len(ks)
    xy = list(zip(x, y))
    x_false = [e for (e, i) in xy if i == 0]
    x_true  = [e for (e, i) in xy if i == 1]
    res = [(k, 0) for k in ks]
    probs = {0: dict(res), 1: dict(res)}
    sum_false = -np.log(alpha_v + sum([sum(e.values()) for e in x_false]))
    sum_true = -np.log(alpha_v + sum([sum(e.values()) for e in x_true]))
    probs[0] = dict([(k, sum_false) for (k,v) in probs[0].items()])
    probs[1] = dict([(k, sum_true)  for (k,v) in probs[1].items()])
    for k in ks:
        s = [0, 0]
        s[0] += alpha
        s[1] += alpha
        for (i,d) in enumerate(x):
            s[y[i]] += d.get(k, 0)
        probs[0][k] += np.log(s[0])
        probs[1][k] += np.log(s[1])
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


      