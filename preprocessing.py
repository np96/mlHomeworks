from collections import defaultdict
import numpy as np
from numpy import mean, std

def normalize(x, method='rescale'):
    for i in range(1, x.shape[1]):
        x[:, i] = normalize_col(x[:, i], method)


def normalize_col(x, method='rescale'):
    if method == 'rescale':
        return (x - min(x)) / (max(x) - min(x))
    elif method == 'mean':
        return (x - mean(x)) / (max(x) - min(x))
    elif method == 'std':
        return (x - mean(x)) / std(x)

def bag_of_words(words, coeff = 1.0):
    bag = defaultdict(lambda: 0.0)
    for word in words.split(' '):
        bag[word] = bag[word] + coeff
    return bag

def keys(dicts):
    return set([k for d in dicts for k in d.keys()])

def dicts_to_np(dicts):
    set_of_words = set([k for d in dicts for k in d.keys()])
    word_by_num = dict(enumerate(set_of_words))
    num_by_word = dict([(v, k) for (k,v) in word_by_num.items()])
    npres = np.zeros((len(dicts), len(word_by_num)))
    for i, d in enumerate(dicts):
        for k, v in d.items():
            npres[i, num_by_word[k]] = v
    return (npres, num_by_word)