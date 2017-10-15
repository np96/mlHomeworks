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
