import numpy as np

p = lambda x: np.maximum(x, 0)
n = lambda x: np.maximum(-x, 0)


def is_metzler(matrix):
    return (matrix - np.diagonal(matrix) < 0).any()


def interval_minus(a):
    return np.flip(-a, 0)


def intervals_product(a, b):
    return np.array(
        [np.dot(p(a[0]), p(b[0])) - np.dot(p(a[1]), n(b[0])) - np.dot(n(a[0]), p(b[1])) + np.dot(n(a[1]), n(b[1])),
         np.dot(p(a[1]), p(b[1])) - np.dot(p(a[0]), n(b[1])) - np.dot(n(a[1]), p(b[0])) + np.dot(n(a[0]), n(b[0]))])


def remap(v, x, y):
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def mesh_box(box, n):
    if np.size(n) < np.shape(box)[1]:
        n = int(np.power(n, 1/np.shape(box)[1]))
        n = np.tile(n, np.shape(box)[1])
    return np.meshgrid(*[np.linspace(box[0, i], box[1, i], n[i]) for i in range(np.shape(box)[1])])
