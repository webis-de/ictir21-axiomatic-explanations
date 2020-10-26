import itertools
import logging

import numpy as np
import pandas as pd

_use_numba = False
try:
    import numba as nb
    _use_numba = True
except ImportError:
    pass


def _weighted_sum(matrices, weights, target, normalize):
    if target is None:
        target = matrices[0].copy()
    else:
        target.data.values[:] = matrices[0].data.values
    v = target.data.values * weights[0]
    for w, m in zip(weights[1:], matrices[1:]):
        v[:] += w * m.data.values
    target.data.values[:] = v
    if normalize:
        target.normalize_preferences()
    return target

if _use_numba:
    _weighted_sum = nb.jit()(_weighted_sum)


class PreferenceMatrix(object):
    def __init__(self, ranking, axiom_name):
        sr = ranking
        self._data = pd.DataFrame(
            data=np.zeros((len(sr), len(sr)), dtype=np.float64),
            columns=sr,
            index=sr)
        self._axiom = axiom_name

    def __repr__(self, *args, **kwargs):
        return self._data.__repr__(*args, **kwargs)

    @property
    def axiom(self):
        return self._axiom

    @property
    def data(self):
        return self._data

    def copy(self):
        mat = PreferenceMatrix(self._data.index, self._axiom, self._topic, self._model, True)
        mat._data.values[:] = self._data.values
        return mat

    def add(self, other, weight=1.0):
        self._data += weight * other._data

    @staticmethod
    def weighted_sum(matrices, weights, target=None, normalize=False):
        return _weighted_sum(matrices, weights, target, normalize)

    @staticmethod
    def random_matrix(ranking, axiom_name=None, topic=None, model=None, rng=None):
        import random
        if rng is None:
            rng = random.Random()
        m = PreferenceMatrix(ranking, axiom_name, topic, model)
        for (i, j) in itertools.combinations(range(m.data.values.shape[0]), 2):
            flip = rng.randint(0, 1)
            m.data.values[i, j] = flip
            m.data.values[j, i] = 1 - flip
        return m

    def normalize_preferences(self):
        """
        For all indices i, j, replace A[i,j] with i > j
        :return:
        """
        v = self._data.values
        for (i, j) in itertools.combinations(range(v.shape[0]), 2):
            p = v[i, j] > v[j, i]
            v[i, j] = p
            v[j, i] = 1 - p

    @staticmethod
    def from_ranking(ranking, axiom_name):
        mat = PreferenceMatrix(ranking, axiom_name)
        for i, d1 in enumerate(ranking):
            for d2 in ranking[i+1:]:
                mat.data.loc[d1, d2] = 1
        return mat




