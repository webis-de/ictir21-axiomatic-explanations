import itertools
import typing
from abc import ABCMeta, abstractmethod

import numbers
import math

from axiomatic.collection import _CollectionItem, Document, Collection
from axiomatic.features import prefs


class Features(object):
    """Collects common feature computations for all axioms."""
    coll: Collection

    def __init__(self, collection: Collection):
        self.coll = collection

    def idf(self, term: str) -> numbers.Number:
        df = self.coll.get_term_df(term)
        if df == 0:
            return 0
        return math.log(self.coll.get_document_count() / df)

    def td(self, term):
        return math.floor(100 * self.idf(term))

    def vocab_overlap(self, w1: typing.Set[str], w2: typing.Set[str]):
        """ Jaccard coefficient """
        ilen = len(w1.intersection(w2))
        if ilen == 0:
            return 0
        return ilen / (len(w1) + len(w2) - ilen)

    def synset_similarity(self, t1, t2, smoothing=0) -> numbers.Number:
        from nltk.corpus import wordnet as wn

        cutoff = smoothing+1
        try:
            t1syn = wn.synsets(t1)[:cutoff]
            t2syn = wn.synsets(t2)[:cutoff]
        except AttributeError:
            # wordnet bug accesses None object for some input terms
            return 0

        n = 0
        sim = 0

        for s1, s2 in itertools.product(t1syn, t2syn):
            s = wn.wup_similarity(s1, s2)
            if s is not None:
                sim += s
                n += 1

        return sim/n if n else 0

    def embedding_similarities(
            self, t1, terms2,
            emb_path="http://magnitude.plasticity.ai/fasttext/medium/wiki-news-300d-1M-subword.magnitude",
            emb_stream=True):
        from pymagnitude import Magnitude

        if not hasattr(self, '_wv') or self._wv is None:
            self._wv = {}
        if emb_path not in self._wv:
            self._wv[emb_path] = Magnitude(emb_path, stream=emb_stream)

        return self._wv[emb_path].similarity(t1, terms2)


    def unique_pairs(self, qterms: typing.Iterable[str]):
        comb = list(itertools.combinations(set(qterms), 2))
        return set(comb)


    def average_between_qterms(self, qterms, dterms):
        number_words=0
        qterms_pairs = self.unique_pairs(qterms)

        if len(qterms_pairs) == 0:
            # single-term queries
            return 0

        for item in qterms_pairs:
            element1_pos = dterms.index(item[0])
            element2_pos = dterms.index(item[1])
            number_words += abs(element1_pos - element2_pos -1)
        #return not rounded value
        return number_words/len(qterms_pairs)


    def takeClosest(self, l, n):
        """ Assumes l is sorted. Returns closest value to n. If two numbers are equally close, return the smallest number.
        https://stackoverflow.com/questions/12141150
        """
        from bisect import bisect_left
        pos = bisect_left(l, n)
        if pos == 0:
            return l[0]
        if pos == len(l):
            return l[-1]
        before = l[pos - 1]
        after = l[pos]
        if after - n < n - before:
            return after
        else:
            return before

    def query_term_groups(
            self,
            qterms: typing.Set[str], dterms: typing.Sequence[str]):
        from collections import defaultdict
        indexes = defaultdict(list)
        for i, t in enumerate(dterms):
            if t in qterms:
                indexes[t].append(i)
        for t in qterms:
            other = qterms.difference([t])
            for i in indexes[t]:
                group = [i] + [self.takeClosest(indexes[o], i) for o in other if len(indexes[o]) > 0]
                yield group

    def closest_grouping_size_and_count(self, qterms: typing.Set[str], dterms: typing.Sequence[str]):
        from collections import Counter
        groups = self.query_term_groups(qterms, dterms)

        groups = [ # number of non-query terms within groups
            len([t for t in dterms[min(group)+1:max(group)]
                 if t not in qterms]) for group in groups]

        occurrences = Counter(groups)
        min_g = min(occurrences.keys())
        return min_g, occurrences[min_g]


    def average_smallest_span(self, qterms: typing.Set[str], dterms: typing.Sequence[str]):
        from statistics import mean
        return mean(max(g) - min(g) for g in self.query_term_groups(qterms, dterms))


    def wordcount(self, d: Document):
        return sum(d.tf.values())

    def approx_same_len(self, d1: Document, d2: Document, marginFrac: float=0.1) -> bool:
        return prefs.approximatelyEqual(self.wordcount(d1), self.wordcount(d2), marginFrac=marginFrac)


# ------------------------------------------------------------------------

# class InMemoryCachedFeatures(object):
#
#     def __init__(self, wrapped):
#         import inspect
#         self.wrapped = wrapped
#         self.cache = {}
#
#         def new_delegator(name):
#             def inner(*args, **kwargs):
#                 k = (name, args, tuple(sorted(kwargs.items())))
#
#                 if k in self.cache:
#                     return self.cache[k]
#
#                 fn = getattr(wrapped, name)
#                 result = fn(*args, **kwargs)
#                 self.cache[k] = result
#                 return result
#             return inner
#
#         for name, _ in inspect.getmembers(Features, predicate=inspect.isfunction):
#             setattr(self, name, new_delegator(name))

