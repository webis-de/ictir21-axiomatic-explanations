import abc
import atexit
import itertools
import os
import sys
import traceback
import typing
from trectools import TrecQrel

import numpy as np

from axiomatic.explanations import pairsampling
from axiomatic.axioms import PairwiseAxiom
from axiomatic.collection import anserini
from tqdm import tqdm

try:
    from pyspark.accumulators import AddingAccumulatorParam
    class IntAggAccumParam(AddingAccumulatorParam):
        def __init__(self, fn):
            super().__init__(0)
            self._fn = fn

        def addInPlace(self, value1, value2):
            return self._fn(value1, value2)
except ImportError:
    pass  # there is no spark


def __rank_in_qrel(qrels_for_topic, doc):
    return 1 + len([i for i in qrels_for_topic if i['rel'] > doc['rel']])


def __generate_single_qrel_pair(qrels_for_topic, first, second):
    rank_of_first = __rank_in_qrel(qrels_for_topic, first)
    rank_of_second = __rank_in_qrel(qrels_for_topic, second)

    return {"query": str(first['query']), "system": "original-qrels", "id1": first['docid'], "id2": second['docid'],
            "upper_rank": rank_of_second, "rankdiff": rank_of_second - rank_of_first,
            "upper_score": float(second['rel']), "scorediff": float(first['rel'] - second['rel'])}


def generate_pairs_from_qrels(qrel_file, topk_rank=20, num_random=5000):
    qrels = TrecQrel(qrel_file)
    for topic in tqdm(qrels.topics()):
        qrels_for_topic = [d.to_dict() for _, d in qrels.qrels_data[qrels.qrels_data['query'] == topic].iterrows()]
        for i in range(0, len(qrels_for_topic)):
            for j in range(0, len(qrels_for_topic)):
                if qrels_for_topic[i]['rel'] > qrels_for_topic[j]['rel']:
                    yield __generate_single_qrel_pair(qrels_for_topic, qrels_for_topic[i], qrels_for_topic[j])


class Accumulators(abc.ABC):

    @abc.abstractmethod
    def track_ranking(self, r):
        pass

    @abc.abstractmethod
    def count_pair(self):
        pass

    @abc.abstractmethod
    def count_run(self, ax):
        pass

    @abc.abstractmethod
    def count_prec_sat(self, ax):
        pass

    @abc.abstractmethod
    def count_error(self, ax):
        pass

    @abc.abstractmethod
    def count_time(self, ax, secs):
        pass

    @abc.abstractmethod
    def count_pref(self, ax):
        pass

    @abc.abstractmethod
    def print(self):
        pass


class SparkAccumulators(Accumulators):

    def __init__(self):
        self._run = {}
        self._prec_sat = {}
        self._pref = {}
        self._error = {}
        self._time = {}


    def _setup(self, sc, axioms):
        self._run = {a.name:sc.accumulator(0) for a in axioms}
        self._prec_sat = {a.name:sc.accumulator(0) for a in axioms}
        self._pref = {a.name:sc.accumulator(0) for a in axioms}
        self._error = {a.name:sc.accumulator(0) for a in axioms}
        self._time = {a.name:sc.accumulator(0.0) for a in axioms}

        self._max_ranking_len = sc.accumulator(0, IntAggAccumParam(lambda a, b: max(a, b)))
        self._min_ranking_len = sc.accumulator(0, IntAggAccumParam(lambda a, b: min(a, b)))
        self._sum_ranking_len = sc.accumulator(0)
        self._num_ranking_lens = sc.accumulator(0)
        self._num_pairs = sc.accumulator(0)

    def track_ranking(self, r):
        l = len(r)
        self._max_ranking_len.add(l)
        self._min_ranking_len.add(l)
        self._sum_ranking_len.add(l)
        self._num_ranking_lens.add(1)


    def _count(self, d, k, v=1):
        if k in d:
            d[k].add(v)

    def count_pair(self):
        self._num_pairs.add(1)

    def count_run(self, ax):
        self._count(self._run, ax.name)

    def count_prec_sat(self, ax):
        self._count(self._prec_sat, ax.name)

    def count_error(self, ax):
        self._count(self._error, ax.name)

    def count_time(self, ax, secs):
        self._count(self._time, ax.name, secs)

    def count_pref(self, ax):
        self._count(self._pref, ax.name)

    def print(self):
        import pandas as pd
        rows = sorted(list(set(self._run.keys()).union(self._prec_sat.keys()).union(self._error.keys())))
        cols = ['Total', 'Sat', 'Pref', 'Error', 'Time']
        dicts = [self._run, self._prec_sat, self._pref, self._error, self._time]
        df = pd.DataFrame(
            [[d[a].value if a in d else 0 for d in dicts] for a in rows],
            index=rows, columns=cols)

        df.Time /= (df.Sat - df.Error)
        df['PSat'] = df.Sat / df.Total
        df['PPref'] = df.Pref / df.Sat

        print(df.to_string(float_format=lambda f: '%.2e' % f))

        avgrlen = self._sum_ranking_len.value / self._num_ranking_lens.value if self._num_ranking_lens.value > 0 else 0
        print(
            f'{self._num_ranking_lens.value} rankings:',
            f'{self._min_ranking_len.value}/{avgrlen:#.1f}/{self._max_ranking_len.value}',
            '(min/mean/max)')


class MapredAccumulators(Accumulators):

    def __init__(self):
        from collections import defaultdict
        self.group = 'AxiomaticExplanations'
        self._flush_every = 1000
        self._n_since_flush = 0
        self._time_buf = defaultdict(float)

    def _set_status(self, message):
        sys.stderr.write(f"reporter:status:{message}\n")
        sys.stderr.flush()

    def _inc(self, counter: str, by=1, group=None):
        import sys
        if group is None:
            group = self.group
        sys.stderr.write(f"reporter:counter:{group},{counter},{by}\n")
        self._n_since_flush += 1
        if self._n_since_flush >= self._flush_every:
            sys.stderr.flush()
            self._n_since_flush = 0

    def track_ranking(self, r: typing.Sequence):
        l = len(r)
        self._inc('rankings.count')
        self._inc('ranking.sum_len', l)

    def count_pair(self):
        self._inc('concordant_pairs_sampled', 1)

    def count_run(self, ax):
        self._inc('ax-run', group='Axiom Runs')

    def count_prec_sat(self, ax):
        self._inc('ax-prec-sat', group='Axiom Precond')

    def count_error(self, ax):
        self._inc('ax-error', group='Axiom Error')

    def count_time(self, ax, secs):
        self._time_buf[ax.name] += secs
        secs = int(self._time_buf[ax.name])
        if secs > 0:
            self._time_buf[ax.name] -= secs
            self._inc('ax-secs', by=secs, group='Axiom Seconds')

    def count_pref(self, ax):
        self._inc('ax-prefs', group='Axiom Pref Nonzero')

    def print(self):
        pass


class MultiCollection(object):
    def __init__(self, coll, icoll):
        self.coll = coll
        self.icoll = icoll

    def __getattr__(self, item):
        for c in [self.coll, self.icoll]:
            if hasattr(c, item):
                return getattr(c, item)

class AxExpEnvironment(object):
    initialized = False

    def __init__(self, anserini_path: str, index_path: str, topics_path: str, use_sparkfiles: bool):
        l = locals()
        del l['self']
        self._conf = dict(l)

    def _resolvepath(self, p):
        if self._conf['use_sparkfiles']:
            from pyspark import SparkFiles
            return SparkFiles.get(p)
        else:
            return p

    def setup(self):
        idx = None
        import nltk
        try:
            anserini.SETUP(self._resolvepath(self._conf['anserini_path']))
        except ValueError as e:
            sys.stderr.write(f'Assuming VM already initialized ({e})')
        idx = self._resolvepath(self._conf['index_path'])
        import tempfile
        tmp = tempfile.mkdtemp()
        atexit.register(lambda: os.system(f'rm -r {tmp}'))
        nltk.data.path = [tmp]
        nltk.download('wordnet', download_dir=tmp)

        from axiomatic.axioms import RerankingContext
        from axiomatic.features import Features
        from axiomatic.collection.trec import TrecRunDfIndexedCollection, TrecRobustQueries
        from axiomatic.collection.anserini import AnseriniLuceneCollection

        icoll = TrecRunDfIndexedCollection(None) # to be configured later
        coll = AnseriniLuceneCollection(idx)
        coll = MultiCollection(coll, icoll)

        self.q_id = TrecRobustQueries(self._resolvepath(self._conf['topics_path']), collection_for_processing=coll)
        self.ctx = RerankingContext(coll, Features(coll))
        self.initialized = True

    def __getstate__(self):
        return dict(initialized=False, _conf=self._conf)


class AxExpPrefProcessor(object):

    def __init__(self,
                 axioms: typing.Sequence[PairwiseAxiom],
                 accumulators: Accumulators,
                 env: AxExpEnvironment):
        self.axioms = axioms
        self.A = accumulators
        self.env = env

    def generate_pairs(self, ranking, topk_rank=20, num_random=5000):
        self.A.track_ranking(ranking)

        sysid = ranking['system'].values[0]
        qid = ranking['query'].values[0]

        indexes = np.arange(len(ranking))

        pair_criterion = pairsampling.mk_topk_random_sampler(
            topk=topk_rank, rnk_len=len(ranking), target_n_rand=num_random, seed=1
        )

        pairs = [(i1, i2) for i1, i2 in itertools.combinations(indexes, 2)
                 if i1 < i2 and pair_criterion(i1, i2)
                ]

        for idx, (i, j) in enumerate(pairs):
            out = dict(query=str(qid), system=str(sysid))

            ri = ranking.iloc[i]
            rj = ranking.iloc[j]

            out['id1'] = str(ri.docid)
            out['id2'] = str(rj.docid)
            out['upper_rank'] = int(rj['rank'])
            out['rankdiff'] = int(rj['rank'] - ri['rank'])
            out['upper_score'] = float(rj['score'])
            out['scorediff'] = float(ri['score'] - rj['score'])
            self.A.count_pair()
            yield out

    def generate_preferences(self, pair_dict):
        if self.env is None or not self.env.initialized:
            self.env.setup()
        d = dict(pair_dict)

        del d['id1']
        del d['id2']
        dd = dict(d)

        d['concordant'] = 1
        dd['concordant'] = -1

        self.env.ctx.c.get_retrieval_score = lambda q, doc: {
            pair_dict['id1']: pair_dict['upper_score'],
            pair_dict['id2']: pair_dict['upper_score'] - pair_dict['scorediff']
        }[doc.doc_id]

        query = self.env.q_id[d['query']]
        d1, d2 = self.env.ctx.c.get_document(pair_dict['id1']), self.env.ctx.c.get_document(pair_dict['id2'])
        for a in self.axioms:
            p = 0
            self.A.count_run(a)
            precondition = a.precondition(self.env.ctx, query, d1, d2)
            if precondition:
                try:
                    import time
                    self.A.count_prec_sat(a)
                    start = time.time()
                    p = a.preference(self.env.ctx, query, d1, d2)
                    self.A.count_time(a, time.time() - start)
                    if p != 0:
                        self.A.count_pref(a)
                except:
                    import sys
                    sys.stderr.write(f'{"*" * 20} ERROR {"*" * 20}\n')
                    sys.stderr.write(f'{query} : {a.name} : {d1.doc_id}, {d2.doc_id}\n')
                    traceback.print_exc()
                    self.A.count_error(a)
            d[f'ax_{a.name}'] = p
            dd[f'ax_{a.name}'] = -p
            d[f'ax_{a.name}_precondition'] = 1 if precondition else 0
            dd[f'ax_{a.name}_precondition'] = d[f'ax_{a.name}_precondition']

        yield d
        yield dd
