from abc import ABC, abstractmethod
import itertools
import typing

import numpy as np

from axiomatic.axioms import PairwiseAxiom, RerankingContext
from axiomatic.collection import Query, Document
from axiomatic.features import prefs

import math

# -- abstract mixins -----------------------------------------

class _WithoutPrecond(PairwiseAxiom, ABC):
    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        return True

class _WithPrecondAllQueryTerms(PairwiseAxiom, ABC):
    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        qterms = set(query.tf.keys())
        return len(qterms) > 1 and len(qterms.intersection(d1.tf.keys())) == len(qterms) and len(qterms.intersection(d2.tf.keys())) == len(qterms)

class _WithPrecondSameQueryTermSubset(PairwiseAxiom, ABC):

    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        """Both documents contain the same set of query terms"""
        if len(query.tf.keys()) < 2:
            # no single-term queries
            return False

        qterms = set(query.termseq)
        d1terms = set(d1.termseq)
        d2terms = set(d2.termseq)
        in_d1 = qterms.intersection(d1terms)
        in_d2 = qterms.intersection(d2terms)

        # both contain the same subset of at least two terms
        return (in_d1 == in_d2) and len(in_d1) > 1

# -- Term frequency ------------------------------------------------------

class TFC1(PairwiseAxiom):
    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        return ctx.f.approx_same_len(d1, d2)

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        tf1 = 0
        tf2 = 0
        for qt in query.termseq:
            tf1 += d1.tf[qt]
            tf2 += d2.tf[qt]
        if not prefs.approximatelyEqual(tf1, tf2):
            # at least 10% difference
            return prefs.strictlygreater(tf1, tf2)
        return 0


class TFC3(PairwiseAxiom):
    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        return ctx.f.approx_same_len(d1, d2)

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        sd1 = 0
        sd2 = 0
        for qt1, qt2 in itertools.combinations(query.tf.keys(), 2):
            if prefs.approximatelyEqual(ctx.f.td(qt1), ctx.f.td(qt2)):
                d1q1 = d1.tf[qt1]
                d2q1 = d2.tf[qt1]
                d1q2 = d1.tf[qt2]
                d2q2 = d2.tf[qt2]

                sd1 += (d2q1 == d1q1 + d1q2) and (d2q2 == 0) and (d1q1 != 0) and (d1q2 != 0)
                sd2 += (d1q1 == d2q1 + d2q2) and (d1q2 == 0) and (d2q1 != 0) and (d2q2 != 0)

        return prefs.strictlygreater(sd1, sd2)


class M_TDC(PairwiseAxiom):
    # modified TDC
    # Shi, S., Wen, J.R., Yu, Q., Song, R., Ma, W.Y.: Gravitation-based model for information retrieval. In: SIGIR ’05.

    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):

        if not ctx.f.approx_same_len(d1, d2):
            return False

        stf1 = 0
        stf2 = 0
        oneCountDiff = False
        for t in query.tf.keys():
            c1 = d1.tf[t]
            c2 = d2.tf[t]
            if c1 != c2:
                oneCountDiff = True
            stf1 += c1
            stf2 += c2

        return (stf1 == stf2) and (oneCountDiff)


    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        score = 0

        for qt1, qt2 in itertools.combinations(query.tf.keys(), 2):

            # qt1 is rarer
            if not ctx.f.idf(qt1) >= ctx.f.idf(qt2):
                qt1, qt2 = qt2, qt1

            # term pair is valid
            if not (
                    (d1.tf[qt1] == d2.tf[qt2] and d1.tf[qt2] == d2.tf[qt1])
                    or (query.tf[qt1] > query.tf[qt2])):
                continue

            # document with more occurrences of qt1 gets a point
            diff = d1.tf[qt1] - d2.tf[qt1]
            score += diff > 0 and 1 or diff < 0 and -1 or 0

        return prefs.strictlygreater(score, 0)

class LEN_M_TDC(M_TDC):
    # Modified M_TDC: the precondition for the documents' lengths can be varied. Default marginFrac: 0.1

    def __init__(self, marginFrac: float):
        self._marginFrac = marginFrac

    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):

        if not ctx.f.approx_same_len(d1, d2, self._marginFrac):
            return False

        stf1 = 0
        stf2 = 0
        oneCountDiff = False
        for t in query.tf.keys():
            c1 = d1.tf[t]
            c2 = d2.tf[t]
            if c1 != c2:
                oneCountDiff = True
            stf1 += c1
            stf2 += c2

        return (stf1 == stf2) and (oneCountDiff)

    @property
    def name(self):
        return 'LEN_M_TDC_' + str(self._marginFrac)

# -- Lenth Norm ----------------------------------------------------------

class LNC1(PairwiseAxiom):

    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        return all([prefs.approximatelyEqual(d1.tf[q], d2.tf[q]) for q in query.tf])

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        # prefer the shorter document
        return prefs.strictlygreater(len(d2), len(d1))


class LNC2(PairwiseAxiom):
    def preference(self, query, d1, d2):
        # LNC2 makes no sense as implemented and was useless in previous trials
        return 0


class TF_LNC(PairwiseAxiom):

    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        # TF-LNC should be always admissible
        return True

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):

        sd1 = 0
        sd2 = 0

        def check(t, dx, dy):
            return dx.tf[t] > dy.tf[t] and len(dx) == (len(dy) + dx.tf[t] - dy.tf[t])

        for t in query.tf.keys():
            if check(t, d1, d2):
                sd1 += 1
            elif check(t, d2, d1):
                sd2 += 1

        return prefs.strictlygreater(sd1, sd2)

# -- Lower bound ---------------------------------------------------------

class LB1(PairwiseAxiom):

    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        return prefs.approximatelyEqual(
            ctx.c.get_retrieval_score(query, d1),
            ctx.c.get_retrieval_score(query, d2)
        )

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        qterms = set(query.termseq)
        tf1 = d1.tf
        tf2 = d2.tf

        for t in qterms:
            if tf1[t] == 0 and tf2[t] > 0:
                return -1
            if tf2[t] == 0 and tf1[t] > 0:
                return 1

        return 0

# -- Query aspects -------------------------------------------------------

class REG(_WithoutPrecond):
    # Ref:  Zheng, W., Fang, H.: Query aspect based term weighting regularization in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.
    
    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        qterms = list(query.tf.keys())
        ssum = np.zeros(len(qterms))

        for i1, i2 in itertools.combinations(np.arange(len(qterms)), 2):
            sim = ctx.f.synset_similarity(qterms[i1], qterms[i2])
            ssum[i1] += sim
            ssum[i2] += sim

        tmax = qterms[np.argmin(ssum)]
        return prefs.strictlygreater(d1.tf[tmax], d2.tf[tmax])

class ANTI_REG(_WithoutPrecond):
    # Ref:  Modified (uses argmax instead of argmin) Zheng, W., Fang, H.: Query aspect based term weighting regularization in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.
    
    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        qterms = list(query.tf.keys())
        ssum = np.zeros(len(qterms))

        for i1, i2 in itertools.combinations(np.arange(len(qterms)), 2):
            sim = ctx.f.synset_similarity(qterms[i1], qterms[i2])
            ssum[i1] += sim
            ssum[i2] += sim

        tmax = qterms[np.argmax(ssum)]
        return prefs.strictlygreater(d1.tf[tmax], d2.tf[tmax])

class AND(_WithoutPrecond):

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        ts = set(query.tf.keys())
        s1 = ts.intersection(d1.tf.keys()) == ts
        s2 = ts.intersection(d2.tf.keys()) == ts
        return prefs.strictlygreater(s1, s2)

class LEN_AND(AND):
    # Modified AND: the precondition for the documents' lengths can be varied. Default marginFrac: 0.1
    def __init__(self, marginFrac: float):
        self._marginFrac = marginFrac

    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        return ctx.f.approx_same_len(d1, d2, self._marginFrac)

    @property
    def name(self):
        return 'LEN_AND_' + str(self._marginFrac)

class M_AND(_WithoutPrecond):
    # modified AND: one document contains a larger subset of query terms

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        ts = set(query.tf.keys())
        s1 = ts.intersection(d1.tf.keys())
        s2 = ts.intersection(d2.tf.keys())
        return prefs.strictlygreater(len(s1), len(s2))

class LEN_M_AND(M_AND):
    # Modified M_AND: the precondition for the documents' lengths can be varied. Default marginFrac: 0.1
    def __init__(self, marginFrac: float):
        self._marginFrac = marginFrac

    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        return ctx.f.approx_same_len(d1, d2, self._marginFrac)

    @property
    def name(self):
        return 'LEN_M_AND_' + str(self._marginFrac)

class DIV(_WithoutPrecond):

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        qts = set(query.tf.keys())
        o1 = ctx.f.vocab_overlap(qts, d1.tf.keys())
        o2 = ctx.f.vocab_overlap(qts, d2.tf.keys())

        return prefs.strictlygreater(o2, o1)

class LEN_DIV(DIV):
    # Modified DIV: the precondition for the documents' lengths can be varied. Default marginFrac: 0.1

    def __init__(self, marginFrac: float):
        self._marginFrac = marginFrac

    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        return ctx.f.approx_same_len(d1, d2, self._marginFrac)

    @property
    def name(self):
        return 'LEN_DIV_' + str(self._marginFrac)

# -- Semantic similarity -------------------------------------------------

class _WordnetSimilarityMixin(object):
    def _similarity(self, ctx: RerankingContext, terms1, terms2):
        result = np.zeros((len(terms1), len(terms2)), dtype=np.float32)
        for i, t in enumerate(terms1):
            for j, u in enumerate(terms2):
                result[i, j] = ctx.f.synset_similarity(t, u)
        return result


class _WordEmbeddingSimilarityMixin(object):
    emb_path = None
    emb_stream = True
    def _similarity(self, ctx: RerankingContext, terms1, terms2):
        result = np.zeros((len(terms1), len(terms2)), dtype=np.float32)
        for i, t in enumerate(terms1):
            result[i,:] = ctx.f.embedding_similarities(t, terms2, self.emb_path, self.emb_stream)
        return result

class _FastTextWikinewsSimilarityMixin(_WordEmbeddingSimilarityMixin):
    emb_path = "http://magnitude.plasticity.ai/fasttext/medium/wiki-news-300d-1M-subword.magnitude"

class _FasttextRobust04SimilarityMixin(_WordEmbeddingSimilarityMixin):
    emb_path = '/mnt/nfs/webis20/data-in-progress/web-search/axiomatic-explanations/robust04.skipgram.100.magnitude'
    emb_stream = False

class STMC1(_WithoutPrecond, _WordnetSimilarityMixin):

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        sim1 = 0
        sim2 = 0

        d1_terms = set(d1.tf)
        d2_terms = set(d2.tf)

        d_terms_both = d1_terms.intersection(d2_terms)

        def sum_sim(ts):
            return np.sum(self._similarity(ctx, ts, query.tf))

        sim1 += sum_sim(d_terms_both)
        sim2 += sum_sim(d_terms_both)
        sim1 += sum_sim(d1_terms.difference(d2_terms))
        sim2 += sum_sim(d2_terms.difference(d1_terms))

        sim1 /= ctx.f.wordcount(d1)
        sim2 /= ctx.f.wordcount(d2)

        return prefs.strictlygreater(sim1, sim2)


class STMC1_f(STMC1, _FastTextWikinewsSimilarityMixin):
    pass


class STMC1_fr(STMC1, _FasttextRobust04SimilarityMixin):
    pass


class STMC2(_WithoutPrecond, _WordnetSimilarityMixin):

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        non_q_terms = list(set(d1.tf.keys()).union(d2.tf.keys()).difference(query.tf.keys()))
        q_terms = list(query.tf.keys())

        sim = self._similarity(ctx, non_q_terms, q_terms)
        max_n, max_q = np.unravel_index(np.argmax(sim), sim.shape)
        max_q, max_n = (q_terms[max_q], non_q_terms[max_n])

        len_ratio = lambda a, b: len(a) / len(b)
        tf_ratio = lambda a, b: (a.tf[max_n] / b.tf[max_q]) if b.tf[max_q] > 0 else float('inf')

        if prefs.approximatelyEqual(
                len_ratio(d2, d1),
                tf_ratio(d2, d1), marginFrac=0.2):
            return 1
        elif prefs.approximatelyEqual(
                len_ratio(d1, d2),
                tf_ratio(d1, d2), marginFrac=0.2):
            return -1

        return 0


class STMC2_f(STMC2, _FastTextWikinewsSimilarityMixin):
    pass


class STMC2_fr(STMC2, _FasttextRobust04SimilarityMixin):
    pass

# -- Proximity axioms ----------------------------------------------------

class PROX1(_WithPrecondSameQueryTermSubset):

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        terms_to_test = set(query.termseq).intersection(d1.termseq).intersection(d2.termseq)

        avg1 = ctx.f.average_between_qterms(terms_to_test, d1.termseq)
        avg2 = ctx.f.average_between_qterms(terms_to_test, d2.termseq)

        return prefs.strictlygreater(avg2, avg1)


class PROX2(_WithPrecondSameQueryTermSubset):

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        fpsum1 = 0
        fpsum2 = 0

        ts = set(d1.termseq).intersection(d2.termseq)
        for t in set(query.termseq):
            if t in ts:
                fpsum1 += d1.termseq.index(t)
                fpsum2 += d2.termseq.index(t)

        return prefs.strictlygreater(fpsum2, fpsum1)


class PROX3(_WithPrecondSameQueryTermSubset):

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        terms = query.termseq
        tl = len(terms)

        def find_idx(seq: typing.Sequence[str]):
            for i in (i for i, e in enumerate(seq) if e == terms[0]):
                if i+tl <= len(seq) and seq[i:i+tl] == terms:
                    return i
            return float('inf')

        return prefs.strictlygreater(find_idx(d2.termseq), find_idx(d1.termseq))

class PROX4(_WithPrecondAllQueryTerms):

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        qterms = set(query.tf.keys())
        sd1 = ctx.f.closest_grouping_size_and_count(qterms, d1.termseq)
        sd2 = ctx.f.closest_grouping_size_and_count(qterms, d2.termseq)
        if sd1[0] == sd2[0]:
            return prefs.strictlygreater(sd1[1], sd2[1])
        else:
            return prefs.strictlygreater(sd2[0], sd1[0])

class PROX5(_WithPrecondAllQueryTerms):
    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        qterms = set(query.tf.keys())
        sd1 = ctx.f.average_smallest_span(qterms, d1.termseq)
        sd2 = ctx.f.average_smallest_span(qterms, d2.termseq)
        return prefs.strictlygreater(sd2, sd1)


# -- Miscellaneous -------------------------------------------------------


class _RetrievalScoreAxiom(_WithoutPrecond, ABC):
    @abstractmethod
    def calc_doc_sim_score(self, sim, query, doc_id) -> float:
        pass

    def retrieval_score(self, ctx: RerankingContext, query: Query, doc: Document) -> float:
        from axiomatic.collection.anserini import _J
        return self.calc_doc_sim_score(ctx.c.document_similarity_score(),
                                       _J.String(query._get_text()),
                                       _J.String(doc.doc_id))

    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        sd1 = self.retrieval_score(ctx, query, d1)
        sd2 = self.retrieval_score(ctx, query, d2)
        return 1 if sd1 > sd2 else (-1 if sd1 < sd2 else 0)


class RS_TF(_RetrievalScoreAxiom):
    def calc_doc_sim_score(self, sim, query: str, doc_id: str) -> float:
        return sim.tfSimilarity(query, doc_id)


class RS_TF_IDF(_RetrievalScoreAxiom):
    def calc_doc_sim_score(self, sim, query: str, doc_id: str) -> float:
        return sim.tfIdfSimilarity(query, doc_id)

class RS_BM25(_RetrievalScoreAxiom):
    def calc_doc_sim_score(self, sim, query: str, doc_id: str) -> float:
        return sim.bm25Similarity(query, doc_id)

class RS_PL2(_RetrievalScoreAxiom):
    def calc_doc_sim_score(self, sim, query: str, doc_id: str) -> float:
        return sim.pl2Similarity(query, doc_id)

class RS_QL(_RetrievalScoreAxiom):
    def calc_doc_sim_score(self, sim, query: str, doc_id: str) -> float:
        return sim.qlSimilarity(query, doc_id)
