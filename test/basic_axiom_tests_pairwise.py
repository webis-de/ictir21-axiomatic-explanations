import unittest
from unittest import mock
import collections

from axiomatic.axioms import RerankingContext
from axiomatic.axioms.pairwise import *
from axiomatic.collection import IndexedCollection
from axiomatic.collection.inmemory import Document, Query
from axiomatic.features import Features


class BasicPairwiseAxiomTests(unittest.TestCase):

    def setUp(self):
        c = mock.Mock(spec=IndexedCollection)
        self.ctx = RerankingContext(
            collection=c,
            features=Features(c)
        )

    def test_tfc1(self):
        q = Query("w1 w2")
        d1 = Document('lorem ipsum w1 w1 w2')
        d2 = Document('foo w1 w2 w1 w1')

        ax = TFC1()

        self.assertTrue(ax.precondition(self.ctx, q, d1, d2))

        # d2 has more than 10% higher query termfreq sum
        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d2))

    def test_tfc3(self):
        q = Query("q1 q2 q3")
        d1 = Document('q1 q1 q1 q2')
        d2 = Document('q1 q1 q2 q3')

        # same term discrimination values
        self.ctx.f.td = mock.MagicMock(
            wraps=lambda t: collections.defaultdict(float, dict(
                q1=1, q2=1, q3=1
        ))[t])

        ax = TFC3()

        self.assertTrue(ax.precondition(self.ctx, q, d1, d2))

        # d2 should be higher because of the (q1, q3) term pair
        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d2))

    def test_tdc(self):
        q = Query("test query words")
        d1 = Document('This is the test document. It contains words and phrases.')
        d2 = Document('Another document. Contains query words but is not very interesting.')

        self.ctx.c.get_document_count = mock.MagicMock(return_value=2)
        self.ctx.c.get_term_df = mock.MagicMock(
            wraps=lambda t: collections.defaultdict(int, dict(
                test=1,
                query=10,
                words=100
            ))[t]
        )

        ax1 = M_TDC()

        prec = ax1.precondition(self.ctx, q, d1, d2)

        # d1, d2 have same length, same query term sum count and at least one query term count difference
        self.assertTrue(prec)

        pref = ax1.preference(self.ctx, q, d1, d2)

        # d1 contains the more discriminative term
        self.assertEqual(1, pref)

    def test_len_tdc_with_false_precondition(self):
        q = Query("test query words")
        d1 = Document('This is the test document. It contains words and phrases. a b c d')
        d2 = Document('Another document. Contains query words but is not very interesting.')

        self.ctx.c.get_document_count = mock.MagicMock(return_value=2)
        self.ctx.c.get_term_df = mock.MagicMock(
            wraps=lambda t: collections.defaultdict(int, dict(
                test=1,
                query=10,
                words=100
            ))[t]
        )

        ax1 = LEN_M_TDC(0.1)

        pref = ax1.preference(self.ctx, q, d1, d2)

        self.assertEqual('LEN_M_TDC_0.1', ax1.name)
        self.assertFalse(ax1.precondition(self.ctx, q, d1, d2))
        # d1 contains the more discriminative term
        self.assertEqual(1, pref)

    def test_lnc1(self):
        q = Query('q1 q2 q3')
        d1 = Document('q1 q2 q3 w1')
        d2 = Document('q1 q2 q3 w1 w2')

        ax = LNC1()

        # same tf for all query terms
        self.assertTrue(ax.precondition(self.ctx, q, d1, d2))

        # prefer the shorter document
        self.assertEqual(1, ax.preference(self.ctx, q, d1, d2))

    def test_len_tdc_with_true_precondition(self):
        q = Query("test query words")
        d1 = Document('This is the test document. It contains words and phrases. a b c d')
        d2 = Document('Another document. Contains query words but is not very interesting.')

        self.ctx.c.get_document_count = mock.MagicMock(return_value=2)
        self.ctx.c.get_term_df = mock.MagicMock(
            wraps=lambda t: collections.defaultdict(int, dict(
                test=1,
                query=10,
                words=100
            ))[t]
        )

        ax1 = LEN_M_TDC(0.3)

        pref = ax1.preference(self.ctx, q, d1, d2)

        self.assertEqual('LEN_M_TDC_0.3', ax1.name)
        self.assertTrue(ax1.precondition(self.ctx, q, d1, d2))
        # d1 contains the more discriminative term
        self.assertEqual(1, pref)

    def test_lnc1(self):
        q = Query('q1 q2 q3')
        d1 = Document('q1 q2 q3 w1')
        d2 = Document('q1 q2 q3 w1 w2')

        ax = LNC1()

        # same tf for all query terms
        self.assertTrue(ax.precondition(self.ctx, q, d1, d2))

        # prefer the shorter document
        self.assertEqual(1, ax.preference(self.ctx, q, d1, d2))

    def test_tf_lnc(self):
        q = Query('q1 q2 q3')

        d1 = Document('q1 q1 q2 x y')
        d2 = Document('q1 q2 x y')

        ax = TF_LNC()

        self.assertTrue(ax.precondition(self.ctx, q, d1, d2))

        self.assertEqual(1, ax.preference(self.ctx, q, d1, d2))


    def test_lb1(self):
        q = Query("test query words")
        d1 = Document(
            'this is the test document. It contains words and phrases.')
        d2 = Document(
            'another test document. Contains query words and interesting very much.')

        ax1 = LB1()

        # same retrieval scores -> precondition fulfilled
        self.ctx.c.get_retrieval_score = mock.MagicMock(return_value=1.0)

        self.assertTrue(ax1.precondition(self.ctx, q, d1, d2))

        pref = ax1.preference(self.ctx, q, d1, d2)

        # d1 doesn't have 'query' but d2 does
        self.assertEqual(-1, pref)

    def test_reg(self):
        q = Query('q1 q2 q3')
        d1 = Document('q1 q2 q3 q2 q3')
        d2 = Document('q1 q2 q3 q1')

        self.ctx.f.synset_similarity = mock.MagicMock(
            wraps=lambda w1, w2: collections.defaultdict(float, {
                ('q1', 'q2'): 1.0,
                ('q1', 'q3'): 0.8,
                ('q2', 'q3'): 0.1,
            })[tuple(sorted((w1, w2)))])

        ax = REG()

        pref = ax.preference(self.ctx, q, d1, d2)

        # d2 has higher tf of q1, which is most similar to the other terms
        self.assertEqual(1, pref)

    def test_anti_reg(self):
        q = Query('q1 q2 q3')
        d1 = Document('q1 q2 q3 q2 q3')
        d2 = Document('q1 q2 q3 q1')

        self.ctx.f.synset_similarity = mock.MagicMock(
            wraps=lambda w1, w2: collections.defaultdict(float, {
                ('q1', 'q2'): 1.0,
                ('q1', 'q3'): 0.8,
                ('q2', 'q3'): 0.1,
            })[tuple(sorted((w1, w2)))])

        ax = ANTI_REG()

        pref = ax.preference(self.ctx, q, d1, d2)

        # d2 has lower tf of q1, which is most similar to the other terms
        self.assertEqual(-1, pref)

    def test_and(self):
        q = Query('q1 q2 q3')
        d1 = Document('a b q1 q2 q2 q2 q1 q1 q2')
        d2 = Document('q3 b q1 q2 q2 q2 q1 q1 q2')

        ax = AND()
        ax1 = M_AND()
        # d2 has all query terms, but not d1
        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d2))
        # d2 also has larger subset of query terms
        self.assertEqual(-1, ax1.preference(self.ctx, q, d1, d2))
        d1 = Document('q1 q1 q1 q1')
        d2 = Document('q1 q1 q1 q2')

        self.assertEqual(0, ax.preference(self.ctx, q, d1, d2))
        self.assertEqual(-1, ax1.preference(self.ctx, q, d1, d2))

    def test_len_and_with_false_precondition(self):
        q = Query('c a')
        d1 = Document('b c')
        d2 = Document('a c b')

        ax = LEN_AND(0.3)

        self.assertEqual('LEN_AND_0.3', ax.name)
        self.assertFalse(ax.precondition(self.ctx, q, d1, d2))
        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d2))

    def test_len_and_with_true_precondition(self):
        q = Query('e b')
        d1 = Document('b e')
        d2 = Document('a c b')

        ax = LEN_AND(0.4)

        self.assertTrue(ax.precondition(self.ctx, q, d1, d2))
        self.assertEqual(1, ax.preference(self.ctx, q, d1, d2))

    def test_len_m_and_with_false_precondition(self):
        q = Query('q1 q2 q3')
        d1 = Document('a b a b q1 q2 q2 q2 q1 q1 q2')
        d2 = Document('q3 b q1 q2 q2 q2 q1 q1 q2')

        ax = LEN_M_AND(0.1)

        self.assertFalse(ax.precondition(self.ctx, q, d1, d2))
        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d2))

        d1 = Document('q1 q1 q1 q1')
        d2 = Document('a b c q1 q1 q1 q2')

        self.assertFalse(ax.precondition(self.ctx, q, d1, d2))
        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d2))

    def test_len_m_and_with_true_precondition(self):
        q = Query('q1 q2 q3')
        d1 = Document('a b a b q1 q2 q2 q2 q1 q1 q2')
        d2 = Document('q3 b q1 q2 q2 q2 q1 q1 q2')

        ax = LEN_M_AND(0.3)

        self.assertTrue(ax.precondition(self.ctx, q, d1, d2))
        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d2))

        d1 = Document('q1 q1 q1 q1')
        d2 = Document('a b c q1 q1 q1 q2')

        self.assertEqual('LEN_M_AND_0.3', ax.name)
        self.assertFalse(ax.precondition(self.ctx, q, d1, d2))
        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d2))

    def test_div(self):
        q = Query('q1 q2 q3')
        d1 = Document('q1 q2 q3')
        d2 = Document('foo bar baz')
        ax = DIV()

        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d2))

    def test_len_div_with_false_precondition(self):
        q = Query('q1 q2 q3')
        d1 = Document('q1 q2 q3')
        d2 = Document('foo bar baz bab bac')
        ax = LEN_DIV(0.1)

        self.assertEqual('LEN_DIV_0.1', ax.name)
        self.assertFalse(ax.precondition(self.ctx, q, d1, d2))
        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d2))

    def test_len_div_with_true_precondition(self):
        q = Query('q1 q2 q3')
        d1 = Document('foo bar baz bab bac')
        d2 = Document('q1 q2 q3')
        ax = LEN_DIV(0.5)

        self.assertEqual('LEN_DIV_0.5', ax.name)
        self.assertTrue(ax.precondition(self.ctx, q, d1, d2))
        self.assertEqual(1, ax.preference(self.ctx, q, d1, d2))

    def test_stmc1(self):
        q = Query("blue car moves")
        d1 = Document('blue auto goes through the city')
        d2 = Document('red airplane flies in the sky')

        self.ctx.f.synset_similarity = mock.MagicMock(
            wraps=lambda w1, w2: collections.defaultdict(float, {
                ('auto', 'car'): 1.0,
                ('airplane', 'car'): 0.2,
        })[tuple(sorted((w1, w2)))])

        ax1 = STMC1()
        pref = ax1.preference(self.ctx, q, d1, d2)

        # d1 should be higher because it has a more similar term
        self.assertEqual(1, pref)

    def test_stmc2(self):
        q = Query("q")
        d1 = Document('q')
        d2 = Document('t t t t')

        termsim = collections.defaultdict(float, {
                ('q', 't'): 1.0
            })

        self.ctx.f.synset_similarity = mock.MagicMock(
            wraps=lambda w1, w2: termsim[tuple(sorted((w1, w2)))])

        ax1 = STMC2()
        pref = ax1.preference(self.ctx, q, d2, d1)

        # d1 should be higher because it has an exact match
        self.assertEqual(-1, pref)

        # a more complex test for list index regression
        q = Query("q1 q2")
        d1 = Document('q1 v v v q2')
        d2 = Document('q2 t t t s')

        termsim[('q1','t')] = 1.0
        termsim[('q2','v')] = 0.5

        pref = ax1.preference(self.ctx, q, d1, d2)
        ## should not raise
        self.assertTrue(True)

    def test_prox1(self):
        q = Query("blue car")
        d1 = Document('a blue car goes through the city')
        d2 = Document('through the city blue goes car goes')

        ax1 = PROX1()
        pref = ax1.preference(self.ctx, q, d1, d2)

        self.assertEqual(1, pref)

    def test_prox2(self):
        q = Query('q1 q2')
        d1 = Document('q1 x q2 y z a b c')
        d2 = Document('x y q1 q2')

        ax = PROX2()

        self.assertTrue(ax.precondition(self.ctx, q, d1, d2))

        # d1 contains query terms earlier
        self.assertEqual(1, ax.preference(self.ctx, q, d1, d2))

    def test_prox3(self):
        q = Query('q1 q2')
        d1 = Document('a b c q1 d q2 e q1 q2')
        d2 = Document('a q2 b q1 q2')
        d3 = Document('q1 b q2')

        ax = PROX3()

        self.assertTrue(ax.precondition(self.ctx, q, d1, d2))
        self.assertTrue(ax.precondition(self.ctx, q, d1, d3))
        self.assertTrue(ax.precondition(self.ctx, q, d2, d3))

        # d2 contains query phrase earlier
        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d2))

        # d3 does not contain query phrase
        self.assertEqual(1, ax.preference(self.ctx, q, d1, d3))

    def test_prox4(self):
        q = Query('q1 q2')
        d1 = Document('a b c q1 d q2 e q1')
        d2 = Document('a q2 b q2 q1')
        d3 = Document('a b c q1 d q2 e q2 f q1')

        d4 = Document('a b c d  q1 q2')
        d5 = Document('a b c q1 q1 q2')

        ax = PROX4()
        # each d contains all query terms
        self.assertTrue(ax.precondition(self.ctx, q, d1, d2))
        self.assertTrue(ax.precondition(self.ctx, q, d1, d3))
        self.assertTrue(ax.precondition(self.ctx, q, d2, d3))

        # d2 has a closer grouping
        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d2))

        # d3 has an equally close grouping more often
        self.assertEqual(-1, ax.preference(self.ctx, q, d1, d3))

        # d5 has an additional zero-gap grouping via repeated query term
        self.assertEqual(-1, ax.preference(self.ctx, q, d4, d5))

    def test_prox5(self):
        q = Query('q1 q2 q3')
        d1 = Document('q1 q2 q3')
        d2 = Document('q1 a q2 b c q3')

        ax = PROX5()

        self.assertEqual(1, ax.preference(self.ctx, q, d1, d2))

    #def test_apd(self):
    #    ## TODO: proper unit test for APD
    #    pass
    #    q = Query("blue car city")
    #    d1 = Document('through the city a blue car goes. we should ban a blue car')
    #    d2 = Document('we should ban a blue car. we should ban a yellow helicopter ')

    #    ax1 = APD()
    #    pref = ax1.preference(q, d1, d2)
    #    print(pref)                 # -1 swap document ordering, 0 and 1 do not change

if __name__ == "__main__":
    unittest.main()
