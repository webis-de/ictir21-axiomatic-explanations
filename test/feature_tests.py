import unittest
from unittest import mock

import numpy as np

from axiomatic.collection import Collection
from axiomatic.features import Features
from axiomatic.features.prefs import approximatelyEqual


class FeatureTests(unittest.TestCase):
    """Regression tests for axiom features and basic components"""

    @classmethod
    def setUpClass(cls):
        import nltk
        nltk.download('wordnet')
        c = mock.MagicMock(spec=Collection)
        cls.f = Features(c)

    def test_approx_equal(self):
        self.assertTrue(approximatelyEqual(11, 10, 10.5, marginFrac=0.1))
        self.assertTrue(approximatelyEqual([11, 10, 10.5], marginFrac=0.1))

    def test_approx_not_equal(self):
        self.assertFalse(approximatelyEqual(9, 8, marginFrac=0.1))
        self.assertFalse(approximatelyEqual([9, 8], marginFrac=0.1))

    def test_approx_equal_zero(self):
        self.assertTrue(approximatelyEqual(0, 0, 0))
        self.assertTrue(approximatelyEqual([0, 0, 0]))

    def test_synset_similarity(self):
        sim = self.f.synset_similarity('boat', 'ship')
        self.assertGreater(sim, .9)

    def test_synset_similarity_nonword(self):
        sim = self.f.synset_similarity('aobesoaosesa', 'xzzaoezzoxzazoao')
        self.assertTrue(True)

    def test_embedding_similarity(self):
        sims = self.f.embedding_similarities("dog", ["cat", "fish", "airplane"])
        midx = np.argmax(sims)
        self.assertEqual(0, midx)

    def test_vocab_overlap(self):
        words = lambda s: set(s.split())
        c1 = words('a b c')
        c2 = words('a b d')

        self.assertEqual(1, self.f.vocab_overlap(c1, c1))
        self.assertEqual(.5, self.f.vocab_overlap(c1, c2))

    def test_avg_between_singleterm_query(self):
        words1 = 'a'.split()
        words2 = 'a b c d a d'.split()
        self.assertEqual(0, self.f.average_between_qterms(words1, words2))


if __name__ == '__main__':
    unittest.main()

