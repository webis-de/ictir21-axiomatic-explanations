import unittest
import os

from axiomatic.axioms import RerankingContext
from axiomatic.collection import anserini
from axiomatic.explanations.experiment_lib import AxExpEnvironment
from axiomatic.features import Features


class IntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
        d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        anserini.SETUP(os.path.join(d, 'lib/Anserini'))
        cls.c = anserini.AnseriniLuceneCollection(
            os.path.join(
                d, 'experiments', 'robust04',
                'lucene-index.robust04.pos+docvectors+rawdocs+transformedDocs'))

    def test_strip_tags(self):
        d = self.c.get_document('LA122990-0029')
        self.assertFalse('<P>' in d.text)

    def test_termseq_nonascii(self):
        d = self.c.get_document('FBIS4-20656')
        # shouldn't crash
        print(d.termseq)

    def test_missing_lemma(self):
        from axiomatic.axioms.pairwise import STMC2
        ax = STMC2()
        d1 = self.c.get_document('FT941-14380')
        d2 = self.c.get_document('FR941017-2-00055')
        q = self.c.get_query('lyme diseas')

        pref = ax.preference(RerankingContext(self.c, Features(self.c)), q, d1, d2)
        print(pref)

    def test_ids(self):
        docs = self.c.get_all_docids()

        print("first ten docs:", " ".join(x[1] for x in zip(range(10), docs)))

        print(self.c.get_document(next(docs)).text)

    def test_full_run(self):
        run = './../test/testrun-qspell-drmm-2397854811.tsv'
        run = './../test/testrun-qspell-drmm-105014.tsv'

        anserini = "../lib/Anserini"
        index = "../experiments/robust04/lucene-index.robust04.pos+docvectors+rawdocs+transformedDocs"
        topics = "../experiments/robust04/topics.webis-qspell-17.txt"

        env = AxExpEnvironment(
            anserini_path=anserini,
            index_path=index,
            topics_path=topics,
            use_sparkfiles=False)

        os.system(f'ls -ahl {run}')


if __name__ == '__main__':
    unittest.main()
