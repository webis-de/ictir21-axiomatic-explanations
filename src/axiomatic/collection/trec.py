import io
import numbers

import trectools

from axiomatic.collection import TestCollectionMixin, Query, Document, IndexedCollectionMixin, Collection, inmemory


class TrecRunIndexedCollection(IndexedCollectionMixin):

    def __init__(self, run_file):
        self.run = trectools.TrecRun(run_file)

    def set_system(self, s):
        self.system = s

    def get_retrieval_score(self, q: Query, d: Document) -> numbers.Real:
        df = self.run.run_data
        result = df[(df['query'] == q.doc_id) & (df['docid'] == d.doc_id) & (df['system'] == self.system)]
        if len(result) > 0:
            return list(result['score'].head(1))[0]


class TrecRunDfIndexedCollection(TrecRunIndexedCollection):
    def __init__(self, run_df):
        self.run = type('run', (object,), {'run_data': run_df})

    def set_rundata(self, run_df):
        self.run.run_data = run_df


class TrecQrelTestCollection(TestCollectionMixin):

    def __init__(self, qrel_file):
        self.qrel = trectools.TrecQrel(qrel_file)

    def get_qrel(self, q: Query, d: Document) -> numbers.Integral:
        return self.qrel.get_judgement(d.doc_id, q.doc_id)


class TrecRobustQueries(object):

    def __init__(self, filename: str, collection_for_processing: Collection = None):
        if 'ms-marco' in filename:
            self._q = self.ms_marco_queries(filename, collection_for_processing)
        else:
            self._q = self.robust_queries(filename, collection_for_processing)

    @staticmethod
    def robust_queries(filename: str, collection_for_processing: Collection = None):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(io.open(filename, 'r'), 'lxml')
        ret = {}
        for topic in soup.findAll('top'):
            ttxt = topic.findNext('num').getText().strip().split('\n')[0].lower()
            ttxt = ttxt.replace('number:', '')
            qid = ttxt.strip()

            qstr = topic.findNext('title').getText().strip().split('\n')[0].strip()

            if collection_for_processing is not None:
                q = collection_for_processing.get_query(qstr, qid)
            else:
                q = inmemory.Query(qstr, qid)

            ret[qid] = q

        return ret

    @staticmethod
    def ms_marco_queries(filename: str, collection_for_processing: Collection = None):
        with open(filename, 'r') as f:
            ret = {}
            for line in f.readlines():
                line = line.split('\t')

                if len(line) != 2:
                    raise ValueError('Parsing Error for query "' + str(line) + '"')

                qid = line[0].strip()
                qstr = line[1].strip()

                if collection_for_processing is not None:
                    q = collection_for_processing.get_query(qstr, qid)
                else:
                    q = inmemory.Query(qstr, qid)

                ret[qid] = q

            return ret

    def __getitem__(self, item):
        return self._q[item]
