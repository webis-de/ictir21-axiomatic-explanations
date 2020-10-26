import collections
import numbers
import os
import typing
import re
import unicodedata
from functools import lru_cache

from axiomatic.collection import Collection, Query, Document

import jnius_config


class _J:
        _types = [
            'java.io.File',
            'java.lang.String',
            'java.net.URI',
            'java.nio.file.Paths',

            'io.anserini.index.IndexUtils',
            'io.anserini.index.generator.LuceneDocumentGenerator',
            'io.anserini.search.SearchCollection',
            'io.anserini.search.SimpleSearcher',
            'io.anserini.search.query.BagOfWordsQueryGenerator',
            'io.anserini.util.AnalyzerUtils',
            'io.anserini.search.similarity.DocumentSimilarityScore',

            'org.apache.lucene.analysis.en.EnglishAnalyzer',
            'org.apache.lucene.index.DirectoryReader',
            'org.apache.lucene.index.Term',
            'org.apache.lucene.index.Terms',
            'org.apache.lucene.search.DocValuesFieldExistsQuery',
            'org.apache.lucene.search.IndexSearcher',
            'org.apache.lucene.store.FSDirectory',
        ]


def SETUP(anserini_repo_dir):
    """Anserini must be built, and JAVA_HOME properly set, before calling this."""
    import sys
    sys.stderr.write(f'Initializing jnius from {anserini_repo_dir}\n')
    jnius_config.set_classpath(os.path.join(
        anserini_repo_dir, "target/anserini-0.3.1-SNAPSHOT-fatjar.jar"))

    from jnius import autoclass
    for t in _J._types:
        setattr(_J, t.split('.')[-1], autoclass(t))


def run_file_to_jsonl(input_file, output_file):
    from trectools import TrecRun
    import json
    with open(output_file, 'w') as out:
        queries = TrecRun(input_file).run_data.groupby('query')
        for query in queries.groups:
            out.write(json.dumps([i[1].to_dict() for i in queries.get_group(query).iterrows()]) + '\n')


def shuffled_jsonl_content(input_file):
    import json
    from random import Random
    from zlib import adler32
    with open(input_file) as f:
        ret = []
        for line in f:
            line = json.loads(line)
            seed = adler32(str.encode(str(line[0]['system']) + '-' + str(line[0]['query']), 'UTF-8'))
            Random(seed).shuffle(line)
            for i in range(0, len(line)):
                line[i]['rank'] = i+1
                line[i]['score'] = len(line) - i
                line[i]['system'] = line[i]['system'] + '-shuffle'

            ret += [json.dumps(line)]

        return '\n'.join(ret)


def shuffled_jsonl_file(input_file):
    with open(input_file + '-shuffled', 'w') as f:
        f.write(shuffled_jsonl_content(input_file))


class AnseriniLuceneDocument(Document):
    _re_striptags = re.compile(r'<[^ ][^>]*>')

    def __init__(self, doc_id, collection):
        self._doc_id = doc_id
        self._collection = collection

    def get_pagerank(self):
        raise NotImplementedError('pagerank')

    def _get_doc_id(self) -> str:
        return self._doc_id

    def _get_term_sequence(self) -> typing.Iterable[str]:
        return self._collection._tokenize(self.text)

    def _get_tf(self) -> typing.Mapping[str, numbers.Integral]:
        return self._collection._termvec(self.doc_id)

    @staticmethod
    def _striptags(text):
        return AnseriniLuceneDocument._re_striptags.sub("", text)

    def _get_text(self) -> str:
        text = self._collection.utils.getRawDocument(_J.String(self.doc_id))
        # anserini and/or jnius don't handle multi-byte chars well
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        text = AnseriniLuceneDocument._striptags(text)
        return text

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.doc_id)


class AnseriniQuery(Query):

    def __init__(self, id, termseq, qstr):
        self._id = id
        self._termseq = termseq
        self._tf = collections.Counter(termseq)
        self._qstr = qstr

    def _get_doc_id(self) -> str:
        return self._id

    def _get_term_sequence(self) -> typing.Iterable[str]:
        return self._termseq

    def _get_tf(self) -> typing.Mapping[str, numbers.Integral]:
        return self._tf

    def _get_text(self) -> str:
        return self._qstr


class AnseriniLuceneCollection(Collection):

    def __init__(self, index_path):
        self.utils = _J.IndexUtils(_J.String(index_path))
        self.directory = _J.FSDirectory.open(_J.File(_J.String(index_path)).toPath())
        self.reader = _J.DirectoryReader.open(self.directory)
        self.analyzer = _J.EnglishAnalyzer()
        self.FIELD_BODY = _J.String(_J.LuceneDocumentGenerator.FIELD_BODY)
        self.FIELD_ID = _J.String(_J.LuceneDocumentGenerator.FIELD_ID)

    def _termvec(self, doc_id):
        doc_id = self.utils.convertDocidToLuceneDocid(_J.String(doc_id))
        terms = self.reader.getTermVector(
            doc_id, self.FIELD_BODY)
        te = terms.iterator()
        tv = collections.defaultdict(int)
        while te.next() is not None:
            k = str(te.term().utf8ToString())
            v = int(te.totalTermFreq())
            tv[k] = v
        return tv

    def _tokenize(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        text = _J.String(text)
        return _J.AnalyzerUtils.tokenize(self.analyzer, text).toArray()

    def get_all_docids(self) -> typing.Iterable[str]:
        q = _J.DocValuesFieldExistsQuery(self.FIELD_ID)
        searcher = _J.IndexSearcher(self.reader.getContext())
        score_docs = searcher.search(q, self.reader.maxDoc(), _J.SearchCollection.BREAK_SCORE_TIES_BY_DOCID).scoreDocs
        for sd in score_docs:
            ifld = searcher.doc(sd.doc).getField(self.FIELD_ID)
            yield str(ifld.stringValue())

    @lru_cache(maxsize=2**12)
    def get_document(self, doc_id: str) -> Document:
        return AnseriniLuceneDocument(doc_id, self)

    def get_query(self, raw_qstr: str, q_id: str = None) -> Query:
        qtok = self._tokenize(raw_qstr)
        return AnseriniQuery(q_id, qtok, raw_qstr)

    @lru_cache(maxsize=None)
    def get_document_count(self) -> numbers.Integral:
        return self.reader.getDocCount(self.FIELD_BODY)

    @lru_cache(maxsize=2**14)
    def get_term_df(self, term: str) -> numbers.Integral:
        tt = self._tokenize(term)
        if len(tt) < 0:
            return 0
        jterm = _J.Term(self.FIELD_BODY, _J.String(tt[0]))
        return self.reader.docFreq(jterm)

    def document_similarity_score(self):
        return _J.DocumentSimilarityScore(self.reader)
