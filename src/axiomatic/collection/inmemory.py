import axiomatic.collection as acol

from collections import Counter

from axiomatic.collection import Query, Document


class InMemoryCollection(acol.Collection):

    def get_retrieval_score(self, q: Query, d: Document):
        raise NotImplementedError()

    docs = {}
    dfs = {}

    def get_term_df(self, term):
        return self.dfs[term]

    def get_qrel(self, q, d):
        pass


# def __init__(self):
    #     self.docs_queries = {}
    #     self.idfs = {}
    #     self.qrels = {}
    #     self.pageranks = {}
    #
    # # ------------------------------------------------------------
    #
    # def set_doc_or_query(self, id, text, tok=lambda s: s.split()):
    #     self.docs_queries[id] = tok(text)
    #
    # def set_idf(self, term, idf):
    #     self.idfs[term] = idf
    #
    # def set_qrel(self, qid, did, qrel):
    #     self.qrels[(qid, did)] = qrel
    #
    # def set_pagerank(self, docid, prank):
    #     self.pageranks[docid] = prank
    #
    # # ------------------------------------------------------------
    #
    # def get_terms(self, id):
    #     return self.docs_queries[id]
    #
    # def get_term_idf(self, term):
    #     return self.idfs[term]
    #
    # def get_document_pagerank(self, docid):
    #     return self.pageranks[docid]
    #
    # def get_qrel(self, qid, docid):
    #     return self.qrels[(qid, docid)]



class _ConcreteInMemoryCollectionItem(acol._CollectionItem):
    def __init__(self, text, id=None):
        self._text = text
        self._id = id

    def _get_doc_id(self):
        return self._id

    def _get_term_sequence(self):
        return self._text.split()

    def _get_tf(self):
        return Counter(self._text.split())

    def _get_text(self):
        return self._text


class Document(acol.Document, _ConcreteInMemoryCollectionItem):

    def get_pagerank(self):
        raise NotImplementedError('pagerank')


    def __str__(self):
        return 'Document(id=%s)' % self.docid


class Query(acol.Query, _ConcreteInMemoryCollectionItem):

    def __str__(self):
        return 'Query(id=%s,text=%s)' % (self.queryid, self.text)


