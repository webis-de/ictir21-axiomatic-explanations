from abc import abstractmethod, ABC
from functools import lru_cache

import numbers
import typing


class _CollectionItem(ABC):
    @property
    def doc_id(self):
        return self._get_doc_id()

    @abstractmethod
    def _get_doc_id(self) -> str:
        """Return a unique identifier (at collection scope) for this item."""
        pass

    @property
    @lru_cache(maxsize=2**14)
    def termseq(self) -> typing.Sequence[str]:
        return self._get_term_sequence()

    @abstractmethod
    def _get_term_sequence(self) -> typing.Sequence[str]:
        """Return an iterable of the terms in this text object
        """
        pass

    @property
    @lru_cache(maxsize=2**14)
    def tf(self) -> typing.Mapping[str, numbers.Integral]:
        return self._get_tf()

    @abstractmethod
    def _get_tf(self) -> typing.Mapping[str, numbers.Integral]:
        """return a dict-like object mapping this document's terms to their frequencies."""
        pass

    @property
    @lru_cache(maxsize=2**14)
    def text(self) -> str:
        return self._get_text()

    @abstractmethod
    def _get_text(self) -> str:
        """Return the body text as a string."""
        pass

    def __len__(self):
        return len(self.termseq)

    def __repr__(self):
        shortcontents = " ".join(self.termseq)

        if len(shortcontents) > 100:
            shortcontents = shortcontents[:97] + '...'

        return "{}({},{})".format(
            self.__class__.__name__,
            self.doc_id,
            shortcontents)


# ------------------------------------------------------------------------


class Query(_CollectionItem):
    pass


# ------------------------------------------------------------------------

class Document(_CollectionItem):
    @abstractmethod
    def get_pagerank(self):
        """Return this document's pagerank."""


# ------------------------------------------------------------------------

class Collection(ABC):

    @abstractmethod
    def get_document_count(self) -> numbers.Integral:
        """Return the number of documents in this collection."""
        pass

    @abstractmethod
    def get_term_df(self, term: str) -> numbers.Integral:
        """Return the document frequency value for the given term
        :param term: a string containing the term.
        """
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Document:
        """Lookup a document by its id."""
        pass

    @abstractmethod
    def get_query(self, raw_qstr: str, q_id: str = None) -> Query:
        """Preprocess a query string to be compatible with the collection's documents."""
        pass


class IndexedCollectionMixin(ABC):
    @abstractmethod
    def get_retrieval_score(self, q: Query, d: Document) -> numbers.Real:
        """Return the retrieval score a given document would receive for a query"""
        pass


class IndexedCollection(Collection, IndexedCollectionMixin, ABC):
    pass


class TestCollectionMixin(ABC):
    @abstractmethod
    def get_qrel(self, q: Query, d: Document) -> numbers.Integral:
        """Return the relevance judgment for the (given query,document) pair
        :param q: the query
        :param d: the document
        """
        pass
