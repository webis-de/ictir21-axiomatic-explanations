import abc
import numbers
import typing

from axiomatic.collection import Collection, Query, Document
from axiomatic.features import Features



class RerankingContext(object):

    f: Features
    c: Collection

    def __init__(self, collection: Collection, features: Features):
        self.f = features
        self.c = collection




class Axiom(metaclass=abc.ABCMeta):

    @property
    def name(self):
        return self.__class__.__name__.split('.')[-1]

    @abc.abstractmethod
    def precondition(self, ctx: RerankingContext, *args, **kwargs) -> bool:
        pass

    @abc.abstractmethod
    def preference(self, ctx: RerankingContext, *args, **kwargs) -> numbers.Integral:
        pass

    def acronym(self):
        return self.name

    def __str__(self):
        return 'Axiom(%s)' % self.acronym()


class PairwiseAxiom(Axiom):
    @abc.abstractmethod
    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        pass

    @abc.abstractmethod
    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document):
        pass


class TriplewiseAxiom(Axiom):
    @abc.abstractmethod
    def precondition(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document, d3: Document):
        pass

    @abc.abstractmethod
    def preference(self, ctx: RerankingContext, query: Query, d1: Document, d2: Document, d3: Document):
        pass


class ListwiseAxiom(Axiom):
    @abc.abstractmethod
    def precondition(self, ctx: RerankingContext, query: Query, ranking: typing.List[Document]):
        pass

    @abc.abstractmethod
    def preference(self, ctx: RerankingContext, query: Query, ranking: typing.List[Document]):
        pass


# ------------------------------------------------------------------------


class ArgumentationAxiom(Axiom):
    """Mixin for argumentation-based axioms. """

    def __str__(self):
        return 'ArgumentationAxiom(%s)' % self.acronym()
