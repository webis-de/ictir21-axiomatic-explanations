import axiomatic.axioms
from .. import preference_matrix


class Orig(axiomatic.axioms.ListwiseAxiom):
    def preference(self, ctx: axiomatic.RerankingContext, query, ranking):
        return preference_matrix.PreferenceMatrix.from_ranking(ranking, self.name)
