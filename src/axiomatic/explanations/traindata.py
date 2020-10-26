import typing
import itertools
import io

import pandas as pd
import numpy as np
import tqdm
import blaze

from axiomatic.axioms import PairwiseAxiom, RerankingContext
from axiomatic.collection import Document, Query


# deprecated.
from axiomatic.collection.anserini import AnseriniLuceneCollection
from axiomatic.collection.trec import TrecRunIndexedCollection, TrecRobustQueries
from axiomatic.features import Features


def get_concordant_pair_preferences(
        context: RerankingContext,
        axioms: typing.Sequence[PairwiseAxiom],
        query: Query,
        ranking: typing.Sequence[Document]):

    indexes = np.arange(len(ranking))
    pairs = [(i1, i2) for i1, i2 in itertools.combinations(indexes, 2) if i1 < i2]

    # rows are axioms, cols are concordant pairs
    preferences = np.zeros((len(axioms), len(pairs)))

    for col, (di1, di2) in enumerate(pairs):
        d1 = ranking[di1]
        d2 = ranking[di2]
        for row, ax in enumerate(axioms):
            if ax.precondition(context, query, d1, d2):
                preferences[row, col] = ax.preference(context, query, d1, d2)

    return preferences


def build_training_set_for_one_ranking(
        context: RerankingContext,
        axioms: typing.Sequence[PairwiseAxiom],
        query: Query, system: str,
        ranking: typing.Sequence[Document], loop_wrapper=lambda i: i):
    # create a docpair-by-axioms matrix
    indexes = np.arange(len(ranking))
    concordant_pairs = [(i1, i2) for i1, i2 in itertools.combinations(indexes, 2) if i1 < i2]

    df = pd.DataFrame(
        columns=['query', 'system'] +
            ['ax_{}'.format(a.name) for a in axioms] +
            ['rankdiff', 'upper_rank', 'scorediff', 'upper_score', 'concordant']
        ,
        index=np.arange(len(concordant_pairs)*2)
    )

    df['query'] = query.doc_id
    df['system'] = system
    from collections import defaultdict
    fail = defaultdict(int)

    for row, (di1, di2) in enumerate(concordant_pairs):
        d1 = ranking[di1]
        d2 = ranking[di2]
        cidx = row
        didx = row + len(concordant_pairs)
        for ax in loop_wrapper(axioms):
            pref = 0
            if ax.precondition(context, query, d1, d2):
                try:
                    context.c.set_system(system)
                    pref = ax.preference(context, query, d1, d2)
                except:
                    # TODO: error handling
                    fail[ax.name] += 1
                    print(dict(fail))

            df.loc[cidx, 'ax_{}'.format(ax.name)] = pref
            df.loc[didx, 'ax_{}'.format(ax.name)] = -pref

        df.loc[cidx, 'concordant'] = 1
        df.loc[didx, 'concordant'] = -1

        rd = np.abs(di1 - di2)
        df.loc[cidx, 'rankdiff'] = rd
        df.loc[didx, 'rankdiff'] = rd
        df.loc[cidx, 'upper_rank'] = di1
        df.loc[didx, 'upper_rank'] = di1

        if hasattr(context.c, 'get_retrieval_score'):
            sc1 = context.c.get_retrieval_score(query, d1)
            sd = np.abs(sc1 - context.c.get_retrieval_score(query, d2))
            df.loc[cidx, 'scorediff'] = sd
            df.loc[didx, 'scorediff'] = sd
            df.loc[cidx, 'upper_score'] = sc1
            df.loc[didx, 'upper_score'] = sc1

    return df

class AutoDelegate(object):
    def __init__(self, *delegates):
        self.delegates = delegates

    def __getattr__(self, item):
        for d in self.delegates:
            if hasattr(d, item):
                return getattr(d, item)


def initialize(rundb, topicsfile, index_path, max_rank):

    #buf = io.StringIO()

    #for fn in runfiles:
    #    with open(fn) as f:
    #        buf.write(f.read())
    #buf.seek(0)

    rcoll = TrecRunIndexedCollection(None)

    #rcoll.run.run_data = pd.read_csv(
    #    buf, sep='\s+', names=['query', 'q0', 'docid', 'rank', 'score', 'system', 'other'])
    rcoll.run.run_data = blaze.data(
        rundb
    )
    #rcoll.run.run_data.sort(['query', 'score'], inplace=True, ascending=[True, False])


    queries_by_id = None

    if index_path is not None:
        icoll = AnseriniLuceneCollection(index_path)
        coll = AutoDelegate(rcoll, icoll)
        queries_by_id = TrecRobustQueries(topicsfile, collection_for_processing=icoll)
    else:
        coll = rcoll

    ctx = RerankingContext(coll, Features(coll))
    rd = rcoll.run.run_data
    rd = rd[rd['rank'] <= max_rank]

    return queries_by_id, rd, ctx



def build_training_set(
        rundb: str,
        topicsfile: str,
        index_path: str,
        axioms: typing.Sequence[PairwiseAxiom],
        max_rank=100):

    queries_by_id, rd, ctx = initialize(rundb, topicsfile, index_path, max_rank)

    system_query = rd[['system', 'query']].distinct().sort('query')

    from blaze import by, merge
    ranking_lengths = by(merge(rd.system, rd.query), n=rd.rank.count()).n
    cpair_count = int((ranking_lengths * ranking_lengths - ranking_lengths).sum())
    iter_count = cpair_count * len(axioms) * 2

    pbar = tqdm.tqdm(total=iter_count)

    def loop(i):
        for item in i:
            pbar.update()
            yield item


    for sys, qid in system_query:
        sqrun = rd[(rd['system'] == sys) & (rd['query'] == qid)]
        ranking = [ctx.c.get_document(did[0]) for did in sqrun.docid]
        query = queries_by_id[qid]

        part = build_training_set_for_one_ranking(ctx, axioms, query, sys, ranking, loop)

        yield part


