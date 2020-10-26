#!/usr/bin/env python

import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64/'

from axiomatic.explanations.traindata import build_training_set
from axiomatic.collection import anserini

import glob
import inspect

axioms = []

try:
    # clean scope for building axioms
    from axiomatic.axioms.pairwise import *
    axioms = [
        TFC1(),
        TFC3(),
        M_TDC(),
        LNC1(),
        TF_LNC(),
        LB1(),
        REG(),
        ANTI_REG(),
        AND(),
        DIV(),
        STMC1(),
        PROX1(),
        PROX2(),
        PROX3(),
    ]
finally:
    pass



print(axioms)

anserini.SETUP('../lib/Anserini')


gen = build_training_set(
    'sqlite:///robust04/runs/topics.robust04.sqlite::rundata',
    './robust04/topics.robust04.301-450.601-700.txt',
    './robust04/lucene-index.robust04.pos+docvectors+rawdocs+transformedDocs',
    axioms,
    max_rank=5
)


outfile = './training_set.csv'



with open(outfile, 'w') as f:
    outparam = dict(index=False, header=True)
    for df in gen:
        f.write(df.to_csv(**outparam))
        f.flush()
        outparam['header'] = False
