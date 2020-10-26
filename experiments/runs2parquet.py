#!/usr/bin/env python3

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import itertools
import sys
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('outfile', type=str)
ap.add_argument('--chunk-size', '-c', default=(1024 * 200))

args = ap.parse_args()

cols = ['query', 'q0', 'docid', 'rank', 'score', 'system', 'other']
types = ['str', 'str', 'str', 'int16', 'float32',  'str', 'int8']

tmp = pd.DataFrame(
    [['q', 1, 'DOC', 1, 1.28392, 'SYS', 0]],
    columns=cols
)
tmp = tmp.astype(dtype=dict(zip(cols, types)))
tmp = pa.Table.from_pandas(tmp)

def data():
    for line in sys.stdin:
        q, q0, d, r, sc, sy, ot = line.split('\t')
        yield [q, q0, d, int(r), float(sc), sy, int(ot)]


def nchnk(it, n=args.chunk_size):
    while True:
        ch = itertools.islice(it, 0, n)
        z = next(ch)
        if z:
            yield itertools.chain([z], ch)
        else:
            break


with pq.ParquetWriter(args.outfile, tmp.schema) as writer:
    for ch in nchnk(data()):
        df = pd.DataFrame(list(ch), columns=cols)
        df = df.astype(dtype=dict(zip(cols, types)))
        writer.write_table(pa.Table.from_pandas(df))
