#!/usr/bin/env python3

import sys
import argparse
import pandas as pd
import io
import json

buf = io.StringIO()

cur_sys, cur_q = None, None


def writebuf():
    buf.seek(0)
    df = pd.read_csv(buf, sep='\t', header=None, names=[
        'query', 'q0', 'docid', 'rank', 'score', 'system', 'other'
    ])
    df.drop(columns=['q0', 'other'], inplace=True)
    d = df.to_dict(orient='list')
    d['system'] = str(d['system'][0])
    d['query'] = str(d['query'][0])
    sys.stdout.write(json.dumps(d))
    sys.stdout.write('\n')


for line in sys.stdin:
    q, _, docid, rank, score, system, _ = line.split('\t')
    if q != cur_q or system != cur_sys:
        if cur_sys is not None:
            writebuf()
        buf.seek(0)
        buf.truncate(0)
        cur_q = q
        cur_sys = system
    buf.write(line)

writebuf()
