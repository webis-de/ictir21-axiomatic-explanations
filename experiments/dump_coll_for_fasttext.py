#!/usr/bin/env python

import os

import tqdm

from axiomatic.collection import anserini

os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
anserini.SETUP(os.path.join(d, 'lib/Anserini'))
coll = anserini.AnseriniLuceneCollection(os.path.join(
        d, 'experiments', 'robust04',
        'lucene-index.robust04.pos+docvectors+rawdocs+transformedDocs'))

# dump normalized collection

import re

tag = re.compile(r'<[^>]+>')
space = re.compile(r'\s+')
notalpha = re.compile(r'[^a-z0-9]')
digits = dict(zip('0123456789', ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']))

def process(text):
    text = tag.sub("", text)
    text = text.lower()
    text = notalpha.sub(" ", text)
    for d in digits:
        text = text.replace(d, f' {digits[d]} ')
    text = space.sub(" ", text)
    return text.strip()

outfile = '/tmp/robust04.txt'


with open(outfile, "w") as f:
    for id in tqdm.tqdm(coll.get_all_docids(), total=coll.get_document_count()):
        doc = coll.get_document(id)
        t = process(doc.text)
        f.write(t)
        f.write(' ')
