#!/bin/bash
# -*- coding: utf-8 ; mode: python -*-
"true" '''\'
# bash:
set -e

log() { >&2 echo $@ ; }
die() { log $@ ; exit 1; }

ENV_DIR=/axiomatic-explainability/axiomatic-explainability

if [ "$MAPRED_DATASET" = "MS-MARCO" ]; then
    ## ms-marco-topics configuration
    NUMBER_OF_REDUCERS=250
    INPUT_PATH=axexp/runs/all-topics-ms-marco.top1000.basic-approaches.jsonl
    OUTPUT_PATH='axexp/prefs/all-topics-ms-marco.top1000.basic-approaches.jsonl'
    ENVIRONMENT_VARIABLES="ENV_DIR=${ENV_DIR} TOPICS_FILE=msmarco-test2019-queries.tsv DATASET=ms-marco JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64"
elif [ "$MAPRED_DATASET" = "ROBUST04" ]; then
    ## robust04-topics configuration
    NUMBER_OF_REDUCERS=250
    INPUT_PATH=axexp/runs/all_topics.top1000.basic-approaches.jsonl
    OUTPUT_PATH='axexp/prefs/all_topics.top1000.basic-approaches.jsonl'
    ENVIRONMENT_VARIABLES="ENV_DIR=${ENV_DIR} TOPICS_FILE=topics.robust04.301-450.601-700.txt DATASET=robust04 JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/"
else
    die MAPRED_DATASET must be set
fi

if [ "$MAPRED_PAIR_METHOD" = "ranking2pairs" ]; then
    MAPPER_METHOD="ranking2pairs"
    die WRITE THIS CODE...
elif [ "$MAPRED_PAIR_METHOD" = "preferences2pairs" ]; then
    MAPPER_METHOD="preferences2pairs"
    INPUT_PATH='axexp/qrel-prefs/qrels-robust04.jsonl'
    OUTPUT_PATH='axexp/prefs/qrels-robust04.jsonl'
else
    die MAPRED_PAIR_METHOD must be set
fi

########################################################################

CEPH_DIR="/mnt/ceph/storage/data-in-progress/kibi9872/axexp/axiomatic-explainability/experiments/"
CONTAINER_DIR="/axiomatic-explainability/axiomatic-explainability/experiments/"
JOB_MOUNT_MAIN_FILE="-v ${CEPH_DIR}build_training_set_mapred.py:${CONTAINER_DIR}build_training_set_mapred.py"
ROBUST_RO_MOUNT="-v ${CEPH_DIR}robust04/lucene-index.robust04.pos+docvectors+rawdocs+transformedDocs:/robust-ro:ro"
MS_MARCO_RO_MOUNT="-v ${CEPH_DIR}ms-marco/lucene-index.msmarco-doc.pos+docvectors+rawdocs:/ms-marco-ro:ro"
JOB_PREFIX="docker run --rm -i ${JOB_MOUNT_MAIN_FILE} ${ROBUST_RO_MOUNT} ${MS_MARCO_RO_MOUNT} axiomatic-explainability-robust04:0.0.1 /bin/bash -c '/usr/bin/env PYTHONPATH=../src/ ${ENVIRONMENT_VARIABLES} pipenv run python3.6 build_training_set_mapred.py"

[ -z ${HADOOP_HOME:+x} ] && die HADOOP_HOME must be set

STREAMJAR=$( find -L ${HADOOP_HOME} -regex '.*/tools/lib/hadoop-streaming-[0-9.]+.jar' -print -quit )

[ -e "${STREAMJAR}" ] || die No streaming jar found below ${HADOOP_HOME}
log Using $STREAMJAR

${HADOOP_HOME}/bin/hadoop fs -rm -r -f "${OUTPUT_PATH}"

MAP_CMD="${JOB_PREFIX} ${MAPPER_METHOD}'"
REDUCER_CMD="${JOB_PREFIX} pairs2prefs'"

echo "${MAP_CMD}"

exit 2

${HADOOP_HOME}/bin/hadoop jar $STREAMJAR \
     -D mapreduce.reduce.memory.mb=$(( 1024 * 3 )) \
     -D mapreduce.map.memory.mb=$(( 1024 * 2 )) \
     -D mapred.reduce.tasks=${NUMBER_OF_REDUCERS} \
     -input "${INPUT_PATH}" \
     -output "${OUTPUT_PATH}" \
     -mapper "${MAP_CMD}" \
     -reducer "${REDUCER_CMD}" \
     ;

exit 0
'''
# python
import sys
import json
import pandas as pd
import os
import random
import string
from axiomatic.explanations.experiment_lib import MapredAccumulators, AxExpEnvironment, AxExpPrefProcessor
from axiomatic.axioms.pairwise import *


def axioms():
    return [
        TFC1(),
        TFC3(),
        M_TDC(),
        LEN_M_TDC(0.25),
        LEN_M_TDC(0.5),
        LEN_M_TDC(1.0),
        LNC1(),
        TF_LNC(),
        LB1(),
        REG(),
        ANTI_REG(),
        AND(),
        LEN_AND(0.1),
        LEN_AND(0.25),
        LEN_AND(0.5),
        LEN_AND(1.0),
        M_AND(),
        LEN_M_AND(0.1),
        LEN_M_AND(0.25),
        LEN_M_AND(0.5),
        LEN_M_AND(1.0),
        DIV(),
        LEN_DIV(0.1),
        LEN_DIV(0.25),
        LEN_DIV(0.5),
        LEN_DIV(1.0),
        STMC1(),
        STMC2(),
        STMC1_f(),
        STMC2_f(),
        STMC1_fr(),
        STMC2_fr(),
        PROX1(),
        PROX2(),
        PROX3(),
        PROX4(),
        PROX5(),
        RS_TF(),
        RS_TF_IDF(),
        RS_BM25(),
        RS_PL2(),
        RS_QL(),
    ]


def axexp_config(env_dir):
    if 'DATASET' in os.environ and 'robust04' == os.environ['DATASET']:
        return {
            'anserini_path': os.path.join(env_dir, 'lib', 'Anserini'),
            'index_path': os.path.join(env_dir, 'experiments', 'robust04',
                                       'lucene-index.robust04.pos+docvectors+rawdocs+transformedDocs'),
            'topics_path': os.path.join(env_dir, 'experiments', 'robust04', os.environ['TOPICS_FILE']),
            'use_sparkfiles': False,
        }
    if 'DATASET' in os.environ and 'ms-marco' == os.environ['DATASET']:

        try:
            os.environ['DATASET'] = 'robust04'
            AxExpEnvironment(**axexp_config(env_dir)).setup()
        except:
            # incredible ugly hack: I saw that anserini-0.7 works only if i load the old anserini up front.
            pass

        os.environ['DATASET'] = 'ms-marco'
        return {
            'anserini_path': os.path.join(env_dir, 'lib', 'anserini-0.7.1-snapshot'),
            'index_path': os.path.join(env_dir, 'experiments', 'ms-marco',
                                       'lucene-index.msmarco-doc.pos+docvectors+rawdocs'),
            'topics_path': os.path.join(env_dir, 'experiments', 'ms-marco', os.environ['TOPICS_FILE']),
            'use_sparkfiles': False,
        }

    raise ValueError('I dont know which configuration to use')


def create_initialized_ax_exp_environment(env_dir):
    env = AxExpEnvironment(**axexp_config(env_dir))
    env.setup()

    return AxExpPrefProcessor(
        axioms=axioms(),
        accumulators=MapredAccumulators(),
        env=env
    )


def emit(key, value):
    sys.stdout.write(str(key))
    sys.stdout.write('\t')
    sys.stdout.write(str(value))
    sys.stdout.write('\n')


def main(mode, env_dir):
    proc = create_initialized_ax_exp_environment(env_dir)

    if mode == 'ranking2pairs':
        # run in mapper
        for line in sys.stdin:
            try:
                ranking = pd.read_json(line)
            except:
                proc.A._inc('invalid json records')
                continue
            for pairdict in proc.generate_pairs(ranking):
                k = ''.join(random.choice(string.ascii_lowercase) for i in range(64))
                emit(k, json.dumps(pairdict))

    elif mode == 'preferences2pairs':
        for line in sys.stdin:
            # pairs are already created
            k = ''.join(random.choice(string.ascii_lowercase) for i in range(64))
            emit(k, line)
            pass

    elif mode == 'pairs2prefs':
        # run in reducer
        for line in sys.stdin:
            k, v = line.split('\t')
            try:
                pairdict = json.loads(v)
            except:
                proc.A._inc('invalid json records')
                continue
            for pref in proc.generate_preferences(pairdict):
                sys.stdout.write(json.dumps(pref))
                sys.stdout.write('\n')

    elif mode == 'ranking2prefs':
        # run in reducer-less mapper
        for line in sys.stdin:
            ranking = pd.read_json(line)
            for pairdict in proc.generate_pairs(ranking, topk_rank=10, num_random=200):
                proc.A._set_status(
                    f'Q={ranking["query"].values[0]} S={ranking["system"].values[0]} P={pairdict["id1"]}+{pairdict["id2"]}'
                )
                for pref in proc.generate_preferences(pairdict):
                    sys.stdout.write(json.dumps(pref))
                    sys.stdout.write('\n')
            proc.A._inc('rankings.done')

    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == '__main__':
    main(
        mode=sys.argv[1],
        env_dir=os.environ['ENV_DIR']
    )

