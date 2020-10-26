#!/usr/bin/env python3
from pyspark import SparkContext, SparkConf, sql
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pyspark.sql.types as T
import argparse
import json


import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('input')
ap.add_argument('output')
ap.add_argument('--num-executors', '-e', type=int, required=True)
ap.add_argument('--num-cores', '-c', type=int, default=2)
ap.add_argument('--num-partitions', '-p', type=int, default=1000)
ap.add_argument('--memory', '-m', type=str, default='4g')

args = ap.parse_args()



sc = SparkContext(
    conf=SparkConf()
        .setAppName('parquet2json')
        .set('spark.executor.instances', str(args.num_executors))
        .set('spark.executor.cores', str(args.num_cores))
        .set('spark.executor.memory', args.memory)
        .set('spark.network.timeout', '480s')
        .set('spark.sql.shuffle.partitions', str(args.num_partitions))
        .set('spark.speculation', 'true')
        .set('spark.speculation.interval', '500ms')
        .set('spark.speculation.quantile', '0.98')
        .setMaster('yarn')
)
spark = sql.SparkSession(sc)




pq = spark.read.parquet(args.input)

cols = sc.broadcast(list(pq.schema.fieldNames()))

@pandas_udf(T.StructType([T.StructField('_json', T.StringType(), False)]), PandasUDFType.GROUPED_MAP)
def ranking_to_json(ranking):
    #ranking = pd.DataFrame(list(ranking_iter), columns=cols.value)

    c = ['_json']
    if len(ranking) < 1:
        return pd.DataFrame(columns=c, data=[])
    else:
        return pd.DataFrame(columns=c, data=[json.dumps(ranking.to_dict(orient='list'))])


(pq.groupby('query', 'system')
 .apply(ranking_to_json)
 .write.mode('overwrite').text(args.output)
)
