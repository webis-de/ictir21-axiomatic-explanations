import socket
import sys
import os
import pickle
import base64

import matplotlib.pyplot as plt
plt.rc('figure', facecolor='w')
import seaborn as sns
import pandas as pd
import numpy as np

import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.types as T

import pyspark.sql.functions as sqlf
from pyspark.sql.functions import udf, pandas_udf, PandasUDFType


import sklearn, sklearn.ensemble
import sklearn.linear_model
import sklearn.tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, GroupShuffleSplit

########################################################################

default_data_file = 'explanations-fidelity-summary.jsonl'

def record_fidelity_experiment(summary_df, item_id, scope, scope_detail, features,
                               perfmeasure='10-fold cv accuracy',
                               acc_key='cv_accuracy_mean',
                               ntr_key='avg_train_samples_per_fold',
                               nts_key='avg_test_samples_per_fold',
                               data_file=default_data_file
                              ):

    st = summary_df.groupby('system').describe().T

    acc_by_system = st.loc[acc_key,'mean']
    num_models_per_sys_fold = st.loc[acc_key,'count'].values[0]

    ntr = st.loc[ntr_key, 'mean'].values[0]
    nts = st.loc[nts_key, 'mean'].values[0]

    data = dict(item_id=item_id, Scope=scope, 
                Scope_detail=scope_detail,
                Perfmeasure=perfmeasure,
                Features=features,
                Num_models_per_system_per_fold=num_models_per_sys_fold,
                Num_train_per_fold=ntr,
                Num_test_per_fold=nts,
                **{'accuracy_%s'%k.replace('-', '_'):v for k,v in acc_by_system.items()})

    store_experiment(data, data_file)
    
    
def store_experiment(data, data_file=default_data_file):
    item_id = data['item_id']
    if os.path.isfile(data_file):
        df = pd.read_json(data_file)
        df.loc[item_id] = pd.Series(data)
    else:
        df = pd.DataFrame([data]).set_index('item_id')
        
    df.to_json(data_file, index=True)
    
    
def ranking_hexbin_plot(df, gridsize=10):
    tmp = df.copy()
    tmp['lower_rank'] = tmp.upper_rank - tmp.rankdiff
    tmp = tmp.groupby(['upper_rank', 'lower_rank']).correct.mean().reset_index()
    nbins = 20
    x, y, z = tmp.upper_rank, tmp.lower_rank, tmp.correct
    f, axx = plt.subplots(1, 2, sharey=True)
    ax = axx.flatten()[0]
    p = ax.hexbin(x, y, gridsize=gridsize)
    ax.set_ylabel('rank of first document')
    ax.set_xlabel('rank of second document')
    ax.set_title('Number of samples by location in ranking')
    plt.colorbar(p, ax=ax)
    ax = axx.flatten()[1]
    p = ax.hexbin(x, y, z, gridsize=gridsize)
    ax.set_xlabel('rank of second document')
    ax.set_title('Accuracy by location in ranking')
    plt.colorbar(p, ax=ax)
    f.set_size_inches((10, 4))
    f.tight_layout()

def should_keep_line(json_line):
    import json
    parsed = json.loads(json_line)

    if 'query' not in parsed or 'system' not in parsed:
        return False

    robust_eval_topics = ['301', '303', '306', '307', '310', '314', '318', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '336', '339', '341', '344', '345', '346', '347', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '362', '363', '365', '367', '372', '374', '375', '378', '379', '383', '385', '389', '393', '394', '397', '399', '401', '404', '405', '408', '409', '414', '416', '419', '425', '426', '427', '433', '435', '436', '439', '441', '442', '443', '445', '448', '450', '610', '611', '612', '613', '614', '615', '616', '617', '618', '619', '635', '641', '646', '647', '648', '649', '655', '680', '681', '682', '684', '685', '686', '687', '688', '689']
    robust_eval_systems = ['DRMM', 'MP-COS', 'PACRR-DRMM', 'bm25', 'tf-idf', 'pl2']
    ms_marco_eval_topics = ['146187', '654723', '719381', '1063750', '588587', '87181', '532603', '1132213', '89928', '1128373', '904965', '855410', '443396', '1059231', '53175', '833860', '474735', '691330', '398483', '903469', '1126813', '1128856', '1119729', '1121166', '1126206', '1104492', '1115569', '432930', '1121986', '405717', '359349', '1133328', '1047259', '1136427', '964223', '1121402', '494835', '1055865', '1134463', '1126814', '1106293', '502261', '278813', '423273', '541571', '523270', '1135377', '1108307', '1113437', '1117346', '1124464', '1119075', '952774', '1121276', '433234', '1127893', '1120868', '551040', '1111546', '805321', '929033', '138632', '67262', '883785', '47923', '165633', '554515', '912070', '452431', '451602', '1134138', '966413', '1129237', '1117099', '335594', '1119092', '25129', '301524', '1104031', '324211', '1119058', '835929', '500575', '427578', '1120447', '1105103', '1044797', '156493', '287683', '1116052', '1108939', '1119827', '315637', '1133167', '1133249', '962179', '1134787', '645693', '1047902', '885490']
    ms_marco_eval_systems = ['ms-marco-jlin', 'ms-marco-dai-maxp', 'ms-marco-qa-maxp', 'ms-marco-bm25', 'ms-marco-tf-idf', 'ms-marco-pl2']

    return (parsed['query'] in robust_eval_topics and parsed['system'] in robust_eval_systems) \
           or \
           (parsed['query'] in ms_marco_eval_topics and parsed['system'] in ms_marco_eval_systems)

def __min_max_df(spark_df):
    spark_df.createOrReplaceTempView('TMP_MIN_MAX_DF')
    return spark.sql('SELECT system, query, min(scorediff) as min, max(scorediff) as max '+
                        'FROM TMP_MIN_MAX_DF ' +
                        'GROUP BY system, query'
                       ).toPandas()

def __min_max_normalize(min_value, max_value, value):
    return (value - min_value) / (max_value - min_value)

def __normalizer(spark_df):
    import json
    ret = {}
    for _, i in __min_max_df(spark_df).iterrows():
        i = json.loads(i.to_json())
        sys = str(i['system'])
        query = str(i['query'])
        min_v = float(i['min'])
        max_v = float(i['max'])
        if sys not in ret:
            ret[sys] = {}
        if query not in ret[sys]:
            ret[sys][query] = {}

        ret[sys][query]['min'] = min_v
        ret[sys][query]['max'] = max_v

    return lambda sys, query, value: __min_max_normalize(
        ret[sys][query]['min'],
        ret[sys][query]['max'],
        value
    )

def add_normalized_score_diff_column(spark_df):
    from pyspark.sql.types import DoubleType
    n = __normalizer(spark_df)
    sd_udf = udf(lambda system, query, scorediff: n(system, query, scorediff), DoubleType())
    return spark_df.withColumn('normalized_score_diff', sd_udf('system', 'query', 'scorediff'))

