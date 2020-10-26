from experiments_common import *

col_predictors = ['ax_AND', 'ax_DIV', 'ax_LB1', 'ax_LNC1', 'ax_M_AND', 'ax_M_TDC', 'ax_PROX1', 'ax_PROX2', 'ax_PROX3', 'ax_PROX4', 'ax_PROX5', 'ax_REG', 'ax_STMC1', 'ax_STMC1_f', 'ax_STMC1_fr', 'ax_STMC2', 'ax_STMC2_f', 'ax_STMC2_fr', 'ax_TFC1', 'ax_TFC3', 'ax_TF_LNC']
col_response = 'concordant'

def spark_context():
	return SparkSession.builder.config(conf=sparkconf).getOrCreate()

if __name__ == '__main__':
	dataset_path = 'axexp/prefs/evaluation-topics-ms-marco-top1000.jsonl'
	dataset_sparkdf = spark.read.json(dataset_path)
	dataset_sparkdf.printSchema()

