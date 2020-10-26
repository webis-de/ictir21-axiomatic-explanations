#!/bin/bash -e

PARALLELISM=160*16

spark-submit \
	--py-files experiments_common.py \
	--executor-cores 16 \
	--conf spark.default.parallelism=${PARALLELISM} \
	--conf spark.driver.maxResultSize=2G \
	--num-executors 160 \
	--executor-memory 64G \
	--driver-memory 32G \
	run-global-analysis.py

