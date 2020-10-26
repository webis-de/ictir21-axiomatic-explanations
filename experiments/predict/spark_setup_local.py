os.environ['PYSPARK_PYTHON'] = 'python3.6'

sparkconf = SparkConf()


if 'gammaweb' in socket.gethostname():
    (sparkconf.setMaster('local[32]')
      .set('spark.driver.memory', '256g')
      .set('spark.executor.memory', '512g'))
    

global spark
if 'spark' in globals() and hasattr(spark, 'stop'):
    spark.stop()
    SparkSession._instantiatedContext = None
spark = SparkSession.builder.config(conf=sparkconf).getOrCreate()


