os.environ['PYSPARK_PYTHON'] = 'python3.6'
os.environ['HADOOP_CONF_DIR'] = os.path.join(os.environ['HOME'], 'aitools4-aq-cluster-computing/conf/hadoop')

sparkconf = SparkConf()


(sparkconf.setMaster('yarn')
  .set('spark.driver.memory', '16g')
  .set('spark.executor.memory', '64g')
 .set('spark.executor.cores', '16')
 .set('spark.executor.instances', '128')
)
    

global spark
spark = SparkSession.builder.config(conf=sparkconf).getOrCreate()


