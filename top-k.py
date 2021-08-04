from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import udf
from pyspark.sql.types import *

# Prepare the Spark context
conf = SparkConf().setMaster("local") \
                  .setAppName("Top-K") \
                  .set("spark.executor.memory", "4g") \
                  .set("spark.executor.instances", 1)
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

DF1=spark.read.csv("hdfs://localhost:9000/user/AB_US_2020.csv",header=True)
DF1.createOrReplaceTempView("airbnb")
DF1.show()
DF2= spark.sql("""select name,sum(number_of_reviews) as Top_reviews from airbnb group by name order by sum(number_of_reviews) desc limit 10""")
DF2.show()

#total_reviews=DF1.groupBy().agg(F.sum("number_of_reviews")).collect()[0][0]
total_reviews=DF1.select(F.sum('number_of_reviews')).head()[0]

def func_divide(x,y):
  return x*100.0/float(y)

divide_func_sp = F.udf(func_divide,FloatType())

DF2 = DF2.withColumn('popularity_index',divide_func_sp('Top_reviews','total_reviews'))
DF2.orderBy('popularity_index', ascending=False).show(10)
DF3 = DF2.select('name','popularity_index')
DF3.show()
