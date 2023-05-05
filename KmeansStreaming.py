from pyspark.ml.clustering import StreamingKMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import from_json, col


spark = SparkSession.builder.appName('streamingKMeans').getOrCreate()

brokers = 'localhost:9092'
topic = 'tweets'

kafka_source_options = {
    "kafka.bootstrap.servers": brokers,
    "subscribe": topic,
    "startingOffsets": "earliest"
}

schema = StructType() \
    .add("created", StringType()) \
    .add("text", StringType()) \
    .add("user", StructType() \
         .add("name", StringType()) \
         .add("screen_name", StringType()))

tweets_df = spark \
    .readStream \
    .format("kafka") \
    .options(**kafka_source_options) \
    .load() \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("tweet"))


vectorAssembler = VectorAssembler(inputCols=["text"], outputCol="features")
tweets_df = vectorAssembler.transform(tweets_df)


streamingKMeans = StreamingKMeans(k=2, maxIter=10, featuresCol="features")

model = streamingKMeans.fit(tweets_df)

predictions = model.transform(tweets_df)

query = predictions \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
