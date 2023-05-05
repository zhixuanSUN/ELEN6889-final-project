import tweepy
from kafka import KafkaProducer
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, struct


consumer_key = "UmusoE1i4X5vUK7MCMSrxP2LV"
consumer_secret = "i02QVtUaxAFxrezcKI4xxd5ujIpmSXvPozYheGr27u1ZHPvy20"
access_token = "1653415138567811073-l7F28c7lAmY4mo4u0IW0h11KEPU4Wq"
access_secret = "9MLdpO6ShlJ54OrcHCfAaf9KJlyg57CBcBRENx4hXxdkT"

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: json.dumps(x).encode('utf-8'))

class TSListener(tweepy.StreamListener):

    def on_data(self, data):
        tweet = json.loads(data)
        producer.send('tweets', tweet)
        return True

    def on_error(self, status_code):
        if status_code == 420:
            return False
        else:
            return True

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
listener = TSListener()
stream = tweepy.Stream(auth=auth, listener=listener)
stream.filter(track=['#hashtag'])

spark = SparkSession.builder.appName('twitterStream').getOrCreate()

brokers = 'localhost:9092'
topic = 'tweets'


kafka_options = {
    "kafka.bootstrap.servers": brokers,
    "subscribe": topic,
    "startingOffsets": "earliest"
}

tweets_df = spark \
    .readStream \
    .format("kafka") \
    .options(**kafka_options) \
    .load() \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("tweet"))

schema = StructType() \
    .add("created", StringType()) \
    .add("text", StringType()) \
    .add("user", StructType() \
         .add("name", StringType()) \
         .add("screen_name", StringType()))

tweets_df = tweets_df \
    .select(col("tweet.created").alias("created_at"),
            col("tweet.text").alias("text"),
            col("tweet.user.name").alias("user_name"),
            col("tweet.user.screen_name").alias("user_screen_name"))
query = tweets_df \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .start()
query.awaitTermination()
