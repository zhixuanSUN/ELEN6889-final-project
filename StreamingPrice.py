from pyspark.streaming.kafka import KafkaUtils
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import avg

sc = StreamingContext(spark.sparkContext, 60)

kafka_params = {"bootstrap.servers": "kafka:9092"}
stream = KafkaUtils.createDirectStream(sc, ["stock-market"])

def save_to_file(record):
    fields = record.split(",")
    t = fields[0]
    price = float(fields[4])
    return (t, price)

data = stream.map(lambda x: save_to_file(x[1])).filter(lambda x: x[0] in ["AAPL", "GOOG", "AMZN"])

windowed_data = data.window(900, 5).groupByKey().mapValues(lambda x: sum(x) / len(x))

windowed_data.pprint()

sc.start()
sc.awaitTermination()
