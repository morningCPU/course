from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, mean, stddev, min as spark_min, max as spark_max
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import sys

def create_spark_session():
    return SparkSession.builder \
        .appName("DataAnalysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

def load_movie_data(spark, path):
    schema = StructType([
        StructField("movie_id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True),
        StructField("year", IntegerType(), True),
        StructField("rating", DoubleType(), True),
        StructField("votes", IntegerType(), True)
    ])
    return spark.read.csv(path, header=False, schema=schema)

def data_cleaning(df):
    print("=" * 60)
    print("原始数据信息:")
    print("=" * 60)
    print(f"总行数: {df.count()}")
    
    print("\n数据Schema:")
    df.printSchema()
    
    print("\n前5行数据:")
    df.show(5)
    
    print("\n各字段缺失值统计:")
    total = df.count()
    for field in df.columns:
        null_count = df.filter(col(field).isNull()).count()
        missing_pct = (null_count / total) * 100 if total > 0 else 0
        print(f"{field}: {null_count} ({missing_pct:.2f}%)")
    
    df_cleaned = df.dropna(subset=["movie_id", "title"])
    
    df_cleaned = df_cleaned.withColumn(
        "year",
        when(col("year").isNull(), 2000).otherwise(col("year"))
    )
    
    df_cleaned = df_cleaned.withColumn(
        "rating",
        when(col("rating").isNull(), 0.0).otherwise(col("rating"))
    )
    
    df_cleaned = df_cleaned.withColumn(
        "votes",
        when(col("votes").isNull(), 0).otherwise(col("votes"))
    )
    
    print("\n" + "=" * 60)
    print("清洗后数据信息:")
    print("=" * 60)
    print(f"总行数: {df_cleaned.count()}")
    
    return df_cleaned

def statistical_analysis(df):
    print("\n" + "=" * 60)
    print("统计分析:")
    print("=" * 60)
    
    print("\n评分统计:")
    df.select(
        mean("rating").alias("平均评分"),
        stddev("rating").alias("评分标准差"),
        spark_min("rating").alias("最低评分"),
        spark_max("rating").alias("最高评分")
    ).show()
    
    print("\n各年份电影数量 (Top 10):")
    df.groupBy("year") \
      .count() \
      .orderBy(col("count").desc()) \
      .show(10)
    
    print("\n各评分区间电影数量:")
    df.withColumn(
        "rating_range",
        when(col("rating") >= 9.0, "9-10")
        .when(col("rating") >= 8.0, "8-9")
        .when(col("rating") >= 7.0, "7-8")
        .when(col("rating") >= 6.0, "6-7")
        .when(col("rating") >= 5.0, "5-6")
        .otherwise("below 5")
    ).groupBy("rating_range").count().orderBy("rating_range").show()
    
    print("\n高评分电影 (评分 >= 8.0):")
    high_rated = df.filter(col("rating") >= 8.0).orderBy(col("rating").desc())
    high_rated.show(10)
    
    print("\n按年代分组统计:")
    df.withColumn(
        "decade",
        (col("year") / 10).cast("int") * 10
    ).groupBy("decade").agg(
        count("*").alias("movie_count"),
        mean("rating").alias("avg_rating")
    ).orderBy("decade").show()

def save_results(df, output_path):
    df.write.mode("overwrite").parquet(output_path)
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "s3a://<BUCKET>/movies.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "s3a://<BUCKET>/output"
    
    spark = create_spark_session()
    
    try:
        df = load_movie_data(spark, input_path)
        df_cleaned = data_cleaning(df)
        statistical_analysis(df_cleaned)
        save_results(df_cleaned, output_path)
    finally:
        spark.stop()
