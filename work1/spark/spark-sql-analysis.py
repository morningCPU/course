"""
Spark SQL 统计分析程序
任务 A-2：完成至少 4 个统计查询，包含：
  - GROUP BY 聚合
  - ORDER BY Top-N
  - 时间维度趋势分析（按年/月）
  - JOIN 操作或窗口函数

用法:
  spark-submit spark-sql-analysis.py s3a://<BUCKET>/movies.csv
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum, avg, min as spark_min, max as spark_max,
    year, month, row_number, rank, dense_rank,
    when, concat_ws, split,explode
)
from pyspark.sql.window import Window
import sys

def create_spark_session():
    """创建 SparkSession"""
    return SparkSession.builder \
        .appName("SparkSQLAnalysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

def load_movie_data(spark, path):
    """加载电影数据"""
    schema = """
        movie_id STRING,
        title STRING,
        genres STRING,
        year INT,
        rating DOUBLE,
        votes INT
    """
    return spark.read.csv(path, header=False, schema=schema)

def query1_group_by_aggregation(df):
    """
    查询1: GROUP BY 聚合
    统计每个年代的的电影数量和平均评分
    """
    print("\n" + "=" * 60)
    print("查询1: GROUP BY 聚合 - 各年代电影数量和平均评分")
    print("=" * 60)

    result = df.groupBy("year") \
        .agg(
            count("*").alias("movie_count"),
            avg("rating").alias("avg_rating"),
            sum("votes").alias("total_votes")
        ) \
        .orderBy("year") \
        .filter(col("year") >= 1990)  # 只看1990年以后的

    result.show(15)

    print("分析: 通过 GROUP BY 按年份聚合，可以看到不同年代的电影产量和受欢迎程度变化趋势。")
    return result

def query2_order_by_top_n(df):
    """
    查询2: ORDER BY Top-N
    找出评分最高的前10部电影（评分人数超过1000）
    """
    print("\n" + "=" * 60)
    print("查询2: ORDER BY Top-N - 评分最高的Top 10电影")
    print("=" * 60)

    window_spec = Window.orderBy(col("rating").desc())

    result = df.filter(col("votes") >= 1000) \
        .withColumn("rank", row_number().over(window_spec)) \
        .filter(col("rank") <= 10) \
        .select("rank", "title", "year", "rating", "votes")

    result.show(10, truncate=False)

    print("分析: 使用 ORDER BY 对评分降序排列，筛选出评分人数超过1000的高质量电影榜单。")
    return result

def query3_time_trend_analysis(df):
    """
    查询3: 时间维度趋势分析（按年）
    分析近10年每年电影的平均评分变化趋势
    """
    print("\n" + "=" * 60)
    print("查询3: 时间维度趋势分析 - 近10年平均评分趋势")
    print("=" * 60)

    from pyspark.sql.functions import lit

    current_year = 2026
    recent_years = current_year - 10

    result = df.filter(col("year") >= recent_years) \
        .groupBy("year") \
        .agg(
            count("*").alias("movie_count"),
            avg("rating").alias("avg_rating"),
            avg("votes").alias("avg_votes")
        ) \
        .orderBy("year")

    result.show(10)

    print("分析: 通过按年聚合可以看到近年电影评分的变化趋势，评分人数反映电影热度。")

    # 继续分析：按月份统计（如果有日期字段的话，这里模拟）
    print("\n补充分析: 电影类型分布统计")
    # 将 genres 拆分为多行
    movies_with_genre = df.withColumn("genre", explode(split(col("genres"), "\\|")))

    genre_stats = movies_with_genre.groupBy("genre") \
        .agg(
            count("*").alias("movie_count"),
            avg("rating").alias("avg_rating")
        ) \
        .orderBy(col("movie_count").desc())

    genre_stats.show(10)

    return result

def query4_window_function(df):
    """
    查询4: 窗口函数
    为每部电影在其所在年代内计算评分排名
    """
    print("\n" + "=" * 60)
    print("查询4: 窗口函数 - 各年代内电影评分排名")
    print("=" * 60)

    # 定义按年份分区的窗口，按评分降序
    window_spec = Window.partitionBy("year").orderBy(col("rating").desc())

    result = df.filter(col("year") >= 2015) \
        .withColumn("rank_in_year", rank().over(window_spec)) \
        .withColumn("dense_rank_in_year", dense_rank().over(window_spec)) \
        .filter(col("dense_rank_in_year") <= 5) \
        .select("year", "rank_in_year", "title", "rating", "votes") \
        .orderBy("year", "rank_in_year")

    result.show(20, truncate=False)

    print("分析: 使用窗口函数 RANK() 对每年电影按评分排名，可识别每年在该年代内的优秀作品。")
    return result

def query5_join_operation(df):
    """
    查询5: JOIN 操作
    找出同时获得高评分和高投票数的"热门佳作"
    """
    print("\n" + "=" * 60)
    print("查询5: JOIN 操作 - 热门佳作筛选")
    print("=" * 60)

    # 创建高评分 DataFrame (评分 >= 8.0)
    high_rated = df.filter(col("rating") >= 8.0) \
        .select(col("movie_id").alias("hr_movie_id"), "title", "rating")

    # 创建高投票 DataFrame (投票数 >= 10000)
    high_votes = df.filter(col("votes") >= 10000) \
        .select(col("movie_id").alias("hv_movie_id"), "votes")

    # JOIN 两个条件
    result = high_rated.join(
        high_votes,
        high_rated.hr_movie_id == high_votes.hv_movie_id,
        "inner"
    ) \
    .select("title", "rating", "votes") \
    .orderBy(col("rating").desc(), col("votes").desc())

    result.show(15, truncate=False)

    print("分析: 通过 INNER JOIN 找出同时满足高评分和高投票数的电影，这类电影是公认的'热门佳作'。")
    return result

def main():
    if len(sys.argv) < 2:
        print("用法: spark-submit spark-sql-analysis.py <input_csv_path>")
        print("示例: spark-submit spark-sql-analysis.py s3a://bucket/movies.csv")
        sys.exit(1)

    input_path = sys.argv[1]

    spark = create_spark_session()

    try:
        print("=" * 60)
        print("Spark SQL 统计分析程序")
        print(f"输入数据: {input_path}")
        print("=" * 60)

        # 加载数据
        df = load_movie_data(spark, input_path)

        print(f"\n数据总行数: {df.count()}")
        print("\n原始数据 Schema:")
        df.printSchema()
        df.show(5)

        # 执行5个查询
        query1_group_by_aggregation(df)
        query2_order_by_top_n(df)
        query3_time_trend_analysis(df)
        query4_window_function(df)
        query5_join_operation(df)

        print("\n" + "=" * 60)
        print("所有查询执行完成")
        print("=" * 60)

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
