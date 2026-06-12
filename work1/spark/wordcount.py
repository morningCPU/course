"""
WordCount 示例程序 - PySpark 入门验证
用于验证 Spark on K8s 环境部署是否成功

用法:
  spark-submit --master k8s://https://<K8S_API> wordcount.py s3a://<BUCKET>/input.txt

功能:
  统计文本文件中每个单词出现的次数
"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import sys

def create_spark_session(app_name="WordCount"):
    """创建 SparkSession"""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

def word_count(sc, text_file):
    """
    WordCount 核心逻辑
    1. 读取文本文件
    2. 将每行拆分为单词
    3. 统计每个单词出现次数
    4. 按次数降序排列
    """
    # 读取文件，每行作为一个元素
    lines = sc.textFile(text_file)

    # 转换：将每行文本拆分为单词列表，再压平为单个单词流
    # "hello world" -> ["hello", "world"] -> "hello", "world"
    words = lines.flatMap(lambda line: line.split())

    # 为每个单词赋值为 1，然后按单词聚合统计次数
    # ("hello", 1), ("world", 1) -> ("hello", 1), ("hello", 1) -> ("hello", 2)
    word_counts = words.countByValue()

    return word_counts

def main():
    # 从命令行参数获取输入文件路径
    # 默认使用 SparkContext 的方式
    if len(sys.argv) < 2:
        print("用法: spark-submit wordcount.py <input_file>")
        print("示例: spark-submit wordcount.py s3a://bucket/input.txt")
        sys.exit(1)

    input_file = sys.argv[1]

    # 创建 Spark 配置
    conf = SparkConf() \
        .setAppName("WordCount") \
        .setMaster("local[*]")  # 本地模式运行

    # 创建 SparkContext
    sc = SparkContext(conf=conf)

    try:
        print("=" * 60)
        print(f"WordCount 程序开始执行")
        print(f"输入文件: {input_file}")
        print("=" * 60)

        # 执行 WordCount
        word_counts = word_count(sc, input_file)

        # 转换为列表并按次数降序排列
        sorted_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # 打印结果 (Top 20)
        print("\nTop 20 单词统计:")
        print("-" * 40)
        for word, count in sorted_counts[:20]:
            print(f"{word}: {count}")

        # 保存结果到文件 (如果指定了输出路径)
        if len(sys.argv) >= 3:
            output_file = sys.argv[2]
            # 将结果转换为 RDD 格式保存
            result_rdd = sc.parallelize(sorted_counts)
            result_rdd.saveAsTextFile(output_file)
            print(f"\n结果已保存到: {output_file}")

        print("\n" + "=" * 60)
        print("WordCount 程序执行完成")
        print("=" * 60)

    finally:
        # 关闭 SparkContext
        sc.stop()

if __name__ == "__main__":
    main()
