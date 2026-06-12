"""
性能对比程序：Pandas vs PySpark
任务 A-3：性能对比与 Amdahl 分析

对比同一查询在 Pandas（单机）和 PySpark（1个/2个 Executor）下的执行时间，
并结合 Amdahl 定律分析加速比未达到线性的原因。

用法:
  spark-submit performance-comparison.py s3a://<BUCKET>/movies.csv
"""

import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, year
import matplotlib.pyplot as plt
import sys

def create_spark_session(executor_instances=1, executor_cores=1, executor_memory="1g"):
    """创建配置不同Executor数量的 SparkSession"""
    return SparkSession.builder \
        .appName(f"PerformanceComparison_Ex{ executor_instances}") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.executor.instances", str(executor_instances)) \
        .config("spark.executor.cores", str(executor_cores)) \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

def pandas_query(df):
    """
    Pandas 版本查询：
    按年份统计电影数量和平均评分
    """
    # 模拟与 Spark 版本相同的查询逻辑
    result = df.groupby('year').agg(
        count='movie_id',
        avg_rating=('rating', 'mean')
    ).reset_index()

    # 筛选条件
    result = result[result['year'] >= 2000]

    return result

def spark_query(spark, path):
    """
    PySpark 版本查询：
    按年份统计电影数量和平均评分
    """
    df = spark.read.csv(path, header=False, schema="""
        movie_id STRING,
        title STRING,
        genres STRING,
        year INT,
        rating DOUBLE,
        votes INT
    """)

    result = df.groupBy("year") \
        .agg(
            count("*").alias("count"),
            avg("rating").alias("avg_rating")
        ) \
        .filter(col("year") >= 2000) \
        .orderBy("year")

    return result.toPandas()

def run_pandas_benchmark(path, runs=3):
    """
    运行 Pandas 基准测试
    """
    print("\n" + "=" * 60)
    print("Pandas (单机) 基准测试")
    print("=" * 60)

    # 读取数据
    pandas_df = pd.read_csv(
        path,
        header=None,
        names=['movie_id', 'title', 'genres', 'year', 'rating', 'votes']
    )

    times = []
    for i in range(runs):
        start = time.time()
        result = pandas_query(pandas_df)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  运行 {i+1}: {elapsed:.4f}秒")

    avg_time = np.mean(times)
    print(f"\n平均执行时间: {avg_time:.4f}秒")
    return avg_time

def run_spark_benchmark(path, executor_instances, runs=3):
    """
    运行 PySpark 基准测试
    """
    print("\n" + "=" * 60)
    print(f"PySpark (Executor={executor_instances}) 基准测试")
    print("=" * 60)

    spark = create_spark_session(executor_instances=executor_instances)

    times = []
    for i in range(runs):
        # 清除缓存确保公平
        spark.catalog.clearCache()

        start = time.time()
        result = spark_query(spark, path)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  运行 {i+1}: {elapsed:.4f}秒")

    avg_time = np.mean(times)
    print(f"\n平均执行时间: {avg_time:.4f}秒")

    spark.stop()
    return avg_time

def plot_comparison(pandas_time, spark1_time, spark2_time):
    """
    绘制性能对比图
    """
    print("\n" + "=" * 60)
    print("绘制性能对比图...")
    print("=" * 60)

    labels = ['Pandas\n(单机)', 'PySpark\n(1 Executor)', 'PySpark\n(2 Executor)']
    times = [pandas_time, spark1_time, spark2_time]

    # 计算加速比
    speedup_1 = pandas_time / spark1_time if spark1_time > 0 else 0
    speedup_2 = pandas_time / spark2_time if spark2_time > 0 else 0
    speedups = [1.0, speedup_1, speedup_2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 执行时间对比
    bars1 = ax1.bar(labels, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax1.set_ylabel('执行时间 (秒)', fontsize=11)
    ax1.set_title('执行时间对比', fontsize=12, weight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱状图上添加数值标签
    for bar, t in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{t:.3f}s', ha='center', va='bottom', fontsize=10)

    # 加速比对比
    speedup_labels = ['基准(1x)', f'{speedup_1:.2f}x', f'{speedup_2:.2f}x']
    bars2 = ax2.bar(labels, speedups, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax2.set_ylabel('加速比', fontsize=11)
    ax2.set_title('加速比对比', fontsize=12, weight='bold')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=2, color='green', linestyle=':', alpha=0.5, label='理想2x')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.legend()

    for bar, s, label in zip(bars2, speedups, speedup_labels):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                label, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("图表已保存: performance_comparison.png")

def amdahl_analysis(speedup_1, speedup_2, p1=1, p2=2):
    """
    Amdahl 定律分析

    加速比公式: S(p) = 1 / ((1-f) + f/p)

    其中 f 是可并行化的比例

    根据实测数据反推 f:
    - 如果加速比远低于进程数，说明串行部分占比较大
    - 原因可能包括：任务粒度太小、数据序列化开销、进程间通信开销等
    """
    print("\n" + "=" * 60)
    print("Amdahl 定律分析")
    print("=" * 60)

    # 根据 2 个 Executor 的加速比反推 f
    # speedup_2 = 1 / ((1-f) + f/2)
    # => 1/speedup_2 = (1-f) + f/2
    # => 1/speedup_2 = 1 - f + f/2
    # => 1/speedup_2 = 1 - f/2
    # => f/2 = 1 - 1/speedup_2
    # => f = 2 * (1 - 1/speedup_2)

    if speedup_2 > 0:
        f = 2 * (1 - 1/speedup_2)
        f = max(0, min(1, f))  # 限制在 [0, 1] 范围内

        print(f"\n根据实测加速比 {speedup_2:.2f}x (2 Executor):")
        print(f"  估算可并行比例 f = {f:.2%}")
        print(f"  估算串行比例 (1-f) = {1-f:.2%}")

        # 计算理论加速比
        theoretical_speedup_1 = 1 / ((1-f) + f/p1)
        theoretical_speedup_2 = 1 / ((1-f) + f/p2)

        print(f"\n理论加速比 (基于估算的 f):")
        print(f"  1 Executor: {theoretical_speedup_1:.2f}x")
        print(f"  2 Executor: {theoretical_speedup_2:.2f}x")

        print(f"\n实测 vs 理论差距分析:")
        print(f"  1 Executor: 实测 {speedup_1:.2f}x vs 理论 {theoretical_speedup_1:.2f}x")
        print(f"  2 Executor: 实测 {speedup_2:.2f}x vs 理论 {theoretical_speedup_2:.2f}x")

        reasons = []
        if speedup_2 < theoretical_speedup_2 * 0.8:
            reasons.append("1. 通信开销：Executor 间数据传输耗时")
        if speedup_2 < p2 * 0.5:
            reasons.append("2. 序列化开销：数据在进程间传递需要序列化/反序列化")
        if speedup_2 < p2 * 0.3:
            reasons.append("3. 任务粒度：每个 Executor 处理的数据量太小")
        reasons.append("4. 启动开销：Spark 作业调度和 Executor 启动耗时")

        print(f"\n可能的原因:")
        for reason in reasons:
            print(f"  {reason}")

        return f
    return None

def main():
    if len(sys.argv) < 2:
        print("用法: spark-submit performance-comparison.py <input_csv_path>")
        print("示例: spark-submit performance-comparison.py s3a://bucket/movies.csv")
        sys.exit(1)

    input_path = sys.argv[1]

    print("=" * 60)
    print("性能对比：Pandas vs PySpark")
    print("=" * 60)

    # 运行基准测试
    pandas_time = run_pandas_benchmark(input_path, runs=3)
    spark1_time = run_spark_benchmark(input_path, executor_instances=1, runs=3)
    spark2_time = run_spark_benchmark(input_path, executor_instances=2, runs=3)

    # 绘制对比图
    plot_comparison(pandas_time, spark1_time, spark2_time)

    # Amdahl 分析
    speedup_1 = pandas_time / spark1_time if spark1_time > 0 else 0
    speedup_2 = pandas_time / spark2_time if spark2_time > 0 else 0
    amdahl_analysis(speedup_1, speedup_2)

    print("\n" + "=" * 60)
    print("性能对比完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
