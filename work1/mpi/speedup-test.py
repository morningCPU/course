"""
MPI 并行性能测试与 Amdahl 分析
任务 B-2：性能测试与 Amdahl 分析

固定问题规模，在 1、2、4 个 MPI 进程下各运行多次取平均，
绘制"实测加速比 vs Amdahl 理论加速比"双折线图。

用法:
  mpirun -np 1 python speedup-test.py [n]

示例:
  mpirun -np 4 python speedup-test.py 10000000
"""

from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def func(x):
    """被积函数: 4 / (1 + x²)"""
    return 4.0 / (1.0 + x**2)

def serial_integration(a, b, n):
    """串行梯形积分"""
    h = (b - a) / n
    result = 0.5 * (func(a) + func(b))
    for i in range(1, n):
        result += func(a + i * h)
    return result * h

def parallel_integration(a, b, n):
    """
    并行梯形积分
    使用 Reduce 汇总各进程的局部和
    """
    h = (b - a) / n

    # 均匀分配任务
    local_n = n // size
    remainder = n % size

    start_idx = rank * local_n + min(rank, remainder)
    end_idx = start_idx + local_n + (1 if rank < remainder else 0)

    local_a = a + start_idx * h
    local_b = a + end_idx * h
    local_n_points = end_idx - start_idx

    # 计算局部积分
    local_sum = 0.5 * (func(local_a) + func(local_b))
    for i in range(1, local_n_points):
        local_sum += func(local_a + i * h)
    local_result = local_sum * h

    # Reduce 汇总到 root 进程
    total_result = comm.reduce(local_result, op=MPI.SUM, root=0)

    return total_result

def run_parallel_benchmark(a, b, n, runs=3):
    """
    并行基准测试
    返回: (平均执行时间, 估算的π值)
    """
    times = []

    for _ in range(runs):
        comm.Barrier()  # 同步所有进程
        start = time.time()
        result = parallel_integration(a, b, n)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    return avg_time, result

def run_serial_benchmark(a, b, n, runs=3):
    """
    串行基准测试（仅 rank 0 执行）
    """
    if rank != 0:
        return None, None

    times = []
    for _ in range(runs):
        start = time.time()
        result = serial_integration(a, b, n)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    return avg_time, result

def calculate_speedup(t_serial, t_parallel):
    """计算加速比"""
    if t_parallel > 0:
        return t_serial / t_parallel
    return 0

def amdahl_speedup(p, f):
    """
    Amdahl 定律计算理论加速比

    参数:
      p: 进程数
      f: 可并行化的比例 (0 <= f <= 1)

    公式: S(p) = 1 / ((1-f) + f/p)
    """
    return 1.0 / ((1.0 - f) + f / p)

def estimate_f(speedup_measured, p):
    """
    根据实测加速比反推 Amdahl 定律的 f 参数

    从 S = 1/((1-f) + f/p) 反推:
    f = p * (S - 1) / (S * (p - 1))
    """
    if speedup_measured <= 0:
        return 0.0

    # 使用多个进程的数据估算 f
    # S = 1/((1-f) + f/p)
    # 1/S = 1 - f + f/p = 1 - f*(1 - 1/p)
    # f*(1 - 1/p) = 1 - 1/S
    # f = (1 - 1/S) / (1 - 1/p)
    f = (1.0 - 1.0/speedup_measured) / (1.0 - 1.0/p)

    # 限制 f 在 [0, 1] 范围内
    return max(0.0, min(1.0, f))

def plot_speedup_comparison(process_counts, measured_speedups, estimated_f, T1):
    """
    绘制实测加速比 vs Amdahl 理论加速比双折线图
    """
    plt.figure(figsize=(8, 5.5), dpi=300)

    # 生成进程数数组
    p_array = np.array(process_counts, dtype=float)

    # 理想线性加速比
    ideal_speedup = p_array

    # Amdahl 理论加速比
    amdahl_speedups = [amdahl_speedup(p, estimated_f) for p in process_counts]

    # 绘制曲线
    plt.plot(process_counts, ideal_speedup, 'g-o', linewidth=2.5,
             label='理想线性加速', linestyle=':')
    plt.plot(process_counts, amdahl_speedups, 'r--s', linewidth=2,
             label=f'Amdahl 理论 (f={estimated_f:.2f})')
    plt.plot(process_counts, measured_speedups, 'b-o', linewidth=2.5,
             label='实测加速比')

    # 图表配置
    plt.xlabel('进程数 $p$', fontsize=11, weight='bold')
    plt.ylabel('加速比 $S$', fontsize=11, weight='bold')
    plt.title('实测加速比 vs Amdahl 理论 vs 理想线性加速', fontsize=13, pad=15, weight='bold')
    plt.xticks(process_counts)
    plt.xlim(0, max(process_counts) + 0.5)
    plt.ylim(0, max(max(measured_speedups), max(process_counts)) + 0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('amdahl_speedup_comparison.png', dpi=300)
    plt.show()

    print("\n图表已保存: amdahl_speedup_comparison.png")

def main():
    # 积分参数
    a, b = 0.0, 1.0
    n = 10000000  # 默认积分区间数

    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            if rank == 0:
                print(f"无效的参数: {sys.argv[1]}，使用默认值: 10000000")
            n = 10000000

    runs = 3  # 每个配置运行3次取平均

    if rank == 0:
        print("=" * 60)
        print("MPI 并行性能测试与 Amdahl 分析")
        print("=" * 60)
        print(f"积分区间: [{a}, {b}]")
        print(f"区间数量: {n:,}")
        print(f"进程数: {size}")
        print(f"每次测试运行 {runs} 次取平均")
        print(f"真实π值: {np.pi:.10f}")
        print("=" * 60)

    # 运行串行基准测试（仅 rank 0）
    if rank == 0:
        print("\n--- 串行基准测试 ---")
        T1, pi_serial = run_serial_benchmark(a, b, n, runs)
        print(f"串行执行时间: {T1:.6f}秒")
        print(f"串行结果: {pi_serial:.10f}")
        print(f"误差: {abs(pi_serial - np.pi):.10f}")

    # 同步
    comm.Barrier()

    # 运行并行基准测试
    print(f"\n--- 并行基准测试 (进程数={size}) ---")
    Tp, pi_parallel = run_parallel_benchmark(a, b, n, runs)

    if rank == 0:
        print(f"并行执行时间: {Tp:.6f}秒")
        print(f"并行结果: {pi_parallel:.10f}")
        print(f"误差: {abs(pi_parallel - np.pi):.10f}")

    # 收集所有进程的结果（用于汇总）
    all_times = comm.gather(Tp, root=0)

    if rank == 0:
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)

        # 打印表格
        print(f"{'进程数 p':<12} {'执行时间 T(p)/s':<18} {'实测加速比 S':<15} {'Amdahl 理论值'}")
        print("-" * 60)

        # 假设串行时间是单进程的时间
        T1 = all_times[0] if all_times[0] else 0
        process_counts = []
        measured_speedups = []

        for i, t in enumerate(all_times):
            if t is not None and t > 0:
                p = i + 1
                s = T1 / t
                process_counts.append(p)
                measured_speedups.append(s)
                print(f"{p:<12} {t:<18.6f} {s:<15.2f}")

        # 估算 Amdahl f 参数
        if len(process_counts) >= 2:
            # 使用最后一个进程的数据估算 f
            estimated_f = estimate_f(measured_speedups[-1], process_counts[-1])

            print("\n" + "=" * 60)
            print("Amdahl 定律分析")
            print("=" * 60)
            print(f"估算的可并行比例 f: {estimated_f:.2%}")
            print(f"估算的串行比例 (1-f): {1-estimated_f:.2%}")

            print("\n各进程数下的 Amdahl 理论加速比:")
            for p, s_measured in zip(process_counts, measured_speedups):
                s_theoretical = amdahl_speedup(p, estimated_f)
                gap = s_measured - s_theoretical
                print(f"  p={p}: 实测={s_measured:.2f}x, 理论={s_theoretical:.2f}x, 差距={gap:+.2f}")

            print("\n实测与理论差距原因分析:")
            print("  1. 通信开销：进程间数据传递（Reduce 操作）需要时间")
            print("  2. 同步开销：Barrier 和 Reduce 操作导致进程等待")
            print("  3. 任务粒度：每个进程处理的区间数太少时，通信占比增大")
            print("  4. 负载不均：n % size != 0 时任务分配不完全均衡")
            print("  5. 启动开销：MPI 初始化和进程间建立连接耗时")

            # 绘制加速比对比图
            plot_speedup_comparison(process_counts, measured_speedups, estimated_f, T1)

        print("\n" + "=" * 60)

    comm.Barrier()

if __name__ == "__main__":
    main()
