"""
串行矩阵乘法程序
任务 B-1/B-2：与 MPI 并行版本对比

提供两种实现：
1. 基础三重循环 O(n³)
2. NumPy 优化版本

用法:
  python serial-matrix.py [n]

示例:
  python serial-matrix.py 800
"""

import numpy as np
import time
import sys

def matrix_multiply_naive(A, B):
    """
    基础串行矩阵乘法 - 三重循环 O(n³)

    A: n×n 矩阵
    B: n×n 矩阵
    返回: C = A × B
    """
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

    return C

def matrix_multiply_numpy(A, B):
    """
    NumPy 优化的矩阵乘法

    使用 BLAS 底层实现，比纯 Python 快很多
    """
    return np.dot(A, B)

def matrix_multiply_numpy_batch(A, B, block_size=64):
    """
    分块矩阵乘法 (Cache 友好)

    将矩阵分成小块以提高 Cache 命中率
    """
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)

    # 确保 A, B 是连续内存布局
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)

    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                # 分块乘法
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, n)
                k_end = min(k + block_size, n)

                C[i:i_end, j:j_end] += np.dot(
                    A[i:i_end, k:k_end],
                    B[k:k_end, j:j_end]
                )

    return C

def verify_result(C1, C2, name1="算法1", name2="算法2"):
    """验证两个结果是否一致"""
    if np.allclose(C1, C2):
        print(f"✓ {name1} 和 {name2} 结果一致")
        return True
    else:
        diff = np.max(np.abs(C1 - C2))
        print(f"✗ {name1} 和 {name2} 结果不一致，最大差异: {diff}")
        return False

def run_benchmark(n, runs=3):
    """
    运行基准测试
    """
    print("=" * 60)
    print(f"串行矩阵乘法基准测试 (矩阵大小: {n}×{n})")
    print("=" * 60)

    # 生成随机矩阵
    print("\n生成随机矩阵...")
    np.random.seed(42)
    A = np.random.random((n, n))
    B = np.random.random((n, n))

    # 1. 基础三重循环版本
    print(f"\n--- 三重循环版本 O(n³) ---")
    times_naive = []
    for i in range(runs):
        start = time.time()
        C_naive = matrix_multiply_naive(A.copy(), B.copy())
        elapsed = time.time() - start
        times_naive.append(elapsed)
        print(f"  运行 {i+1}: {elapsed:.4f}秒")

    avg_naive = np.mean(times_naive)
    print(f"  平均时间: {avg_naive:.4f}秒")

    # 2. NumPy 版本
    print(f"\n--- NumPy 优化版本 ---")
    times_numpy = []
    for i in range(runs):
        start = time.time()
        C_numpy = matrix_multiply_numpy(A.copy(), B.copy())
        elapsed = time.time() - start
        times_numpy.append(elapsed)
        print(f"  运行 {i+1}: {elapsed:.4f}秒")

    avg_numpy = np.mean(times_numpy)
    print(f"  平均时间: {avg_numpy:.4f}秒")

    # 3. 分块版本（仅大矩阵时）
    if n >= 200:
        print(f"\n--- 分块优化版本 ---")
        times_block = []
        for i in range(runs):
            start = time.time()
            C_block = matrix_multiply_numpy_batch(A.copy(), B.copy())
            elapsed = time.time() - start
            times_block.append(elapsed)
            print(f"  运行 {i+1}: {elapsed:.4f}秒")

        avg_block = np.mean(times_block)
        print(f"  平均时间: {avg_block:.4f}秒")

    # 验证结果一致性
    print("\n--- 结果验证 ---")
    verify_result(C_naive, C_numpy, "三重循环", "NumPy")
    if n >= 200:
        verify_result(C_naive, C_block, "三重循环", "分块优化")

    # 性能对比
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)
    print(f"矩阵大小: {n}×{n}")
    print(f"三重循环: {avg_naive:.4f}秒 (基准)")
    print(f"NumPy:    {avg_numpy:.4f}秒 (加速比: {avg_naive/avg_numpy:.2f}x)")
    if n >= 200:
        print(f"分块优化:  {avg_block:.4f}秒 (加速比: {avg_naive/avg_block:.2f}x)")

    return {
        'naive': avg_naive,
        'numpy': avg_numpy,
        'block': avg_block if n >= 200 else None,
        'n': n
    }

def main():
    # 默认矩阵大小
    n = 200

    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"无效的矩阵大小: {sys.argv[1]}")
            print("使用默认值: 200")
            n = 200

    if n < 10:
        print("矩阵大小太小，无法有效测试")
        print("请使用大于 10 的矩阵大小")
        return

    result = run_benchmark(n, runs=3)

    print("\n" + "=" * 60)
    print("基准测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
