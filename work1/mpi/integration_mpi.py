from mpi4py import MPI
import numpy as np
import time

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
    使用Scatter分发任务，Reduce汇总结果
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    h = (b - a) / n
    
    local_n = n // size
    remainder = n % size
    
    start_idx = rank * local_n + min(rank, remainder)
    end_idx = start_idx + local_n + (1 if rank < remainder else 0)
    
    local_a = a + start_idx * h
    local_b = a + end_idx * h
    local_n_points = end_idx - start_idx
    
    local_sum = 0.5 * (func(local_a) + func(local_b))
    for i in range(1, local_n_points):
        local_sum += func(local_a + i * h)
    local_result = local_sum * h
    
    total_result = comm.reduce(local_result, op=MPI.SUM, root=0)
    
    return total_result

def parallel_integration_nonblocking(a, b, n):
    """
    并行梯形积分 - 非阻塞通信版本
    计算与通信重叠
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    h = (b - a) / n
    
    local_n = n // size
    remainder = n % size
    
    start_idx = rank * local_n + min(rank, remainder)
    end_idx = start_idx + local_n + (1 if rank < remainder else 0)
    
    local_a = a + start_idx * h
    local_b = a + end_idx * h
    local_n_points = end_idx - start_idx
    
    local_sum = 0.5 * (func(local_a) + func(local_b))
    for i in range(1, local_n_points):
        local_sum += func(local_a + i * h)
    local_result = local_sum * h
    
    request = comm.Ireduce(local_result, op=MPI.SUM, root=0)
    
    request.Wait()
    
    total_result = 0
    if rank == 0:
        recv_buf = np.zeros(1)
        comm.Reduce(local_result, recv_buf, op=MPI.SUM, root=0)
        total_result = recv_buf[0]
    
    return total_result

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=" * 60)
        print("并行数值积分 (梯形法则)")
        print("=" * 60)
    
    a, b = 0.0, 1.0
    n = 10000000
    
    if rank == 0:
        print(f"积分区间: [{a}, {b}]")
        print(f"区间数量: {n:,}")
        print(f"进程数: {size}")
        print(f"真实π值: {np.pi:.10f}")
        
        print("\n--- 串行计算 ---")
        start = time.time()
        result_serial = serial_integration(a, b, n)
        serial_time = time.time() - start
        print(f"结果: {result_serial:.10f}")
        print(f"误差: {abs(result_serial - np.pi):.10f}")
        print(f"耗时: {serial_time:.4f}秒")
        
        print("\n--- 并行计算 (阻塞通信) ---")
    
    comm.Barrier()
    
    start = time.time()
    result_parallel = parallel_integration(a, b, n)
    parallel_time = time.time() - start
    
    if rank == 0:
        print(f"结果: {result_parallel:.10f}")
        print(f"误差: {abs(result_parallel - np.pi):.10f}")
        print(f"耗时: {parallel_time:.4f}秒")
        print(f"加速比: {serial_time / parallel_time:.2f}x")
        
        print("\n--- 并行计算 (非阻塞通信) ---")
        
        start = time.time()
        result_nonblocking = parallel_integration_nonblocking(a, b, n)
        nonblocking_time = time.time() - start
        
        print(f"结果: {result_nonblocking:.10f}")
        print(f"误差: {abs(result_nonblocking - np.pi):.10f}")
        print(f"耗时: {nonblocking_time:.4f}秒")
        print(f"加速比: {serial_time / nonblocking_time:.2f}x")
        
        print("\n" + "=" * 60)
        print("性能对比:")
        print(f"  串行耗时:       {serial_time:.4f}秒")
        print(f"  并行(阻塞):     {parallel_time:.4f}秒")
        print(f"  并行(非阻塞):   {nonblocking_time:.4f}秒")
        print("=" * 60)
    
    comm.Barrier()
