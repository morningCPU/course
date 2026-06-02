from mpi4py import MPI
import numpy as np
import time

def compute_pi_monte_carlo(n):
    """
    使用蒙特卡洛方法并行计算π值
    每个进程独立生成随机点，统计落在单位圆内的点数量
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    np.random.seed(42 + rank)
    
    local_n = n // size
    if rank == 0:
        local_n += n % size
    
    x = np.random.random(local_n)
    y = np.random.random(local_n)
    
    inside = np.sum(x**2 + y**2 <= 1.0)
    
    total_inside = comm.reduce(inside, op=MPI.SUM, root=0)
    
    if rank == 0:
        pi = 4.0 * total_inside / n
        return pi
    return None

def compute_pi_numerical_integration(n):
    """
    使用梯形法则并行计算π值
    ∫₀¹ 4/(1+x²) dx = π
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    h = 1.0 / n
    
    start = rank * (n // size)
    end = start + (n // size)
    if rank == size - 1:
        end += n % size
    
    local_sum = 0.0
    for i in range(start, end):
        x = h * (i + 0.5)
        local_sum += 4.0 / (1.0 + x**2)
    
    local_sum *= h
    
    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    
    if rank == 0:
        return total_sum
    return None

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 60)
        print("MPI π 计算程序")
        print("=" * 60)
    
    n = 10000000
    
    if rank == 0:
        print(f"\n使用蒙特卡洛方法计算π (n={n}):")
    start_time = time.time()
    pi_mc = compute_pi_monte_carlo(n)
    elapsed = time.time() - start_time
    
    if rank == 0:
        print(f"  计算结果: {pi_mc:.10f}")
        print(f"  真实值:   {np.pi:.10f}")
        print(f"  误差:     {abs(pi_mc - np.pi):.10f}")
        print(f"  耗时:     {elapsed:.4f}秒")
    
    if rank == 0:
        print(f"\n使用数值积分方法计算π (n={n}):")
    start_time = time.time()
    pi_int = compute_pi_numerical_integration(n)
    elapsed = time.time() - start_time
    
    if rank == 0:
        print(f"  计算结果: {pi_int:.10f}")
        print(f"  真实值:   {np.pi:.10f}")
        print(f"  误差:     {abs(pi_int - np.pi):.10f}")
        print(f"  耗时:     {elapsed:.4f}秒")
        
        print("\n" + "=" * 60)
