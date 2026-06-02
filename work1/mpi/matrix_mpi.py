from mpi4py import MPI
import numpy as np
import time

def parallel_matrix_multiply(A, B, n):
    """
    并行矩阵乘法 (Fox算法)
    A: n×n 矩阵
    B: n×n 矩阵
    返回: C = A × B
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    sqrt_size = int(np.sqrt(size))
    if sqrt_size * sqrt_size != size:
        if rank == 0:
            print("警告: 进程数应为完全平方数以获得最佳性能")
        sqrt_size = 1
    
    block_size = n // sqrt_size
    
    if rank == 0:
        A_blocks = A.reshape(sqrt_size, block_size, n)
        B_blocks = B.reshape(n, sqrt_size, block_size).transpose(1, 0, 2)
    else:
        A_blocks = None
        B_blocks = None
    
    local_A = np.zeros((block_size, n))
    local_B = np.zeros((n, block_size))
    local_C = np.zeros((block_size, block_size))
    
    comm.Scatter(A_blocks, local_A, root=0)
    comm.Scatter(B_blocks, local_B, root=0)
    
    for k in range(sqrt_size):
        A_block = np.zeros((block_size, n))
        B_block = np.zeros((n, block_size))
        
        comm.Bcast(local_A, root=(rank // sqrt_size))
        comm.Bcast(local_B, root=(rank % sqrt_size))
        
        local_C += np.dot(local_A[:, k*block_size:(k+1)*block_size], 
                         local_B[k*block_size:(k+1)*block_size, :])
    
    C_blocks = np.zeros((sqrt_size, sqrt_size, block_size, block_size))
    comm.Gather(local_C, C_blocks, root=0)
    
    if rank == 0:
        return C_blocks.transpose(0, 2, 1, 3).reshape(n, n)
    return None

def serial_matrix_multiply(A, B):
    """串行矩阵乘法"""
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

def optimized_serial_multiply(A, B):
    """优化的串行矩阵乘法 (使用NumPy)"""
    return np.dot(A, B)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n = 800
    
    if rank == 0:
        print("=" * 60)
        print("并行矩阵乘法 (Fox算法)")
        print("=" * 60)
        print(f"矩阵大小: {n}×{n}")
        print(f"进程数: {size}")
        
        A = np.random.random((n, n))
        B = np.random.random((n, n))
        
        print("\n执行串行乘法...")
        start = time.time()
        C_serial = optimized_serial_multiply(A, B)
        serial_time = time.time() - start
        print(f"串行耗时: {serial_time:.4f}秒")
        
        print("\n执行并行乘法...")
        start = time.time()
        C_parallel = parallel_matrix_multiply(A.copy(), B.copy(), n)
        parallel_time = time.time() - start
        print(f"并行耗时: {parallel_time:.4f}秒")
        
        speedup = serial_time / parallel_time
        print(f"\n加速比: {speedup:.2f}x")
        
        if np.allclose(C_serial, C_parallel):
            print("✓ 结果验证正确")
        else:
            print("✗ 结果不一致!")
        
        print("=" * 60)
    
    comm.Barrier()
