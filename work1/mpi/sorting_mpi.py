from mpi4py import MPI
import numpy as np
import time

def parallel_odd_even_sort(data):
    """
    并行奇偶换序排序算法
    基于冒泡排序的并行实现，相邻进程间进行比较与交换
    
    算法步骤：
    1. 将数据均匀分配到各个进程
    2. 每个进程对本地数据进行排序
    3. 进行多轮奇偶交替的全局排序：
       - 偶数轮：偶数编号进程与右侧相邻进程交换数据
       - 奇数轮：奇数编号进程与右侧相邻进程交换数据
    4. 收集最终排序结果
    
    参数：
    data: 待排序的数组（仅在rank 0上有效）
    
    返回：
    排序后的数组（仅在rank 0上返回）
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 初始化数据（仅rank 0有数据）
    if rank == 0:
        n = len(data)
        # 将数据分成size个部分
        local_n = n // size
        remainder = n % size
        
        # 准备发送给各个进程的数据
        sendbuf = []
        start = 0
        for i in range(size):
            end = start + local_n + (1 if i < remainder else 0)
            sendbuf.append(data[start:end].copy())
            start = end
    else:
        sendbuf = None
    
    # 分散数据到各个进程
    local_data = comm.scatter(sendbuf, root=0)
    local_n = len(local_data)
    
    # 步骤1：本地排序（使用快速排序）
    local_data.sort()
    
    # 步骤2：并行奇偶排序阶段
    for phase in range(size):
        # 偶数阶段：偶数rank与rank+1交换
        if phase % 2 == 0:
            if rank % 2 == 0 and rank < size - 1:
                # 偶数rank发送数据给rank+1，并接收rank+1的数据
                partner = rank + 1
                send_data = local_data.copy()
                recv_data = np.empty(local_n, dtype=local_data.dtype)
                
                # 发送并接收
                comm.send(send_data, dest=partner, tag=0)
                recv_data = comm.recv(source=partner, tag=0)
                
                # 合并并排序
                merged = np.concatenate((local_data, recv_data))
                merged.sort()
                
                # 保留前半部分
                local_data = merged[:local_n]
            
            elif rank % 2 == 1 and rank < size - 1:
                # 奇数rank接收数据并发送
                partner = rank - 1
                send_data = local_data.copy()
                recv_data = np.empty(local_n, dtype=local_data.dtype)
                
                recv_data = comm.recv(source=partner, tag=0)
                comm.send(send_data, dest=partner, tag=0)
                
                # 合并并排序
                merged = np.concatenate((recv_data, local_data))
                merged.sort()
                
                # 保留后半部分
                local_data = merged[local_n:]
        
        # 奇数阶段：奇数rank与rank+1交换
        else:
            if rank % 2 == 1 and rank < size - 1:
                # 奇数rank发送数据给rank+1
                partner = rank + 1
                send_data = local_data.copy()
                recv_data = np.empty(local_n, dtype=local_data.dtype)
                
                comm.send(send_data, dest=partner, tag=1)
                recv_data = comm.recv(source=partner, tag=1)
                
                merged = np.concatenate((local_data, recv_data))
                merged.sort()
                local_data = merged[:local_n]
            
            elif rank % 2 == 0 and rank > 0:
                # 偶数rank接收数据并发送
                partner = rank - 1
                send_data = local_data.copy()
                recv_data = np.empty(local_n, dtype=local_data.dtype)
                
                recv_data = comm.recv(source=partner, tag=1)
                comm.send(send_data, dest=partner, tag=1)
                
                merged = np.concatenate((recv_data, local_data))
                merged.sort()
                local_data = merged[local_n:]
        
        # 同步所有进程
        comm.Barrier()
    
    # 收集所有进程的排序结果
    result = comm.gather(local_data, root=0)
    
    if rank == 0:
        # 合并所有部分
        sorted_data = np.concatenate(result)
        return sorted_data
    return None

def serial_sort(data):
    """串行排序（使用Python内置排序）"""
    return np.sort(data)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 测试数据大小
    n = 100000
    
    if rank == 0:
        print("=" * 60)
        print("MPI 并行奇偶换序排序")
        print("=" * 60)
        print(f"数据规模: {n}")
        print(f"进程数: {size}")
        
        # 生成随机测试数据
        np.random.seed(42)
        data = np.random.randint(0, 1000000, size=n)
        
        # 保存原始数据用于验证
        original_data = data.copy()
        
        print("\n执行串行排序...")
        start_time = time.time()
        serial_result = serial_sort(data.copy())
        serial_time = time.time() - start_time
        print(f"串行耗时: {serial_time:.4f}秒")
        
        print("\n执行并行排序...")
        start_time = time.time()
        parallel_result = parallel_odd_even_sort(data.copy())
        parallel_time = time.time() - start_time
        print(f"并行耗时: {parallel_time:.4f}秒")
        
        # 计算加速比
        speedup = serial_time / parallel_time
        print(f"\n加速比: {speedup:.2f}x")
        
        # 验证结果
        if np.array_equal(serial_result, parallel_result):
            print("✓ 排序结果验证正确")
        else:
            print("✗ 排序结果不一致!")
        
        print("=" * 60)
    
    else:
        # 非0进程只需参与排序
        parallel_odd_even_sort(None)