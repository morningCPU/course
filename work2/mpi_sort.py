from mpi4py import MPI
import numpy as np
import random
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def merge_split_left(local_arr, recv_arr):
    # 左边进程：合并后保留较小的一半
    merged = np.sort(np.concatenate((local_arr, recv_arr)))
    return merged[:len(local_arr)]

def merge_split_right(local_arr, recv_arr):
    # 右边进程：合并后保留较大的一半
    merged = np.sort(np.concatenate((local_arr, recv_arr)))
    return merged[len(local_arr):]

if __name__ == "__main__":
    N = 1000000  # 全局数据规模
    local_n = N // size
    
    global_data = None
    if rank == 0:
        random.seed(42)
        global_data = np.array([random.randint(0, 10000) for _ in range(N)], dtype=np.int32)
        start_time = time.time()
    else:
        global_data = None

    # 创建本地接收数组
    local_data = np.empty(local_n, dtype=np.int32)

    # 1. 通信原语：Scatter
    # 数据流向：主进程 (Rank 0) -> 均匀切分并分发给所有进程 (包含自己) 的 local_data 缓冲区
    comm.Scatter(global_data, local_data, root=0)

    # 每个进程先对本地局部数据进行一次排序
    local_data.sort()

    # 奇偶换序并行排序核心循环
    for phase in range(size):
        # 决定当前阶段的邻居
        if phase % 2 == 1: # 奇阶段
            if rank % 2 == 1:
                neighbor = rank + 1 if rank + 1 < size else MPI.PROC_NULL
                is_left = True
            else:
                neighbor = rank - 1 if rank - 1 >= 0 else MPI.PROC_NULL
                is_left = False
        else: # 偶阶段
            if rank % 2 == 0:
                neighbor = rank + 1 if rank + 1 < size else MPI.PROC_NULL
                is_left = True
            else:
                neighbor = rank - 1 if rank - 1 >= 0 else MPI.PROC_NULL
                is_left = False

        if neighbor != MPI.PROC_NULL:
            recv_buffer = np.empty(local_n, dtype=np.int32)
            
            # 2. 通信原语：Sendrecv (同步阻塞式对等通信)
            # 数据流向：
            # - 将本地 local_data 发送给目标邻居进程 neighbor
            # - 同时从目标邻居进程 neighbor 接收数据到本地的 recv_buffer
            comm.Sendrecv(sendbuf=local_data, dest=neighbor, sendtag=phase,
                          recvbuf=recv_buffer, source=neighbor, recvtag=phase)

            # 根据自己在相邻对中的左右位置，保留对应的一半数据
            if is_left:
                local_data = merge_split_left(local_data, recv_buffer)
            else:
                local_data = merge_split_right(local_data, recv_buffer)

    # 最终结果收集
    gathered_data = None
    if rank == 0:
        gathered_data = np.empty(N, dtype=np.int32)

    # 3. 通信原语：Gather
    # 数据流向：所有进程将各自最终有序的 local_data -> 发送回主进程 (Rank 0) 组合进 gathered_data
    comm.Gather(local_data, gathered_data, root=0)

    if rank == 0:
        end_time = time.time()
        print(f"[{size} processes] Parallel Sorting completed in {end_time - start_time:.6f}s")
        print(f"[Parallel] First 10 elements: {gathered_data[:10]}")
        print(f"[Parallel] Last 10 elements: {gathered_data[-10:]}")