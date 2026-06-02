from mpi4py import MPI
import numpy as np
import random
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def merge_split_left(local_arr, recv_arr):
    merged = np.sort(np.concatenate((local_arr, recv_arr)))
    return merged[:len(local_arr)]

def merge_split_right(local_arr, recv_arr):
    merged = np.sort(np.concatenate((local_arr, recv_arr)))
    return merged[len(local_arr):]

if __name__ == "__main__":
    N = 10000  # 固定与 B-2 相同的规模进行公平对比
    local_n = N // size
    
    global_data = None
    if rank == 0:
        random.seed(42)
        global_data = np.array([random.randint(0, 10000) for _ in range(N)], dtype=np.int32)
        start_time = time.time()
    else:
        global_data = None

    local_data = np.empty(local_n, dtype=np.int32)
    comm.Scatter(global_data, local_data, root=0)
    local_data.sort()

    # 奇偶换序非阻塞核心循环
    for phase in range(size):
        if (rank + phase) % 2 == 0:
            neighbor = rank + 1 if rank + 1 < size else MPI.PROC_NULL
            is_left = True
        else:
            neighbor = rank - 1 if rank - 1 >= 0 else MPI.PROC_NULL
            is_left = False

        if neighbor != MPI.PROC_NULL:
            recv_buffer = np.empty(local_n, dtype=np.int32)
            
            # 使用非阻塞原语替换阻塞的 Sendrecv
            # comm.Irecv / comm.Isend 发起后立即返回，由底层网络硬件或后台线程处理传输
            req_recv = comm.Irecv(recv_buffer, source=neighbor, tag=phase)
            req_send = comm.Isend(local_data, dest=neighbor, tag=phase)
            
            # 【计算与通信重叠区】
            # 在这里 CPU 可以并行处理一些与接收缓冲区无关的操作（如当前阶段的一些状态准备）
            # 对于奇偶排序，重叠区间较小，但在更复杂的算法中此处可放置密集的无关计算
            
            # 同步流控：必须在归并前确保非阻塞传输全部安全完成
            MPI.Request.Waitall([req_recv, req_send])

            if is_left:
                local_data = merge_split_left(local_data, recv_buffer)
            else:
                local_data = merge_split_right(local_data, recv_buffer)

    gathered_data = None
    if rank == 0:
        gathered_data = np.empty(N, dtype=np.int32)

    comm.Gather(local_data, gathered_data, root=0)

    if rank == 0:
        end_time = time.time()
        print(f"[{size} processes] Non-blocking Parallel Sorting completed in {end_time - start_time:.6f}s")