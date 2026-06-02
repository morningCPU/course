from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 10_000_000
local_count = 0
for _ in range(N):
    x, y = random.random(), random.random()
    if x*x + y*y <= 1.0:
        local_count += 1

total = comm.reduce(local_count, op=MPI.SUM, root=0)

if rank == 0:
    pi = 4.0 * total / (N * size)
    print(f"[{size} processes] π ≈ {pi:.6f}")
