import random
import time

def odd_even_sort_serial(arr):
    n = len(arr)
    is_sorted = False
    while not is_sorted:
        is_sorted = True
        # 奇阶段
        for i in range(1, n - 1, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                is_sorted = False
        # 偶阶段
        for i in range(0, n - 1, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                is_sorted = False
    return arr

if __name__ == "__main__":
    # 固定随机种子确保可对比性，生成 2000 个随机数
    random.seed(42)
    data = [random.randint(0, 10000) for _ in range(2000)]
    
    start = time.time()
    sorted_data = odd_even_sort_serial(data)
    end = time.time()
    
    print(f"[Serial] Sorting completed in {end - start:.6f}s")
    print(f"[Serial] First 10 elements: {sorted_data[:10]}")
    print(f"[Serial] Last 10 elements: {sorted_data[-10:]}")