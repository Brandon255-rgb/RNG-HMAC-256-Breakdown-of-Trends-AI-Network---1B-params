# python 3.10+, run: python gen_1b_rolls.py
import hmac, hashlib, math, os, mmap
import numpy as np
from multiprocessing import Pool, cpu_count

SERVER = "3428e6f9695f8643802530f8694b75a4efd9f22e50cbf7d5f6a1e21ce0e8bb92"  # ASCII key
CLIENT = "3f95f77b5e864e15"
CURSOR = 0

OUT_PATH = "rolls_1e9.u16"        # 2.0 GB for 1e9 rows
N_ROLLS  = 1_000_000_000
CHUNK    = 1_000_000              # tune for your CPU
PROCS    = max(1, cpu_count() - 1)

def map4(b1, b2, b3, b4):
    # first 4 bytes -> uniform in [0,10000]
    u = (b1/256) + (b2/256**2) + (b3/256**3) + (b4/256**4)
    return int(u * 10001)  # 0..10000

def make_chunk(args):
    start, count, server, client, cursor = args
    key = server.encode()
    out = np.empty(count, dtype=np.uint16)
    msg_prefix = f"{client}:".encode()
    colon = b":"
    for i in range(count):
        n = start + i
        # build message: client:nonce:cursor
        m = msg_prefix + str(n).encode() + colon + str(cursor).encode()
        h = hmac.new(key, m, hashlib.sha256).digest()
        out[i] = map4(h[0], h[1], h[2], h[3])
    return start, out

def main():
    # preallocate file size
    with open(OUT_PATH, "wb") as f:
        f.truncate(N_ROLLS * 2)

    tasks = []
    for off in range(0, N_ROLLS, CHUNK):
        tasks.append((off, min(CHUNK, N_ROLLS - off), SERVER, CLIENT, CURSOR))

    with Pool(PROCS) as pool, open(OUT_PATH, "r+b", buffering=0) as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
        try:
            idx = 0
            for start, arr in pool.imap_unordered(make_chunk, tasks, chunksize=1):
                mm[start*2 : (start + arr.size)*2] = arr.tobytes()
                idx += arr.size
                if idx % 50_000_000 == 0:
                    print(f"wrote {idx:,} rolls")
        finally:
            mm.flush()
            mm.close()

if __name__ == "__main__":
    main()
