# Adapted for Private 3D Studio; original copyright Gaurav Kumar (MIT).
"""Copies the weights dataset to local disk with many parallel reads.

Kaggle streams dataset files over the network on first read (~17 MB/s for one sequential reader), so the models
loaded straight from /kaggle/input in 7+ min each. Reading 64 MB chunks on 16 threads runs during the install.
Usage: python stage_weights.py <dataset_dir> <local_dir>
"""
import os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

src, dst = sys.argv[1], sys.argv[2]
CHUNK = 64 << 20

files = [os.path.join(d, f) for d, _, fs in os.walk(src) for f in fs]
total = sum(os.path.getsize(p) for p in files)
jobs = []
for p in files:
    out = os.path.join(dst, os.path.relpath(p, src))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    size = os.path.getsize(p)
    with open(out, "wb") as f:
        f.truncate(size)
    jobs += [(p, out, off, min(CHUNK, size - off)) for off in range(0, size, CHUNK)]
jobs.sort(key=lambda j: j[2])  # spread early chunks across all files

copied, lock = 0, threading.Lock()


def copy(job):
    global copied
    p, out, off, n = job
    with open(p, "rb") as fi, open(out, "r+b") as fo:
        fi.seek(off)
        fo.seek(off)
        while n:
            buf = fi.read(min(n, 8 << 20))
            if not buf:
                raise IOError(f"short read: {p}")
            fo.write(buf)
            n -= len(buf)
            with lock:
                copied += len(buf)


def report():
    while copied < total:
        time.sleep(30)
        print(f"staged {copied / 2**30:.1f} / {total / 2**30:.1f} GB "
              f"({copied / 2**20 / (time.time() - start):.0f} MB/s)", flush=True)


start = time.time()
threading.Thread(target=report, daemon=True).start()
with ThreadPoolExecutor(16) as pool:
    list(pool.map(copy, jobs))
open(os.path.join(dst, "_STAGED"), "w").close()
print(f"STAGED {total / 2**30:.1f} GB in {time.time() - start:.0f}s", flush=True)
