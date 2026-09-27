#!/usr/bin/env python3
"""Parallel, resumable HTTP-Range downloader (v2).

v1 的教训：用 os.dup() 复制 fd 后多线程 seek+write，会共享同一个文件偏移，
互相踩写、还会把文件写长。本版改用 os.pwrite（按偏移原子写，不需要 seek），
并且进度状态是「每块的字节数」而不是区间列表，重启后能精确续传。

用法: fetch_par2.py <url> <out> <total_bytes> [workers] [chunk_MiB] [sha256]
"""
import hashlib
import json
import os
import queue
import socket
import sys
import threading
import time
import urllib.request

MIB = 1024 * 1024
IO_BUF = 4 * MIB


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Progress:
    def __init__(self, state_path, total):
        self.path = state_path
        self.total = total
        self.lock = threading.Lock()
        self.chunks = {}
        self.t0 = time.time()
        self.last_report = 0.0
        if os.path.exists(state_path):
            try:
                with open(state_path) as fh:
                    self.chunks = {int(k): int(v) for k, v in json.load(fh).items()}
            except Exception:  # noqa: BLE001
                self.chunks = {}
        self.bytes_done = sum(self.chunks.values())

    def add(self, idx, n):
        with self.lock:
            self.chunks[idx] = self.chunks.get(idx, 0) + n
            self.bytes_done += n
            now = time.time()
            if now - self.last_report > 3:
                self.last_report = now
                self.save()
                rate = self.bytes_done / max(1e-9, now - self.t0)
                log(f"{self.bytes_done * 100 / self.total:5.1f}%  "
                    f"{self.bytes_done / 1e9:.2f}/{self.total / 1e9:.2f} GB  {rate / 1e6:.1f} MB/s")

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({str(k): v for k, v in self.chunks.items()}, fh)
        os.replace(tmp, self.path)


def fetch_chunk(url, start, end, fd, idx, prog):
    done = prog.chunks.get(idx, 0)
    while done < end - start:
        lo = start + done
        req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{end - 1}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status != 206:
                    raise RuntimeError(f"expected 206, got {resp.status}")
                while done < end - start:
                    buf = resp.read(IO_BUF)
                    if not buf:
                        break
                    if done + len(buf) > end - start:
                        buf = buf[: end - start - done]
                    os.pwrite(fd, buf, start + done)
                    done += len(buf)
                    prog.add(idx, len(buf))
        except Exception as exc:  # noqa: BLE001
            log(f"chunk {idx} retry@{done / 1e6:.0f}MB: {type(exc).__name__}: {exc}")
            time.sleep(3)


def sha256_file(path, total):
    h = hashlib.sha256()
    read = 0
    t0 = time.time()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(16 * MIB)
            if not buf:
                break
            h.update(buf)
            read += len(buf)
            if time.time() - t0 > 10:
                t0 = time.time()
                log(f"sha256 {read * 100 / total:.0f}%")
    return h.hexdigest()


def main():
    url, out, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    chunk_mib = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    want_sha = sys.argv[6] if len(sys.argv) > 6 else None
    chunk = chunk_mib * MIB

    if not os.path.exists(out) or os.path.getsize(out) != total:
        log(f"pre-allocate {out} -> {total} bytes")
        with open(out, "wb") as fh:
            fh.truncate(total)

    prog = Progress(f"{out}.state.json", total)
    bounds = list(range(0, total, chunk)) + [total]
    jobs = queue.Queue()
    for idx in range(len(bounds) - 1):
        if prog.chunks.get(idx, 0) >= bounds[idx + 1] - bounds[idx]:
            continue
        jobs.put((idx, bounds[idx], bounds[idx + 1]))
    log(f"{len(bounds) - 1} chunks of {chunk_mib} MiB, {jobs.qsize()} to fetch, "
        f"already {prog.bytes_done / 1e9:.2f} GB")

    fd = os.open(out, os.O_RDWR)

    def worker():
        while True:
            try:
                idx, start, end = jobs.get_nowait()
            except queue.Empty:
                return
            fetch_chunk(url, start, end, fd, idx, prog)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    prog.save()
    os.close(fd)

    missing = [i for i in range(len(bounds) - 1)
               if prog.chunks.get(i, 0) < bounds[i + 1] - bounds[i]]
    if missing:
        log(f"INCOMPLETE: {len(missing)} chunks missing (first {missing[:5]})")
        sys.exit(2)
    log(f"all {len(bounds) - 1} chunks complete")

    if want_sha:
        got = sha256_file(out, total)
        ok = got == want_sha
        log(f"sha256 {got} {'== expected ✓' if ok else '!= expected ' + want_sha}")
        sys.exit(0 if ok else 3)


if __name__ == "__main__":
    socket.setdefaulttimeout(60)
    main()
