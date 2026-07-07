#!/usr/bin/env python3
"""单进程总 tqdm：汇总各分片写入的 .infer_progress_shard_* 计数文件（由 infer_seed --shard_progress_file 更新）。"""
import argparse
import os
import signal
import sys
import time

from tqdm import tqdm


def _read_int(path: str) -> int:
    try:
        with open(path, 'r', encoding='ascii') as f:
            return int((f.read() or '0').strip() or 0)
    except Exception:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--total', type=int, required=True)
    ap.add_argument('--num_shards', type=int, required=True)
    ap.add_argument('--interval', type=float, default=0.8)
    args = ap.parse_args()

    paths = [
        os.path.join(args.outdir, f'.infer_progress_shard_{i}')
        for i in range(args.num_shards)
    ]

    stop = False

    def _on_term(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    def sum_done() -> int:
        return sum(_read_int(p) for p in paths)

    last = 0
    with tqdm(
        total=args.total,
        desc='Total wav',
        unit='wav',
        file=sys.stderr,
        mininterval=args.interval,
        dynamic_ncols=True,
    ) as bar:
        while not stop:
            cur = min(sum_done(), args.total)
            if cur > last:
                bar.update(cur - last)
                last = cur
            if last >= args.total:
                time.sleep(args.interval * 2)
                cur2 = min(sum_done(), args.total)
                if cur2 >= args.total:
                    break
            time.sleep(args.interval)
        if last < args.total:
            cur = min(sum_done(), args.total)
            if cur > last:
                bar.update(cur - last)


if __name__ == '__main__':
    main()
