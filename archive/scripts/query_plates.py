#!/usr/bin/env python3
"""
query_plates.py — PC 端查询板子上识别到的车牌

用法:
  python3 query_plates.py                    # 显示全部
  python3 query_plates.py --tail 20          # 最近 20 条
  python3 query_plates.py --watch            # 实时监控 (每 2 秒刷新)

依赖: sshpass (可选，免密用 key)
"""

import subprocess
import sys
import time
import os

BOARD = "root@192.168.0.232"
PLATES_FILE = "/mnt/sdcard/plates/plates.txt"


def fetch():
    cmd = ["ssh", "-o", "ConnectTimeout=5",
           "-o", "StrictHostKeyChecking=no",
           BOARD, f"cat {PLATES_FILE} 2>/dev/null || echo '(no results yet)'"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()


def main():
    args = sys.argv[1:]

    if "--watch" in args:
        print(f"Watching {PLATES_FILE} on {BOARD}...\n")
        seen = 0
        try:
            while True:
                text = fetch()
                lines = text.split('\n') if text else []
                if len(lines) > seen:
                    for line in lines[seen:]:
                        if line.strip():
                            print(line)
                    seen = len(lines)
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    tail = None
    for i, a in enumerate(args):
        if a == "--tail" and i + 1 < len(args):
            tail = int(args[i + 1])

    text = fetch()

    if tail:
        lines = text.split('\n')
        text = '\n'.join(lines[-tail:])

    print(text or "(no results yet)")


if __name__ == "__main__":
    main()
