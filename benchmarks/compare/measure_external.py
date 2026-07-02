"""
Launch a command as a subprocess and sample its process-tree peak RSS + wall
time. Used to measure PRISM (which spawns worker subprocesses) with the same
tree-aware method bench_metrics uses in-process for the competitors.

Usage:
  python measure_external.py <out_json> -- <command...>
"""
import sys, time, json, threading, subprocess
import psutil


def main():
    sep = sys.argv.index('--')
    out_json = sys.argv[1]
    cmd = sys.argv[sep + 1:]

    proc = subprocess.Popen(cmd)
    p = psutil.Process(proc.pid)
    peak = [0.0]
    running = [True]

    def sample():
        while running[0]:
            try:
                total = p.memory_info().rss
                for c in p.children(recursive=True):
                    try:
                        total += c.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak[0] = max(peak[0], total / 1024 / 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.1)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0 = time.perf_counter()
    proc.wait()
    wall = time.perf_counter() - t0
    running[0] = False
    th.join(timeout=2)

    json.dump({"peak_rss_mb": round(peak[0], 1), "wall_s": round(wall, 1)},
              open(out_json, 'w'), indent=2)
    print(f"[measure] peak_rss={peak[0]:.0f}MB wall={wall:.1f}s")


if __name__ == '__main__':
    main()
