"""Run all 4 images in one process so models stay warm after the first call."""
import sys, time, os, psutil

process = psutil.Process(os.getpid())
images  = ['image.png', 'image2.png', 'image3.jpeg', 'image4.png']

from orchestrate import main  # import once — loads nothing yet

for img in images:
    sys.argv = ['orchestrate.py', img, '--profile']
    t0 = time.perf_counter()
    main()
    elapsed = time.perf_counter() - t0
    ram = process.memory_info().rss / 1024 / 1024
    print(f"\n>>> {img}: {elapsed:.1f}s total  |  RAM now: {ram:.0f}MB\n")
    print("=" * 60)
