import csv, statistics

rows = list(csv.DictReader(open('benchmark_results/benchmark_results.csv')))
print(f'Pages: {len(rows)}')

def stats(key):
    vals = [float(r[key]) for r in rows]
    return f'mean={statistics.mean(vals):.3f}  median={statistics.median(vals):.3f}  min={min(vals):.3f}  max={max(vals):.3f}  stdev={statistics.stdev(vals):.3f}'

print()
print('=== ACCURACY ===')
print('Overall EDR: ', stats('overall_edr'))
print('Text EDR:    ', stats('text_edr'))
print('Math EDR:    ', stats('math_edr'))
print('BLEU-4:      ', stats('bleu'))
print('ROUGE-L:     ', stats('rouge'))
print('CER (%):     ', stats('cer'))
print('WER (%):     ', stats('wer'))
print()
print('=== LATENCY (s) ===')
print('Total:       ', stats('latency'))
print('OCR:         ', stats('ocr_latency'))
print('Layout:      ', stats('layout_latency'))
print('Math:        ', stats('math_latency'))
print('Table:       ', stats('table_latency'))
print()
print('=== RAM (MB) ===')
print('Peak:        ', stats('mem_peak'))
print()

lats = sorted(float(r['latency']) for r in rows)
n = len(lats)
print('=== LATENCY PERCENTILES ===')
print(f'p50={lats[n//2]:.2f}s  p75={lats[int(n*0.75)]:.2f}s  p90={lats[int(n*0.90)]:.2f}s  p95={lats[int(n*0.95)]:.2f}s  p99={lats[int(n*0.99)]:.2f}s')
print()

slowest = sorted(rows, key=lambda r: float(r['latency']), reverse=True)[:5]
print('=== SLOWEST 5 PAGES ===')
for r in slowest:
    print(f'  Page {r["id"]}: {float(r["latency"]):.1f}s  EDR={float(r["overall_edr"]):.3f}  math_lat={float(r["math_latency"]):.1f}s  RAM={float(r["mem_peak"]):.0f}MB')
print()

failures = [r for r in rows if float(r['overall_edr']) < 0.05]
print(f'=== NEAR-ZERO EDR PAGES ({len(failures)}) ===')
for r in failures:
    print(f'  Page {r["id"]}: EDR={float(r["overall_edr"]):.4f}')
print()

# EDR distribution buckets
buckets = {'<30%': 0, '30-50%': 0, '50-70%': 0, '70-90%': 0, '>90%': 0}
for r in rows:
    e = float(r['overall_edr'])
    if e < 0.30: buckets['<30%'] += 1
    elif e < 0.50: buckets['30-50%'] += 1
    elif e < 0.70: buckets['50-70%'] += 1
    elif e < 0.90: buckets['70-90%'] += 1
    else: buckets['>90%'] += 1
print('=== EDR DISTRIBUTION ===')
for k, v in buckets.items():
    print(f'  {k}: {v} pages ({v/len(rows)*100:.1f}%)')
