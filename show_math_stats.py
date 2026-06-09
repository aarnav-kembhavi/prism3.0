import csv
rows = list(csv.DictReader(open('benchmark_results/benchmark_results.csv')))

worst = sorted(rows, key=lambda r: float(r['math_edr']))[:8]
print('Worst math EDR pages:')
for r in worst:
    print(f'  Page {r["id"]}: math_edr={float(r["math_edr"]):.3f}  text_edr={float(r["text_edr"]):.3f}  math_lat={float(r["math_latency"]):.2f}s')

print()
pages_with_math = [r for r in rows if float(r['math_latency']) > 0]
print(f'Pages that hit Texo: {len(pages_with_math)}/{len(rows)}')
import statistics
if pages_with_math:
    medrs = [float(r['math_edr']) for r in pages_with_math]
    tedrs = [float(r['text_edr']) for r in pages_with_math]
    print(f'  Math EDR on those pages: mean={statistics.mean(medrs):.3f}  median={statistics.median(medrs):.3f}')
    print(f'  Text EDR on those pages: mean={statistics.mean(tedrs):.3f}  median={statistics.median(tedrs):.3f}')

pages_no_math = [r for r in rows if float(r['math_latency']) == 0]
print(f'Pages with no Texo (text-only): {len(pages_no_math)}/{len(rows)}')
if pages_no_math:
    tedrs = [float(r['text_edr']) for r in pages_no_math]
    medrs = [float(r['math_edr']) for r in pages_no_math]
    print(f'  Text EDR on those pages: mean={statistics.mean(tedrs):.3f}  median={statistics.median(tedrs):.3f}')
    print(f'  Overall EDR on those pages: mean={statistics.mean([float(r["overall_edr"]) for r in pages_no_math]):.3f}')
