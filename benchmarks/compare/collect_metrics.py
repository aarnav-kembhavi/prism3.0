"""Collect all OmniDocBench paper metrics for a set of eval result files into a
single table (and optional JSON). Usage:
    python collect_metrics.py <result_tag1> <result_tag2> ...
where each tag maps to omnidocbench_eval/result/<tag>_quick_match_metric_result.json
"""
import json, os, sys

R = "omnidocbench_eval/result/"
CATS = ['text_block', 'display_formula', 'table', 'reading_order']


def load(tag):
    p = os.path.join(R, f"{tag}_quick_match_metric_result.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding='utf-8'))


def metrics(d):
    ed = lambda m: d[m]['all']['Edit_dist']['ALL_page_avg']
    m = {c: ed(c) for c in CATS}
    m['overall_edit'] = sum(m[c] for c in CATS) / 4
    m['table_TEDS'] = d['table']['all']['TEDS']['all']
    m['table_TEDS_struct'] = d['table']['all']['TEDS_structure_only']['all']
    m['pages'] = d.get('match_debug', {}).get('page_count', '?')
    # language splits
    for lang in ['english', 'simplified_chinese', 'en_ch_mixed']:
        k = f'language: {lang}'
        for cat in ['text_block', 'display_formula']:
            v = d[cat]['page']['Edit_dist'].get(k)
            if v is not None:
                m[f'{cat[:4]}_{lang[:2]}'] = v
    return m


if __name__ == '__main__':
    rows = {}
    for tag in sys.argv[1:]:
        d = load(tag)
        rows[tag] = metrics(d) if d else None
    print(f"{'model':22s}{'pg':>4s}{'text':>7s}{'formula':>8s}{'table':>7s}{'order':>7s}{'TEDS':>7s}{'OVER':>7s}")
    print("-" * 60)
    for tag, m in rows.items():
        if m is None:
            print(f"{tag:22s}  MISSING"); continue
        print(f"{tag:22s}{str(m['pages']):>4s}{m['text_block']:7.3f}{m['display_formula']:8.3f}"
              f"{m['table']:7.3f}{m['reading_order']:7.3f}{m['table_TEDS']:7.3f}{m['overall_edit']:7.3f}")
    json.dump(rows, open("benchmarks/compare/metrics_collected.json", "w"), indent=2)
