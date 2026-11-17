"""Is the balanced runs' trace error (max |tr rho - 1|) inherited from the input PT?

Compares, per seed, max_t |tr rho - 1| of the untruncated reference outputs (seed_<s>/ref/ref_*.out)
with that of the truncated runs (seed_<s>/maxk_<k>/runs/<pt>_<ctx>.out) and reports the largest
deviation |tr rho_run - tr rho_ref| per method. A closure-protected sweep keeps the trace of the
input PT, so for balanced runs the deviation should be far below the reference's own trace error.
Standard library only (runs on ARCHER2 login nodes):  python3 trace_floor.py results/joined_cavity_mixed
"""
import collections
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_benchmark import read_outfile  # noqa: E402


def main(root):
    for sdir in sorted(glob.glob(os.path.join(root, 'seed_*'))):
        ref = {}
        for f in glob.glob(os.path.join(sdir, 'ref', 'ref_*.out')):
            ref[os.path.basename(f)[len('ref_'):]] = read_outfile(f)[2]
        if not ref:
            continue
        ref_err = max(abs(x - 1) for tr in ref.values() for x in tr)
        print(f'{os.path.basename(sdir)}: reference ({len(ref)} contexts) max |tr - 1| = {ref_err:.2e}')
        dev = collections.defaultdict(float)
        own = collections.defaultdict(float)
        for f in glob.glob(os.path.join(sdir, 'maxk_*', 'runs', '*.out')):
            name = os.path.basename(f)
            tag, _, ctx = name.partition('.pt_')
            if ctx not in ref:
                continue
            tr = read_outfile(f)[2]
            method = tag.rsplit('_', 1)[0] + ' eta ' + tag.rsplit('_', 1)[1] if tag.startswith('balanced_') else tag
            dev[method] = max(dev[method], max(abs(a - b) for a, b in zip(tr, ref[ctx])))
            own[method] = max(own[method], max(abs(a - 1) for a in tr))
        for m in sorted(dev, key=lambda s: (s.split('_')[0], int(s.split('_')[1].split()[0]), s)):
            print(f'  {m:22s} max|tr - 1| = {own[m]:.2e}   max|tr - tr_ref| = {dev[m]:.2e}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join('results', 'joined_cavity_mixed'))
