from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running as: python src/compare_synth.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synthesizer import main as synth_main  # noqa: E402


def _run_one(backend: str, multiplier: int, seed: int, ctgan_epochs: int) -> None:
    # Invoke synthesizer via argv-style to keep logic in one place.
    argv = [
        'src/synthesizer.py',
        '--backend',
        backend,
        '--multiplier',
        str(multiplier),
        '--seed',
        str(seed),
        '--ctgan-epochs',
        str(ctgan_epochs),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        synth_main()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(description='Run GaussianCopula vs CTGAN synthesis and save a comparison table.')
    parser.add_argument('--multiplier', type=int, default=1, help='1 for +1x, 3 for +3x')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ctgan-epochs', type=int, default=150)
    args = parser.parse_args()

    backends = ['sdv_gcopula', 'sdv_ctgan']

    for backend in backends:
        _run_one(backend, multiplier=int(args.multiplier), seed=int(args.seed), ctgan_epochs=int(args.ctgan_epochs))

    # Combine QC summaries
    results_dir = Path('results')
    rows = []
    for backend in backends:
        p = results_dir / f'synth_qc_{args.multiplier}x_{backend}_seed{args.seed}.csv'
        if p.exists():
            df = pd.read_csv(p)
            rows.append(df)

    if not rows:
        raise FileNotFoundError('No synth QC summaries found. Run src/cleaning.py first.')

    out = pd.concat(rows, ignore_index=True)
    out_path = results_dir / f'synth_compare_{args.multiplier}x_seed{args.seed}.csv'
    out.to_csv(out_path, index=False)
    print('Saved comparison table:', out_path)
    print(out)


if __name__ == '__main__':
    main()
