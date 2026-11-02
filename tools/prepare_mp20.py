"""Prepare the published MP-20 CSV splits without changing their assignment."""
from pathlib import Path
import argparse
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'data/mp_20')
    args = parser.parse_args()
    from dataset import MatDataset
    args.output.mkdir(parents=True, exist_ok=True)
    for split, expected in [('train',25339),('val',8469),('test',8484)]:
        source = args.csv_dir/f'{split}.csv'
        dest = args.output/source.name
        if not source.is_file(): raise FileNotFoundError(source)
        if dest.exists() or dest.with_suffix('.pt').exists():
            raise FileExistsError(f'Refusing to replace {dest} or its preprocessed cache')
        shutil.copy2(source,dest)
        ds = MatDataset(str(dest), graph_method='crystalnn')
        if len(ds)!=expected:
            raise RuntimeError(f'{split}: obtained {len(ds)} instead of the paper split {expected}; inspect the source version/parse failures before training')
        print(f'{split}: {len(ds)} structures -> {dest.with_suffix(".pt")}')

if __name__=='__main__': main()
