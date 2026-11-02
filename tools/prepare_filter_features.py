"""Extract pre-relaxation features from a completed development collection."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'final_experiments/experiment_speed')]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'result/filter_features/pre_relax_features.csv')
    a=p.parse_args()
    import pandas as pd
    from filter_features import structure_features
    records=[]
    for split in ('train','calibration','development_test'):
        frame=pd.read_csv(a.data_root/'evaluation'/split/'relaxed.csv',low_memory=False)
        for row in frame.itertuples(index=False):
            f=structure_features(row.cif_str_unrelaxed,with_smact=True)
            if not f['parseable']: raise ValueError(f'Cannot parse {row.dataset_id}: {f["feature_error"]}')
            records.append({'dataset_id':row.dataset_id,'source_split':split,**f})
    out=pd.DataFrame(records)
    if out.dataset_id.duplicated().any(): raise ValueError('Duplicate dataset_id')
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output,index=False)
    print(a.output)

if __name__=='__main__': main()
