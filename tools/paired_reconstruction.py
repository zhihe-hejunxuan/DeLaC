"""Paired full-structure match statistics for the normalization diagnostic."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--none',type=Path,required=True)
    p.add_argument('--scaled-rmsnorm',type=Path,required=True)
    p.add_argument('--bootstrap-samples',type=int,default=10000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import json
    import numpy as np
    import pandas as pd
    from scipy.stats import binomtest
    def load(path):
        f=pd.read_csv(path).set_index('index')
        if f.index.duplicated().any(): raise ValueError('Duplicate reconstruction index')
        return f.full_match.astype(str).str.lower().isin(('true','1')).sort_index()
    x,y=load(a.none),load(a.scaled_rmsnorm)
    if not x.index.equals(y.index): raise ValueError('Reconstruction subsets differ')
    delta=y.to_numpy().astype(int)-x.to_numpy().astype(int)
    rng=np.random.default_rng(a.seed)
    boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean()*100 for _ in range(a.bootstrap_samples)])
    b,c=int((delta==-1).sum()),int((delta==1).sum())
    result={'n':len(delta),'difference_percentage_points':float(delta.mean()*100),
            'bootstrap_95_ci_percentage_points':np.quantile(boot,[.025,.975]).tolist(),
            'exact_mcnemar_p':float(binomtest(b,b+c,.5).pvalue) if b+c else 1.,
            'bootstrap_samples':a.bootstrap_samples,'seed':a.seed}
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(a.output)

if __name__=='__main__': main()
