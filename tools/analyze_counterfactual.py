"""Apply the frozen filter to completed Raw evaluations; use measured costs/labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'final_experiments/experiment_speed')]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw-relaxed',type=Path,required=True)
    p.add_argument('--sun-labels',type=Path,required=True)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--id-column',default='candidate_id')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    import joblib
    import numpy as np
    import pandas as pd
    from filter_features import structure_features
    raw=pd.read_csv(a.raw_relaxed,low_memory=False)
    labels=pd.read_csv(a.sun_labels)
    key=a.id_column
    from record_ids import validate_label_join
    validate_label_join(raw,labels,key)
    for col in ('cif_str_unrelaxed','relax_wall_seconds','relax_converged','relax_steps'):
        if col not in raw: raise ValueError(f'Missing measured Raw column: {col}')
    data=raw.drop(columns=['SUN'],errors='ignore').merge(labels[[key,'SUN']],on=key,how='left',validate='one_to_one',indicator=True)
    data['sun_label_available']=data._merge.eq('both')
    features=pd.DataFrame([structure_features(x,with_smact=True) for x in data.cif_str_unrelaxed])
    if not features.parseable.all(): raise ValueError('Unparseable Raw pre-relaxation structure')
    artifact=joblib.load(a.artifact)
    cols=artifact['features']
    matrix=features[cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(pd.Series(artifact['medians'])).fillna(0)
    risk=artifact['model'].predict_proba(matrix)[:,1]
    reject=risk>=artifact['threshold']
    cost=pd.to_numeric(data.relax_wall_seconds,errors='coerce')
    if cost.isna().any() or (cost<0).any(): raise ValueError('Missing/invalid measured relaxation time')
    sun=data.SUN.astype(str).str.lower().isin(('true','1'))
    result=pd.DataFrame({'candidate_id':data[key],'risk_score':risk,'rejected':reject,
                         'relax_time_s':cost,'is_sun':sun,'sun_label_available':data.sun_label_available,
                         'converged':data.relax_converged,'relax_steps':data.relax_steps})
    summary={'raw_candidates':len(result),'rejected_fraction':float(reject.mean()),
             'fraction_of_measured_relaxation_cost':float(cost[reject].sum()/cost.sum()) if cost.sum() else None,
             'sun_retention':float((sun&~reject).sum()/sun.sum()) if sun.sum() else None,
             'rows_without_sun_label':int((~data.sun_label_available).sum())}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    result.to_csv(a.output_dir/'candidate_counterfactual.csv',index=False)
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(a.output_dir)

if __name__=='__main__': main()
