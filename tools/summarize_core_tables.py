"""Aggregate completed benchmark runs for main-text Tables 1, 2, and 4."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from record_ids import validate_label_join

METHODS=('scaled_rmsnorm_empirical_gaussian','flowprep_scaled_rmsnorm_maf','diffcsp','mattergen_mp20','cdvae','symat','ftcp')
SEEDS=(13,37,73)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delac-run',type=Path,required=True)
    p.add_argument('--baseline-run',type=Path,required=True)
    p.add_argument('--include-learned',action='store_true',help='Also write Table 4 from paired Raw/Learned runs')
    p.add_argument('--all-arms',action='store_true',help='Also aggregate Rule/Union policies for supplementary tables')
    p.add_argument('--sun-overrides',type=Path,help='JSON mapping method/seed/arm to a completed SUN recovery directory')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    overrides=json.loads(a.sun_overrides.read_text()) if a.sun_overrides else {}
    arms=('raw','rule_filter','learned_filter','rule_union_learned') if a.all_arms else (('raw','learned_filter') if a.include_learned else ('raw',))
    valid_keys={f'{method}/{seed}/{arm}' for method in METHODS for seed in SEEDS for arm in arms}
    if set(overrides)-valid_keys: raise ValueError(f'Unknown or unselected SUN overrides: {sorted(set(overrides)-valid_keys)}')
    import numpy as np
    import pandas as pd
    ours=json.loads((a.delac_run/'pipeline_status.json').read_text())
    others=json.loads((a.baseline_run/'pipeline_status.json').read_text())
    rows=[]
    for method in METHODS:
        for seed in SEEDS:
            if method==METHODS[0]:
                state=ours['seed_results'][str(seed)];base=a.delac_run/f'seed_{seed}'
            else:
                state=others['models'][method]['seeds'][str(seed)];base=a.baseline_run/method/f'seed_{seed}'
            for arm in arms:
                record=state['arms'][arm]
                if record['status'] not in ('completed','sun_timeout'):
                    raise ValueError(f'{method}/{seed}/{arm}: evaluation is not complete')
                directory=base/arm
                override=overrides.get(f'{method}/{seed}/{arm}')
                evaluated=Path(override) if override else directory
                if override and not evaluated.is_absolute(): evaluated=a.sun_overrides.resolve().parent/evaluated
                labels=pd.read_csv(evaluated/'sun_relaxed_labels.csv')
                metrics=json.loads((evaluated/'sun_relaxed.json').read_text())
                relaxed=pd.read_csv(directory/f'ordinary_workers_2_{arm}/relaxed_merged.csv',low_memory=False)
                validate_label_join(relaxed,labels,'candidate_id')
                def number(key):
                    if key not in metrics:
                        counts={'frac_stable_structures':'stable_count','frac_unique_structures':'unique_count',
                                'frac_novel_structures':'novel_count','frac_novel_unique_stable_structures':'sun_count'}
                        return float(metrics[counts[key]])/int(metrics['retained_materials_after_hull_timeout'])
                    v=metrics[key]
                    return float(v['value'] if isinstance(v,dict) else v)
                if 'candidate_id' in labels and labels.candidate_id.duplicated().any(): raise ValueError('Duplicate candidate_id in SUN labels')
                sun=int(labels.SUN.astype(str).str.lower().isin(('true','1','yes')).sum())
                if not np.isclose(number('frac_novel_unique_stable_structures'),sun/len(labels)):
                    raise ValueError(f'{method}/{seed}/{arm}: SUN summary/label mismatch')
                timing=record['timing']
                stage1=float(timing['stage1_full_seconds'])/60
                stage2=float(timing['T_stage2_corrected_seconds'])/60
                end=float(timing['system_e2e_corrected_seconds'])/60
                if not np.isclose(stage1+stage2,end): raise ValueError('Inconsistent end-to-end timing')
                rows.append({'method':method,'seed':seed,'arm':arm,'controller_status':record['status'],
                    'requests_or_admitted':1000,'evaluable_labels':len(labels),'sun_source':str(evaluated),
                    'sun_count':sun,'sun_percent':sun/10,'stable_percent':100*number('frac_stable_structures'),
                    'unique_percent':100*number('frac_unique_structures'),'novel_percent':100*number('frac_novel_structures'),
                    'stage1_min':stage1,'stage2_min':stage2,'e2e_min':end})
    frame=pd.DataFrame(rows)
    raw=frame.loc[frame.arm.eq('raw')]
    table1=[];table4=[]
    for method in METHODS:
        data=raw.loc[raw.method.eq(method)]
        assert len(data)==3
        summary={'method':method}
        for key in ('sun_percent','stable_percent','unique_percent','novel_percent','stage1_min','stage2_min','e2e_min'):
            summary[key+'_mean']=float(data[key].mean());summary[key+'_sample_std']=float(data[key].std(ddof=1))
        table1.append(summary)
        if a.include_learned or a.all_arms:
            learned=frame.loc[frame.method.eq(method)&frame.arm.eq('learned_filter')]
            assert len(learned)==3
            t_raw=float(data.e2e_min.mean());t_learned=float(learned.e2e_min.mean())
            table4.append({'method':method,'raw_e2e_min':t_raw,'learned_e2e_min':t_learned,
                'time_change_percent':100*(t_learned/t_raw-1),
                'raw_sun_percent':float(data.sun_percent.mean()),'learned_sun_percent':float(learned.sun_percent.mean()),
                'raw_throughput_sun_per_min':float(data.sun_count.mean()/t_raw),
                'learned_throughput_sun_per_min':float(learned.sun_count.mean()/t_learned)})
    a.output_dir.mkdir(parents=True,exist_ok=False)
    frame.to_csv(a.output_dir/'by_seed.csv',index=False)
    pd.DataFrame(table1).to_csv(a.output_dir/'table1_quality_and_table2_timing.csv',index=False)
    if a.include_learned or a.all_arms:pd.DataFrame(table4).to_csv(a.output_dir/'table4_screening.csv',index=False)
    if a.all_arms:
        columns=['sun_percent','stable_percent','unique_percent','novel_percent','e2e_min']
        policy=frame.groupby(['method','arm'])[columns].agg(['mean','std'])
        policy.columns=[metric+'_'+('sample_std' if stat=='std' else stat) for metric,stat in policy.columns]
        policy.reset_index().to_csv(a.output_dir/'supplementary_four_policy_summary.csv',index=False)
    (a.output_dir/'aggregation.json').write_text(json.dumps({'seeds':SEEDS,'standard_deviation':'sample, ddof=1',
        'sun_denominator':1000,'throughput':'mean SUN count / mean E2E minutes; not mean of per-seed ratios',
        'sun_evaluation_included_in_e2e':False},indent=2)+'\n')
    print(a.output_dir)


if __name__=='__main__':main()
