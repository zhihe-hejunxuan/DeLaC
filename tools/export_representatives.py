"""Export paired input/final CIFs for the three SUN examples in the supplement."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw-relaxed',type=Path,help='Optional structure CSV from the SAME archived representative run')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--sun-labels',type=Path,help='Original archived SUN-label CSV from the representative run')
    source.add_argument('--data-dir',type=Path,help='External directory containing pairs.csv and its referenced before/after CIFs')
    p.add_argument('--id-column',default='request_id')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    import pandas as pd
    from pymatgen.core import Structure, Composition
    if a.raw_relaxed and not a.sun_labels: p.error('--raw-relaxed requires --sun-labels from the same archive')
    if a.sun_labels:
        labels=pd.read_csv(a.sun_labels)
    else:
        labels=pd.read_csv(a.data_dir/'pairs.csv')
        labels['cif_str_unrelaxed']=[(a.data_dir/f).read_text() for f in labels.before_cif]
        labels['cif_str']=[(a.data_dir/f).read_text() for f in labels.after_cif]
    raw=pd.read_csv(a.raw_relaxed,low_memory=False) if a.raw_relaxed else labels.copy()
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from record_ids import validate_label_join
    validate_label_join(raw,labels,a.id_column)
    raw=raw.set_index(a.id_column);labels=labels.set_index(a.id_column)
    validated=[]
    for candidate,formula in ((384,'ClO2'),(132,'Ti4CuNi3'),(710,'Tb3Tl')):
        if str(labels.loc[candidate,'SUN']).lower() not in ('true','1'): raise ValueError(f'Candidate {candidate} is not SUN in the supplied run')
        row=raw.loc[candidate]
        for phase,column in [('before','cif_str_unrelaxed'),('after','cif_str')]:
            value=str(row[column]).replace('\\n','\n')
            structure=Structure.from_str(value,fmt='cif')
            if structure.composition.reduced_composition != Composition(formula).reduced_composition:
                raise ValueError(f'Candidate {candidate} composition differs from the supplemental example {formula}')
            validated.append((candidate,phase,structure))
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for candidate,phase,structure in validated:
        folder=a.output_dir/f'seed13_request{candidate:04d}'
        folder.mkdir(exist_ok=True)
        structure.to(filename=str(folder/f'{phase}.cif'))
    print(a.output_dir)

if __name__=='__main__': main()
