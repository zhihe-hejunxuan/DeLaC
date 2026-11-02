"""Validate DPA single-point energies against the released fixed reference hull labels."""
from __future__ import annotations
import argparse, hashlib, json, math, os, time
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
STABLE_THRESHOLD=0.1


def atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_structure(value: str) -> Structure:
    from pymatgen.core import Structure
    return Structure.from_str(str(value).replace("\\n", "\n"), fmt="cif")


def predict_energies(frame: pd.DataFrame, model_path: Path, phase_dir: Path, save_every: int) -> pd.DataFrame:
    from deepmd.calculator import DP
    from pymatgen.io.ase import AseAtomsAdaptor
    checkpoint = phase_dir / "energy_checkpoint.csv"
    existing: dict[int, dict] = {}
    if checkpoint.exists():
        for record in pd.read_csv(checkpoint).to_dict("records"):
            existing[int(record["source_index"])] = record

    print(f"[MODEL] loading {model_path}", flush=True)
    t_load = time.perf_counter()
    calc = DP(model=str(model_path))
    model_load_seconds = time.perf_counter() - t_load
    print(f"[MODEL] loaded in {model_load_seconds:.3f}s; resume={len(existing)}/{len(frame)}", flush=True)

    records = list(existing.values())
    started = time.perf_counter()
    adaptor = AseAtomsAdaptor()
    pending_since_save = 0
    for position, row in frame.iterrows():
        source_index = int(row["source_index"])
        if source_index in existing:
            continue
        one_started = time.perf_counter()
        record = {"source_index": source_index, "status": "error", "error": None,
                  "Etot_pred_from_dpa": math.nan, "energy_seconds": math.nan}
        try:
            structure = parse_structure(row["cif_str"])
            atoms: Atoms = adaptor.get_atoms(structure)
            atoms.calc = calc
            record["Etot_pred_from_dpa"] = float(atoms.get_potential_energy())
            record["status"] = "ok"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["energy_seconds"] = time.perf_counter() - one_started
        records.append(record)
        existing[source_index] = record
        pending_since_save += 1
        if pending_since_save >= save_every or position + 1 == len(frame):
            pd.DataFrame(records).sort_values("source_index").to_csv(checkpoint, index=False)
            pending_since_save = 0
        if (position + 1) % 10 == 0 or position + 1 == len(frame):
            elapsed = time.perf_counter() - started
            new_done = sum(int(r["source_index"] not in {}) for r in records)
            print(f"[ENERGY] position={position+1}/{len(frame)} checkpoint_rows={len(records)} elapsed={elapsed:.1f}s", flush=True)

    result = pd.DataFrame(records).drop_duplicates("source_index", keep="last").sort_values("source_index")
    atomic_json(phase_dir / "energy_timing.json", {
        "model_load_seconds": model_load_seconds,
        "this_invocation_seconds": time.perf_counter() - started,
        "checkpoint_rows": len(result),
        "successful": int(result.status.eq("ok").sum()),
        "failed": int(result.status.ne("ok").sum()),
        "sum_per_structure_energy_seconds": float(result.energy_seconds.fillna(0).sum()),
    })
    return result


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    stable_true = y_true <= STABLE_THRESHOLD
    stable_pred = y_pred <= STABLE_THRESHOLD
    return {
        "n": int(len(y_true)),
        "mae_eV_per_atom": float(mean_absolute_error(y_true, y_pred)),
        "rmse_eV_per_atom": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "spearman_rho": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
        "stable_accuracy": float(np.mean(stable_true == stable_pred)),
        "stable_ratio_true": float(np.mean(stable_true)),
        "stable_ratio_pred": float(np.mean(stable_pred)),
        "stable_false_positive": int(np.sum(~stable_true & stable_pred)),
        "stable_false_negative": int(np.sum(stable_true & ~stable_pred)),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=REPO/'data/dpa_validation')
    p.add_argument('--model',type=Path,default=REPO/'mlff_choosing/dpa-3.1-mptrj.pth')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--save-every',type=int,default=10)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.output_dir.exists() and not a.resume: raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(a.data_dir/'reference_ehull.csv')
    if len(frame)!=4620 or frame.structure_id.duplicated().any(): raise ValueError('Expected 4620 distinct validation structures')
    frame['source_index']=np.arange(len(frame))
    frame['cif_str']=[(a.data_dir/f).read_text() for f in frame.cif_file]
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from evaluation.metrics.evaluator import MetricsEvaluator
    from evaluation.reference.presets import ReferenceMP2020Correction
    from evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher
    manifest={'model_sha256':sha256(a.model),'reference_labels_sha256':sha256(a.data_dir/'reference_ehull.csv'),'cif_content_sha256':hashlib.sha256(''.join(frame.cif_str).encode()).hexdigest(),'rows':len(frame)}
    manifest_path=a.output_dir/'inputs.json'
    if a.resume and manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Resume inputs/checkpoint do not match the previous run')
    atomic_json(manifest_path,manifest)
    energy=predict_energies(frame,a.model,a.output_dir,a.save_every)
    valid=frame.merge(energy,on='source_index',validate='one_to_one').query('status == "ok"').copy()
    evaluated=MetricsEvaluator.from_structures_and_energies(
        structures=[parse_structure(x) for x in valid.cif_str],
        energies=valid.Etot_pred_from_dpa.astype(float).tolist(),
        reference=ReferenceMP2020Correction(),structure_matcher=DefaultDisorderedStructureMatcher(),
        energy_correction_scheme=MaterialsProject2020Compatibility()).as_dataframe(metrics='all')
    if len(evaluated)!=len(valid): raise RuntimeError('Hull row count changed')
    valid['predicted_ehull_eV_per_atom']=evaluated.energy_above_hull.to_numpy(dtype=float)
    valid.drop(columns='cif_str').to_csv(a.output_dir/'predictions.csv',index=False)
    report=metrics(valid.reference_ehull_eV_per_atom.to_numpy(float),valid.predicted_ehull_eV_per_atom.to_numpy(float))
    report.update({'input_rows':len(frame),'failed_energy_rows':len(frame)-len(valid)})
    atomic_json(a.output_dir/'metrics.json',report)
    print(a.output_dir)

if __name__=='__main__': main()
