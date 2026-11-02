#!/usr/bin/env python3
"""Three-seed DeLaC paper run: prepare four paired arms, relax selected arms."""
from __future__ import annotations

import os, sys

import argparse, atexit, hashlib, importlib.util, json, os, signal, subprocess, threading, time, traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pymatgen.core import Structure

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CRYS = Path(os.environ.get("DELAC_PYTHON", sys.executable))
sys.path.insert(0,str(HERE))
from runtime_resources import missing_model_resources
WORKER = HERE / "joint_four_arm_stage1_worker.py"
SUN = REPO / "script/eval_sun_from_csv_cli.py"
ARTIFACT = REPO / "result/final_experiments/experiment_speed/frozen_i3_t2_hgb/i3_t2_hgb.joblib"
DEFAULT_OUT = REPO / "result/final_experiments/experiment_speed/delac_four_arm"
METHOD = "scaled_rmsnorm_empirical_gaussian"
SEEDS = (13, 37, 73)
ARMS = ("raw", "rule_filter", "learned_filter", "rule_union_learned")


def module(path):
    spec = importlib.util.spec_from_file_location("ordinary_two_worker", path)
    item = importlib.util.module_from_spec(spec); spec.loader.exec_module(item); return item


BENCH = module(HERE / "relaxation_workers.py")
EVENT = None; STOP = threading.Event()


def now(): return datetime.now().astimezone().isoformat(timespec="seconds")
def atomic(value, path):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)); tmp.replace(path)
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()
def event(kind, **fields):
    if EVENT is None: return
    with EVENT.open("a") as h: h.write(json.dumps({"time": now(), "event": kind, **fields}, ensure_ascii=False)+"\n"); h.flush(); os.fsync(h.fileno())
def heartbeat():
    while not STOP.wait(30): event("heartbeat")


def prepare(source: Path, destination: Path, arm: str):
    frame = pd.read_csv(source, low_memory=False); rows = []
    for _, row in frame.iterrows():
        try: structure = Structure.from_str(str(row.cif_str).replace("\\n", "\n"), fmt="cif")
        except Exception: continue
        item = row.to_dict(); item["original_request_id"] = item.get("request_id"); item["benchmark_num_atoms"] = len(structure); rows.append(item)
    if arm != "raw" and len(rows) != 1000: raise RuntimeError(f"{arm}: parseable={len(rows)} expected=1000")
    out = pd.DataFrame(rows); out["request_id"] = range(len(out)); destination.parent.mkdir(parents=True, exist_ok=True); out.to_csv(destination, index=False)
    return {"requested_denominator": 1000, "selected_for_relaxation": len(out), "path": str(destination), "sha256": sha256(destination)}


def relax_args(gpu, arm):
    return SimpleNamespace(dry_run=False, gpu=gpu, fmax=.05, max_steps=300, crystal_timeout_seconds=3600,
        serial_retry_timeout_seconds=600, memory_limit_mib=12288, memory_safety_mib=1024, monitor_interval=1.,
        omp_threads=1, dp_intra_threads=1, dp_inter_threads=1, sharding_strategy="round_robin",
        neighbor_cutoff=6., save_interval=1, configuration_name=f"ordinary_workers_2_{arm}")


def run_sun(relaxed: Path, directory: Path):
    metrics, labels = directory/"sun_relaxed.json", directory/"sun_relaxed_labels.csv"
    cmd = [str(CRYS), "-u", str(SUN), "--csv", str(relaxed), "--out_json", str(metrics),
           "--labels_csv", str(labels), "--etot_name", "E_tot_pred_dpa_raw"]
    began = time.perf_counter()
    with (directory/"sun.log").open("a") as h: subprocess.run(cmd, cwd=REPO, stdout=h, stderr=subprocess.STDOUT, check=True)
    labels_frame = pd.read_csv(labels); count = int(labels_frame.SUN.fillna(False).astype(str).str.lower().isin({"true","1","yes"}).sum())
    return {"sun_count": count, "sun_percent_requested": count/10, "denominator": 1000,
            "evaluation_seconds": time.perf_counter()-began, "metrics": str(metrics), "labels": str(labels)}


def audit(seeds, arms):
    errors = []
    frozen_subset = tuple(seed for seed in SEEDS if seed in seeds)
    if not seeds or tuple(seeds) != frozen_subset or len(set(seeds)) != len(seeds):
        errors.append(f"seeds must be a unique ordered subset of frozen seeds {SEEDS}")
    if not arms or any(x not in ARMS for x in arms) or len(set(arms)) != len(arms): errors.append("invalid/duplicate arm")
    paths = [WORKER, SUN, REPO/"mlff_choosing/dpa-3.1-mptrj.pth",
             Path(os.environ.get("DELAC_REFERENCE",str(REPO/"data-release/alex-mp/reference_MP2020correction.gz")))]
    if arms != ("raw",): paths += [ARTIFACT, ARTIFACT.parent/"deployment_spec.json"]
    errors.extend(missing_model_resources([METHOD],REPO))
    for path in paths:
        if not path.is_file(): errors.append(f"missing {path}")
    return {"status":"passed" if not errors else "failed", "errors":errors, "method":METHOD,
        "frozen_seeds":list(SEEDS), "requested_seeds":list(seeds), "prepared_arms":(["raw"] if arms==("raw",) else list(ARMS)), "default_relaxed_arms":list(arms), "batch_size":32,
        "target_requests":1000, "stage2":{"workers":2,"sharding":"round_robin","fmax":.05,"max_steps":300,
        "crystal_timeout_seconds":3600,"memory_limit_mib":12288}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--resume-run", type=Path)
    ap.add_argument("--seeds", default=",".join(map(str,SEEDS)))
    ap.add_argument("--arms", default=",".join(ARMS), help="Raw alone needs no filter; any filtered arm prepares the shared four-arm pool")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args(); seeds = tuple(int(x) for x in args.seeds.split(",") if x); arms = tuple(x for x in args.arms.split(",") if x)
    report = audit(seeds, arms)
    if args.audit_only: print(json.dumps(report,indent=2,ensure_ascii=False)); raise SystemExit(report["status"]!="passed")
    if report["status"] != "passed": raise SystemExit(json.dumps(report,ensure_ascii=False))
    root = args.resume_run.resolve() if args.resume_run else args.output_root.resolve()/("run_"+datetime.now().strftime("%Y%m%d_%H%M%S")); status_path=root/"pipeline_status.json"
    if args.resume_run:
        state=json.loads(status_path.read_text()); previous=list(state.get("requested_stage2_arms",[]))
        state.update(status="running",resumed_at=now(),resume_count=int(state.get("resume_count",0))+1,
                     requested_stage2_arms=list(dict.fromkeys(previous+list(arms))))
    else:
        root.mkdir(parents=True,exist_ok=False); atomic(report,root/"protocol_audit.json")
        state={"status":"running","started_at":now(),"root":str(root),"method":METHOD,"seeds":list(SEEDS),
               "prepared_arms":report["prepared_arms"],"requested_stage2_arms":list(arms),"seed_results":{}}
    global EVENT; EVENT=root/"controller_events.jsonl"; threading.Thread(target=heartbeat,daemon=True).start(); atexit.register(lambda:STOP.set())
    def stop(signum,_): event("signal",signal=signum); raise SystemExit(128+signum)
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP): signal.signal(sig,stop)
    atomic(state,status_path); event("controller_started",arms=arms)

    # Every seed gets a fresh, continuous Stage 1 and all four material pools.
    for seed in seeds:
        item=state["seed_results"].setdefault(str(seed),{"seed":seed,"arms":{}}); stage=root/f"seed_{seed}"/"stage1"
        if not (stage/"stage1_summary.json").is_file():
            item["status"]="preparing"; atomic(state,status_path)
            if stage.exists(): stage.replace(stage.parent/("stage1_incomplete_"+datetime.now().strftime("%Y%m%d_%H%M%S")))
            cmd=[str(CRYS),"-u",str(WORKER),"--method",METHOD,"--seed",str(seed),"--target","1000","--output-dir",str(stage)]
            cmd += ["--raw-only"] if arms==("raw",) else ["--artifact",str(ARTIFACT)]
            try:
                stage.parent.mkdir(parents=True,exist_ok=True)
                with (stage.parent/"stage1.log").open("a") as h: result=subprocess.run(cmd,cwd=REPO,stdout=h,stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError(f"stage1 exit={result.returncode}")
            except BaseException as exc:
                item.update(status="prepare_failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc()); atomic(state,status_path)
                if isinstance(exc,(KeyboardInterrupt,SystemExit)): raise
                continue
        item["stage1"]=json.loads((stage/"stage1_summary.json").read_text())
        if not set(arms)<=set(item["stage1"]["arms"]):
            raise RuntimeError("This run contains Raw-only Stage 1; start a fresh paired run for filtered arms")
        item["status"]="prepared"; atomic(state,status_path)
    if args.prepare_only:
        state.update(status="prepared",finished_at=now()); atomic(state,status_path); print(root); return

    # Relax the selected arms independently.
    for arm in ARMS:
        if arm not in arms: continue
        state["current_arm"]=arm; atomic(state,status_path)
        for seed in seeds:
            item=state["seed_results"][str(seed)]; arm_state=item["arms"].setdefault(arm,{})
            if arm_state.get("status")=="completed" or item.get("status")=="prepare_failed": continue
            directory=root/f"seed_{seed}"/arm; source=root/f"seed_{seed}"/"stage1"/arm/"selected_1000.csv"
            try:
                arm_state.update(status="running",started_at=arm_state.get("started_at",now())); atomic(state,status_path)
                fixed=directory/"accepted_for_relaxation.csv"; prep=prepare(source,fixed,arm)
                relax=BENCH.run_configuration(relax_args(args.gpu,arm),fixed,2,directory); sun=run_sun(Path(relax["merged_output"]),directory)
                stage1=json.loads((source.parent/"stage1_summary.json").read_text()); t=stage1["timing_seconds"]
                corrected=float(relax["T_stage2_corrected_seconds"])
                timing={"stage1_full_seconds":t["stage1_full"],"inference_seconds":t["inference_total"],
                        "T_stage2_corrected_seconds":corrected,
                        "system_e2e_corrected_seconds":t["stage1_full"]+corrected,
                        "compute_e2e_corrected_seconds":t["inference_total"]+corrected,
                        "diagnostic_resumed_wall_seconds":float(relax["wall_seconds"])}
                arm_state.update(status="completed",finished_at=now(),input=prep,stage1=stage1,relaxation=relax,sun=sun,timing=timing)
            except BaseException as exc:
                arm_state.update(status="failed",finished_at=now(),error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
                if isinstance(exc,(KeyboardInterrupt,SystemExit)): atomic(state,status_path); raise
            atomic(state,status_path)
    state["current_arm"]=None; requested=[state["seed_results"][str(s)]["arms"].get(a,{}).get("status") for a in arms for s in seeds]
    state.update(status="completed" if requested and all(x=="completed" for x in requested) else "completed_with_failures",finished_at=now())
    atomic(state,status_path); event("pipeline_finished",status=state["status"]); print(json.dumps(state,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
