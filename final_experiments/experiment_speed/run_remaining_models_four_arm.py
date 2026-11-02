#!/usr/bin/env python3
"""Fresh four-arm E2E benchmark for the six remaining generators.

Execution is deliberately model-major.  For one model, prepare all three
seed-specific shared candidate streams first, then finish all four independent
Stage-2 arms, and only then advance to the next model.
"""
from __future__ import annotations

import os, sys

import argparse, atexit, importlib.util, json, os, signal, subprocess, threading, time, traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CRYS = Path(os.environ.get("DELAC_PYTHON", sys.executable))
sys.path.insert(0,str(HERE))
from runtime_resources import missing_model_resources
WORKER = HERE / "joint_four_arm_stage1_worker.py"
ARTIFACT = REPO / "result/final_experiments/experiment_speed/frozen_i3_t2_hgb/i3_t2_hgb.joblib"
DEFAULT_OUT = REPO / "result/final_experiments/experiment_speed/remaining_models_four_arm"
SEEDS = (13, 37, 73)
METHODS = ("mattergen_mp20", "cdvae", "diffcsp", "flowprep_scaled_rmsnorm_maf", "ftcp", "symat")
ARMS = ("raw", "rule_filter", "learned_filter", "rule_union_learned")
PYTHONS = {
    "mattergen_mp20": Path(os.environ.get("MATTERGEN_PYTHON", sys.executable)),
    "cdvae": Path(os.environ.get("CDVAE_PYTHON", sys.executable)),
    "diffcsp": Path(os.environ.get("DIFFCSP_PYTHON", sys.executable)),
    "flowprep_scaled_rmsnorm_maf": CRYS,
    "ftcp": CRYS,
    "symat": Path(os.environ.get("SYMAT_PYTHON", sys.executable)),
}


def load_local(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


BENCH = load_local("ordinary_two_worker", HERE / "relaxation_workers.py")
OURS = load_local("ours_four_arm_controller", HERE / "run_delac_four_arm.py")
EVENT_PATH = None
STOP = threading.Event()


class SunEvaluationTimeout(RuntimeError):
    """Raised after preserving a completed relaxation whose SUN audit timed out."""


def run_sun_with_timeout(relaxed: Path, directory: Path, timeout_seconds: int):
    metrics, labels = directory/"sun_relaxed.json", directory/"sun_relaxed_labels.csv"
    timeout_record = directory/"sun_timeout.json"
    cmd = [str(CRYS), "-u", str(REPO/"script/eval_sun_from_csv_cli.py"),
           "--csv", str(relaxed), "--out_json", str(metrics),
           "--labels_csv", str(labels), "--etot_name", "E_tot_pred_dpa_raw"]
    began = time.perf_counter()
    try:
        with (directory/"sun.log").open("a") as handle:
            subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - began
        record = {"status":"sun_timeout", "timeout_seconds":timeout_seconds,
                  "elapsed_seconds":elapsed, "relaxed_csv":str(relaxed),
                  "metrics":str(metrics), "labels":str(labels), "time":now()}
        atomic(record, timeout_record)
        raise SunEvaluationTimeout(
            f"SUN evaluation exceeded {timeout_seconds}s; relaxation preserved; {timeout_record}"
        ) from exc
    labels_frame = pd.read_csv(labels)
    count = int(labels_frame.SUN.fillna(False).astype(str).str.lower().isin({"true","1","yes"}).sum())
    return {"sun_count":count, "sun_percent_requested":count/10, "denominator":1000,
            "evaluation_seconds":time.perf_counter()-began,
            "metrics":str(metrics), "labels":str(labels)}


def now(): return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)); tmp.replace(path)


def event(kind, **fields):
    if EVENT_PATH is None: return
    try:
        with EVENT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": now(), "event": kind, **fields}, ensure_ascii=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
    except OSError:
        pass


def heartbeat():
    while not STOP.wait(30): event("heartbeat")


def ordered_subset(requested, frozen):
    return bool(requested) and len(set(requested)) == len(requested) and tuple(requested) == tuple(x for x in frozen if x in requested)


def audit(methods, seeds, arms, sun_timeout_seconds=3600):
    errors = []
    if not ordered_subset(methods, METHODS): errors.append(f"methods must be a unique ordered subset of {METHODS}")
    if not ordered_subset(seeds, SEEDS): errors.append(f"seeds must be a unique ordered subset of {SEEDS}")
    if not ordered_subset(arms, ARMS): errors.append(f"arms must be a unique ordered subset of {ARMS}")
    paths = [WORKER, REPO/"mlff_choosing/dpa-3.1-mptrj.pth",
             REPO/"script/eval_sun_from_csv_cli.py", HERE/"relaxation_workers.py",
             Path(os.environ.get("DELAC_REFERENCE",str(REPO/"data-release/alex-mp/reference_MP2020correction.gz")))]
    if arms != ("raw",): paths += [ARTIFACT, ARTIFACT.parent/"deployment_spec.json"]
    errors.extend(missing_model_resources(methods,REPO))
    paths += [PYTHONS[m] for m in methods]
    for path in paths:
        if not Path(path).exists(): errors.append(f"missing {path}")
    resources = {
        "mattergen_mp20": REPO/"final_experiments/experiment_3/vendor/mattergen-official/checkpoints/mp_20_base",
        "cdvae": REPO/"result/final_experiments/experiment_3/training/cdvae_mp20/epoch=144-step=15370.ckpt",
        "diffcsp": REPO/"final_experiments/experiment_3/vendor/diffcsp-official/pretrained/mp_gen",
        "flowprep_scaled_rmsnorm_maf": REPO/"result/MAF/no_cond/mp_20/h_rmsnorm/0.0003/512/model_best.pt",
        "ftcp": REPO/"third_party/FTCP/runs/mp20_ablation_B_niggli_properties/best.pt",
        "symat": REPO/"SyMat/result/mp_20/model_299.pth",
    }
    for method in methods:
        if not resources[method].exists(): errors.append(f"missing model resource {method}: {resources[method]}")
    if "cdvae" in methods:
        cdvae_dir = resources["cdvae"].parent
        for name in ("hparams.yaml", "lattice_scaler.pt", "prop_scaler.pt"):
            if not (cdvae_dir / name).is_file(): errors.append(f"missing CDVAE epoch-144 companion file: {cdvae_dir/name}")
    return {"status":"passed" if not errors else "failed", "errors":errors,
        "frozen_method_order":list(METHODS), "requested_methods":list(methods),
        "seeds":list(seeds), "arms":list(arms), "target_requests":1000, "batch_size":32,
        "stage1":"one resident model per model×seed; four arms share initial requests and filtered arms share refill stream",
        "stage2":{"independent_per_arm":True,"workers":2,"sharding":"round_robin","model":"DPA-3.1-MPtrj/DeePMD",
                  "optimizer":"FIRE+FrechetCellFilter","fmax":.05,"max_steps":300,
                  "crystal_timeout_seconds":3600,"memory_limit_mib":12288,
                  "sun_timeout_seconds":sun_timeout_seconds,"sun_timeout_action":"preserve relaxation and skip arm",
                  "primary_time_field":"T_stage2_corrected_seconds"},
        "models":{m:str(resources[m]) for m in methods}, "classifier_artifact":(str(ARTIFACT) if arms!=("raw",) else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--resume-run", type=Path)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", default=",".join(map(str,SEEDS)))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--sun-timeout-seconds", type=int, default=3600)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--prepare-only", action="store_true", help="prepare all selected model/seed pools without Stage 2")
    args = ap.parse_args()
    methods = tuple(x for x in args.methods.split(",") if x)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    arms = tuple(x for x in args.arms.split(",") if x)
    if args.sun_timeout_seconds <= 0: raise SystemExit("--sun-timeout-seconds must be positive")
    report = audit(methods, seeds, arms, args.sun_timeout_seconds)
    if args.audit_only:
        print(json.dumps(report, indent=2, ensure_ascii=False)); raise SystemExit(report["status"] != "passed")
    if report["status"] != "passed": raise SystemExit(json.dumps(report, ensure_ascii=False))

    root = args.resume_run.resolve() if args.resume_run else args.output_root.resolve()/("run_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    status_path = root/"pipeline_status.json"
    if args.resume_run:
        state = json.loads(status_path.read_text())
        state.update(status="running", resumed_at=now(), resume_count=int(state.get("resume_count",0))+1)
        state["requested_methods"] = list(dict.fromkeys(state.get("requested_methods",[])+list(methods)))
        state["requested_seeds"] = list(dict.fromkeys(state.get("requested_seeds",[])+list(seeds)))
        state["requested_arms"] = list(dict.fromkeys(state.get("requested_arms",[])+list(arms)))
    else:
        root.mkdir(parents=True, exist_ok=False); atomic(report, root/"protocol_audit.json")
        state = {"status":"running","started_at":now(),"root":str(root),"execution_order":"model -> all seed Stage1 -> all arm Stage2 -> next model",
                 "requested_methods":list(methods),"requested_seeds":list(seeds),"requested_arms":list(arms),"models":{}}
    global EVENT_PATH; EVENT_PATH=root/"controller_events.jsonl"
    threading.Thread(target=heartbeat,daemon=True).start(); atexit.register(lambda:STOP.set())
    def stop(signum,_): event("signal",signal=signum); atomic(state,status_path); raise SystemExit(128+signum)
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP): signal.signal(sig,stop)
    atomic(state,status_path); event("controller_started",resume=bool(args.resume_run),methods=methods,seeds=seeds,arms=arms)

    for method in methods:
        model_state = state["models"].setdefault(method,{"status":"running","seeds":{}})
        state["current_model"] = method; state["current_phase"] = "stage1"; atomic(state,status_path)
        # Complete every requested seed's shared four-arm Stage 1 before any Stage 2.
        for seed in seeds:
            seed_state = model_state["seeds"].setdefault(str(seed),{"seed":seed,"arms":{}})
            stage = root/method/f"seed_{seed}"/"stage1"
            if (stage/"stage1_summary.json").is_file():
                seed_state["stage1"] = json.loads((stage/"stage1_summary.json").read_text())
                if not set(arms)<=set(seed_state["stage1"]["arms"]):
                    raise RuntimeError("This run contains Raw-only Stage 1; start a fresh paired run for filtered arms")
                seed_state["status"]="prepared"; continue
            seed_state["status"]="preparing"; state["current_seed"]=seed; atomic(state,status_path)
            if stage.exists():
                archived=stage.parent/("stage1_incomplete_"+datetime.now().strftime("%Y%m%d_%H%M%S")); stage.replace(archived)
                event("stage1_incomplete_archived",method=method,seed=seed,path=str(archived))
            cmd=[str(PYTHONS[method]),"-u",str(WORKER),"--method",method,"--seed",str(seed),"--target","1000",
                 "--output-dir",str(stage)]
            cmd += ["--raw-only"] if arms==("raw",) else ["--artifact",str(ARTIFACT)]
            try:
                stage.parent.mkdir(parents=True,exist_ok=True); event("stage1_started",method=method,seed=seed)
                with (stage.parent/"stage1.log").open("a") as handle:
                    result=subprocess.run(cmd,cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError(f"stage1 exit={result.returncode}; see {stage.parent/'stage1.log'}")
                seed_state["stage1"]=json.loads((stage/"stage1_summary.json").read_text()); seed_state["status"]="prepared"
                event("stage1_completed",method=method,seed=seed)
            except BaseException as exc:
                seed_state.update(status="prepare_failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
                event("stage1_failed",method=method,seed=seed,error=str(exc))
                if isinstance(exc,(KeyboardInterrupt,SystemExit)): atomic(state,status_path); raise
            atomic(state,status_path)
        if args.prepare_only:
            seed_statuses = [model_state["seeds"].get(str(seed), {}).get("status") for seed in seeds]
            model_state["status"] = (
                "prepared" if all(status == "prepared" for status in seed_statuses)
                else "prepared_with_failures"
            )
            atomic(state,status_path); continue

        state["current_phase"]="stage2"; atomic(state,status_path)
        for arm in ARMS:
            if arm not in arms: continue
            state["current_arm"]=arm; atomic(state,status_path)
            for seed in seeds:
                state["current_seed"]=seed
                seed_state=model_state["seeds"][str(seed)]; arm_state=seed_state["arms"].setdefault(arm,{})
                # A timed-out SUN audit has already preserved a complete relaxation
                # and must stay skipped across controller restarts.  It can be
                # recomputed later from relaxed_merged.csv without blocking this run.
                if arm_state.get("status") in {"completed", "sun_timeout"}: continue
                if seed_state.get("status")=="prepare_failed":
                    arm_state.update(status="skipped",reason="Stage 1 failed"); atomic(state,status_path); continue
                directory=root/method/f"seed_{seed}"/arm
                selected=root/method/f"seed_{seed}"/"stage1"/arm/"selected_1000.csv"
                try:
                    arm_state.update(status="running",started_at=arm_state.get("started_at",now())); atomic(state,status_path)
                    fixed=directory/"accepted_for_relaxation.csv"; prep=OURS.prepare(selected,fixed,arm)
                    event("stage2_started",method=method,seed=seed,arm=arm)
                    relax=BENCH.run_configuration(OURS.relax_args(args.gpu,arm),fixed,2,directory)
                    stage1=json.loads((selected.parent/"stage1_summary.json").read_text()); timing1=stage1["timing_seconds"]
                    corrected=float(relax["T_stage2_corrected_seconds"])
                    timing={"stage1_full_seconds":timing1["stage1_full"],"inference_seconds":timing1["inference_total"],
                            "T_stage2_corrected_seconds":corrected,
                            "system_e2e_corrected_seconds":timing1["stage1_full"]+corrected,
                            "compute_e2e_corrected_seconds":timing1["inference_total"]+corrected,
                            "diagnostic_resumed_wall_seconds":float(relax["wall_seconds"])}
                    try:
                        sun=run_sun_with_timeout(Path(relax["merged_output"]),directory,args.sun_timeout_seconds)
                    except SunEvaluationTimeout as exc:
                        arm_started_at = arm_state.get("started_at", now())
                        arm_state.clear(); arm_state.update(
                            status="sun_timeout", started_at=arm_started_at, finished_at=now(), input=prep,
                            stage1=stage1, relaxation=relax, timing=timing,
                            sun={"status":"timeout", "timeout_seconds":args.sun_timeout_seconds,
                                 "record":str(directory/"sun_timeout.json")}, error=str(exc))
                        event("sun_timeout",method=method,seed=seed,arm=arm,
                              timeout_seconds=args.sun_timeout_seconds,
                              T_stage2_corrected_seconds=corrected)
                        atomic(state,status_path)
                        continue
                    arm_started_at = arm_state.get("started_at", now())
                    arm_state.clear(); arm_state.update(status="completed",started_at=arm_started_at,finished_at=now(),input=prep,
                                                        stage1=stage1,relaxation=relax,sun=sun,timing=timing)
                    event("stage2_completed",method=method,seed=seed,arm=arm,sun=sun["sun_count"],T_stage2_corrected_seconds=corrected)
                except BaseException as exc:
                    arm_state.update(status="failed",finished_at=now(),error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
                    event("stage2_failed",method=method,seed=seed,arm=arm,error=str(exc))
                    if isinstance(exc,(KeyboardInterrupt,SystemExit)): atomic(state,status_path); raise
                atomic(state,status_path)
        statuses=[a.get("status") for x in model_state["seeds"].values() for a in x.get("arms",{}).values() if a]
        model_state["status"]="completed" if statuses and all(x=="completed" for x in statuses) else "completed_with_failures"
        model_state["finished_at"]=now(); event("model_finished",method=method,status=model_state["status"]); atomic(state,status_path)

    state.update(current_model=None,current_seed=None,current_arm=None,current_phase=None)
    model_status=[state["models"][m].get("status") for m in methods]
    if args.prepare_only:
        final_status = "prepared" if model_status and all(x == "prepared" for x in model_status) else "prepared_with_failures"
    else:
        final_status = "completed" if model_status and all(x == "completed" for x in model_status) else "completed_with_failures"
    state.update(status=final_status,finished_at=now())
    atomic(state,status_path); event("pipeline_finished",status=state["status"])
    try: print(json.dumps(state,indent=2,ensure_ascii=False))
    except BrokenPipeError: pass


if __name__=="__main__": main()
