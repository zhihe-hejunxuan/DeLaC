#!/usr/bin/env python3
"""One resident generator builds paired Raw/Rule/Learned/Union material pools."""
from __future__ import annotations

import os, sys

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/crystal_gen_flow_matplotlib")
sys.path[:0] = [str(REPO), str(HERE)]
import persistent_stage1_worker as generator
from filter_features import structure_features

BATCH = 32
FILTERED_ARMS = ("rule_filter", "learned_filter", "rule_union_learned")
ALL_ARMS = ("raw",) + FILTERED_ARMS
CRYS = Path(os.environ.get("DELAC_PYTHON", sys.executable))
SCORER = HERE / "score_frozen_hgb.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(value: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def score(frame: pd.DataFrame, spec: dict, artifact: Path) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    features = pd.DataFrame([structure_features(value, with_smact=True) for value in frame.cif_str])
    physical_features = time.perf_counter() - started
    enriched = pd.concat([frame.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    medians = pd.Series(spec["medians"])
    matrix = enriched[spec["features"]].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan).fillna(medians).fillna(0.)
    started = time.perf_counter()
    result = subprocess.run([str(CRYS), str(SCORER), "--artifact", str(artifact)],
                            input=matrix.to_json(orient="split"), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"frozen HGB scorer failed: {result.stderr[-2000:]}")
    risks = np.asarray(json.loads(result.stdout), dtype=float)
    hgb_seconds = time.perf_counter() - started
    enriched["hgb_risk"] = risks
    enriched["hgb_threshold"] = spec["threshold"]
    enriched["hgb_rejected"] = ~enriched.parseable.astype(bool) | (risks >= spec["threshold"])
    enriched["rule_rejected"] = (~enriched.parseable.astype(bool)
        | ~enriched.volume_per_atom_A3.between(3, 80)
        | enriched.lattice_aspect_ratio.gt(10)
        | enriched.min_distance_A.lt(.5)
        | ~enriched.density_g_cm3.between(.1, 30))
    enriched["union_rejected"] = enriched.rule_rejected | enriched.hgb_rejected
    base_seconds = float(pd.to_numeric(enriched.base_feature_seconds, errors="coerce").fillna(0).sum())
    smact_seconds = float(pd.to_numeric(enriched.smact_seconds, errors="coerce").fillna(0).sum())
    return enriched, {"physical_feature_seconds": physical_features,
                      "base_feature_seconds": base_seconds, "smact_seconds": smact_seconds,
                      "hgb_scoring_seconds": hgb_seconds,
                      "rule_filter_seconds": base_seconds,
                      "learned_filter_seconds": base_seconds + smact_seconds + hgb_seconds,
                      "union_filter_seconds": base_seconds + smact_seconds + hgb_seconds}


def accepted(frame: pd.DataFrame, arm: str) -> pd.Series:
    if arm == "raw": return frame.phase.eq("initial") & frame.generation_success.astype(bool)
    column = {"rule_filter": "rule_rejected", "learned_filter": "hgb_rejected",
              "rule_union_learned": "union_rejected"}[arm]
    return ~frame[column].astype(bool)


def needed_batches(pool: pd.DataFrame, arm: str, target: int) -> list[int]:
    selected = pool.loc[accepted(pool, arm)].head(target)
    if len(selected) != target: raise RuntimeError(f"{arm}: only {len(selected)} accepted")
    last = int(selected.candidate_order.max())
    return sorted(pool.loc[pool.phase.eq("refill") & pool.candidate_order.le(last), "refill_batch"].astype(int).unique().tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--artifact", type=Path)
    ap.add_argument("--raw-only", action="store_true", help="Prepare only Raw, without a filter or replacement candidates")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-candidates", type=int, default=10000)
    args = ap.parse_args()
    if args.target != 1000: raise SystemExit("formal protocol fixes target=1000")
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    spec_path = None
    spec = None
    if not args.raw_only:
        if args.artifact is None: ap.error("--artifact is required unless --raw-only is set")
        spec_path = args.artifact.parent / "deployment_spec.json"
        spec = json.loads(spec_path.read_text())
        recorded = Path(spec["artifact"])
        if recorded.name != args.artifact.name: raise RuntimeError("artifact/spec filename mismatch")
        expected_hash = spec.get("artifact_sha256")
        manifest_path = args.artifact.parent / "manifest.json"
        if expected_hash is None and manifest_path.is_file():
            expected_hash = json.loads(manifest_path.read_text()).get("artifact_sha256")
        if expected_hash:
            if sha256(args.artifact) != expected_hash: raise RuntimeError("artifact/spec checksum mismatch")
        elif (recorded if recorded.is_absolute() else spec_path.parent/recorded).resolve() != args.artifact.resolve():
            raise RuntimeError("Relocated legacy filter needs its checksum manifest")

    process_started = time.perf_counter()
    load_started = time.perf_counter(); sample, convert = generator.load_sampler(args.method, args.seed); generator.sync()
    model_load = time.perf_counter() - load_started
    frames, batches = [], []
    returned = 0
    for batch_id, start in enumerate(range(0, args.target, BATCH)):
        requested = min(BATCH, args.target - start)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        generator.sync(); began = time.perf_counter(); native = sample(requested); generator.sync(); inference = time.perf_counter() - began
        began = time.perf_counter(); structures = convert(native); conversion = time.perf_counter() - began
        frame = pd.DataFrame({"candidate_id": [f"initial_{i:06d}" for i in range(start, start + len(structures))],
            "generation_batch": batch_id, "generation_success": [x is not None for x in structures],
            "cif_str": [x.to(fmt="cif") if x is not None else None for x in structures],
            "phase": "initial", "candidate_order": np.arange(start, start + len(structures)), "refill_batch": -1})
        frames.append(frame); returned += len(frame)
        batches.append({"batch": batch_id, "phase": "initial", "requested": requested, "returned": len(frame),
                        "gpu_inference_seconds": inference, "native_to_cif_seconds": conversion,
                        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated()/1048576 if torch.cuda.is_available() else 0.})
    # Explicit failed-request placeholders retain the 1,000-request denominator.
    missing = args.target - returned
    if missing:
        start = returned
        frames.append(pd.DataFrame({"candidate_id": [f"initial_missing_{i:04d}" for i in range(missing)],
            "generation_batch": -1, "generation_success": False, "cif_str": np.nan,
            "phase": "initial", "candidate_order": np.arange(start, start + missing), "refill_batch": -1}))
    initial_plain = pd.concat(frames, ignore_index=True)
    initial_plain["candidate_order"] = np.arange(args.target)
    raw_dir = out / "raw"; raw_dir.mkdir()
    raw_write_started = time.perf_counter(); initial_plain.to_csv(raw_dir / "selected_1000.csv", index=False); raw_write = time.perf_counter() - raw_write_started
    raw_stage1 = time.perf_counter() - process_started
    if args.raw_only:
        initial_inference = sum(x["gpu_inference_seconds"] for x in batches)
        initial_conversion = sum(x["native_to_cif_seconds"] for x in batches)
        setup = max(0., raw_stage1-model_load-initial_inference-initial_conversion-raw_write)
        summary = {"status":"completed", "arm":"raw", "method":args.method, "seed":args.seed,
            "requested":args.target, "returned_rows":returned,
            "generation_success":int(initial_plain.generation_success.fillna(False).astype(bool).sum()),
            "missing_requests":missing, "selected_csv":str(raw_dir/"selected_1000.csv"),
            "timing_seconds":{"process_setup":setup,"model_load":model_load,
                "initial_inference":initial_inference,"initial_native_to_cif":initial_conversion,
                "selection_write":raw_write,"stage1_full":raw_stage1,"inference_total":initial_inference}}
        atomic(summary,raw_dir/"stage1_summary.json")
        master={"status":"completed","method":args.method,"seed":args.seed,"batch_size":BATCH,
                "target_requests":args.target,"arms":["raw"],"batch_records":batches,"arm_summaries":{"raw":summary}}
        atomic(master,out/"stage1_summary.json")
        print(json.dumps(master,indent=2)); return
    initial, initial_cost = score(initial_plain, spec, args.artifact)
    frames = [initial]
    refill_id = 0
    while True:
        pool = pd.concat(frames, ignore_index=True)
        counts = {arm: int(accepted(pool, arm).sum()) for arm in FILTERED_ARMS}
        if all(value >= args.target for value in counts.values()): break
        if len(pool) + BATCH > args.max_candidates: raise RuntimeError(f"max candidates exceeded: {counts}")
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        generator.sync(); began = time.perf_counter(); native = sample(BATCH); generator.sync(); inference = time.perf_counter() - began
        began = time.perf_counter(); structures = convert(native); conversion = time.perf_counter() - began
        start = len(pool)
        frame = pd.DataFrame({"candidate_id": [f"refill_{i:06d}" for i in range(start-args.target, start-args.target+len(structures))],
            "generation_batch": 32 + refill_id, "generation_success": [x is not None for x in structures],
            "cif_str": [x.to(fmt="cif") if x is not None else None for x in structures],
            "phase": "refill", "candidate_order": np.arange(start, start + len(structures)), "refill_batch": refill_id})
        frame, cost = score(frame, spec, args.artifact); frames.append(frame)
        batches.append({"batch": refill_id, "phase": "refill", "requested": BATCH, "returned": len(frame),
                        "gpu_inference_seconds": inference, "native_to_cif_seconds": conversion, **cost,
                        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated()/1048576 if torch.cuda.is_available() else 0.})
        refill_id += 1
        print(f"[REFILL] seed={args.seed} batches={refill_id} accepted={counts}", flush=True)
    pool = pd.concat(frames, ignore_index=True); pool_path = out / "shared_candidate_pool.csv"; pool.to_csv(pool_path, index=False)

    initial_inference = sum(x["gpu_inference_seconds"] for x in batches if x["phase"] == "initial")
    initial_conversion = sum(x["native_to_cif_seconds"] for x in batches if x["phase"] == "initial")
    setup = max(0., raw_stage1 - model_load - initial_inference - initial_conversion - raw_write)
    summaries = {}
    # Raw may contain fewer than 1,000 parseable structures, but its denominator remains 1,000 requests.
    raw_success = int(initial_plain.generation_success.fillna(False).astype(bool).sum())
    raw_summary = {"status": "completed", "arm": "raw", "method": args.method, "seed": args.seed,
        "requested": args.target, "returned_rows": returned, "generation_success": raw_success,
        "selected_for_relaxation_before_parse_check": raw_success, "missing_requests": missing,
        "selected_csv": str(raw_dir / "selected_1000.csv"), "timing_seconds": {
            "process_setup": setup, "model_load": model_load, "initial_inference": initial_inference,
            "initial_native_to_cif": initial_conversion, "selection_write": raw_write,
            "stage1_full": raw_stage1, "inference_total": initial_inference}}
    atomic(raw_summary, raw_dir / "stage1_summary.json"); summaries["raw"] = raw_summary

    cost_key = {"rule_filter": "rule_filter_seconds", "learned_filter": "learned_filter_seconds",
                "rule_union_learned": "union_filter_seconds"}
    for arm in FILTERED_ARMS:
        required = needed_batches(pool, arm, args.target)
        prefix = pool.loc[pool.phase.eq("initial") | pool.refill_batch.isin(required)].copy()
        ok = accepted(prefix, arm); selected = prefix.loc[ok].head(args.target).copy(); selected.insert(0, "request_id", range(args.target))
        selected_orders = set(selected.candidate_order.astype(int)); rejected = prefix.loc[~ok]; overflow = prefix.loc[ok & ~prefix.candidate_order.isin(selected_orders)]
        arm_dir = out / arm; arm_dir.mkdir(); began = time.perf_counter()
        selected.to_csv(arm_dir / "selected_1000.csv", index=False); rejected.to_csv(arm_dir / "rejected.csv", index=False)
        overflow.to_csv(arm_dir / "accepted_overflow.csv", index=False); prefix.to_csv(arm_dir / "decisions.csv", index=False)
        write_seconds = time.perf_counter() - began
        used = [x for x in batches if x["phase"] == "refill" and x["batch"] in required]
        refill_inference = sum(x["gpu_inference_seconds"] for x in used); refill_conversion = sum(x["native_to_cif_seconds"] for x in used)
        filter_seconds = initial_cost[cost_key[arm]] + sum(x[cost_key[arm]] for x in used)
        full = setup + model_load + initial_inference + initial_conversion + filter_seconds + refill_inference + refill_conversion + write_seconds
        summary = {"status": "completed", "arm": arm, "method": args.method, "seed": args.seed,
            "target": args.target, "selected": len(selected), "rejected_in_prefix": len(rejected), "overflow": len(overflow),
            "prefix_candidates": len(prefix), "required_refill_batches": required, "selected_csv": str(arm_dir / "selected_1000.csv"),
            "timing_seconds": {"process_setup": setup, "model_load": model_load, "initial_inference": initial_inference,
                "initial_native_to_cif": initial_conversion, "filter_total": filter_seconds,
                "refill_inference": refill_inference, "refill_native_to_cif": refill_conversion,
                "selection_write": write_seconds, "stage1_full": full, "inference_total": initial_inference + refill_inference}}
        atomic(summary, arm_dir / "stage1_summary.json"); summaries[arm] = summary
    master = {"status": "completed", "method": args.method, "seed": args.seed, "batch_size": BATCH,
        "target_requests": args.target, "arms": list(ALL_ARMS), "artifact": str(args.artifact.resolve()),
        "artifact_sha256": sha256(args.artifact), "deployment_spec_sha256": sha256(spec_path),
        "shared_pool": str(pool_path), "shared_pool_sha256": sha256(pool_path), "pool_rows": len(pool),
        "batch_records": batches, "arm_summaries": summaries,
        "protocol_note": "one resident model and one RNG stream; all four arms share initial 1000 requests and filtered arms share refill candidates"}
    atomic(master, out / "stage1_summary.json"); print(json.dumps(master, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
