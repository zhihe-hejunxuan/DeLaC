"""Paired before/after relaxation metrics for generated periodic crystals.

The RMSD implementation intentionally mirrors the public MatterGen evaluator:
the initial structure is refined, made primitive, LLL-reduced, and then aligned
to the relaxed structure with a permissive periodic StructureMatcher.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


MATTERGEN_MAX_RMSD = 0.5
RMSD_THRESHOLDS_ANGSTROM = (0.05, 0.1, 0.2)


def _read_structure(value: Any) -> Structure | None:
    if value is None or pd.isna(value):
        return None
    try:
        return Structure.from_str(str(value).replace("\\n", "\n"), fmt="cif")
    except Exception:
        return None


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value) if not pd.isna(value) else False


def preprocess_initial_like_mattergen(structure: Structure) -> Structure:
    """Replicate MatterGen's preprocessing of the pre-relaxation structure."""
    return (
        SpacegroupAnalyzer(structure)
        .get_refined_structure()
        .get_primitive_structure()
        .get_reduced_structure(reduction_algo="LLL")
    )


def _average_lattice(first: Lattice, second: Lattice) -> Lattice:
    parameters = (np.array(first.parameters) + np.array(second.parameters)) / 2
    return Lattice.from_parameters(*parameters)


def mattergen_rmsd_angstrom(
    relaxed: Structure, initial: Structure
) -> tuple[float, bool]:
    """Return MatterGen-compatible RMSD in Å and whether matcher alignment succeeded."""
    prepared_initial = preprocess_initial_like_mattergen(initial)
    matcher = StructureMatcher(
        ltol=0.5,
        stol=MATTERGEN_MAX_RMSD,
        angle_tol=10,
        primitive_cell=True,
        scale=False,
        attempt_supercell=True,
        allow_subset=False,
    )
    match = matcher.get_rms_dist(relaxed, prepared_initial)
    average_lattice = _average_lattice(relaxed.lattice, prepared_initial.lattice)
    normalization = (len(relaxed) / average_lattice.volume) ** (1 / 3)
    if match is None:
        return MATTERGEN_MAX_RMSD / normalization, False
    return float(match[0] / normalization), True


def _retention_match(
    first: Structure, second: Structure, matcher: StructureMatcher
) -> bool:
    try:
        if first.composition.reduced_composition != second.composition.reduced_composition:
            return False
        return bool(matcher.fit(first, second))
    except Exception:
        return False


def _space_group_number(structure: Structure) -> float:
    try:
        return float(SpacegroupAnalyzer(structure).get_space_group_number())
    except Exception:
        return math.nan


def _distribution(values: pd.Series | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    quantiles = np.quantile(array, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
    names = ("q00", "q10", "q25", "q50", "q75", "q90", "q95", "q99", "q100")
    result: dict[str, float | int | None] = {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }
    result.update({name: float(value) for name, value in zip(names, quantiles)})
    return result


def _fraction(mask: pd.Series | np.ndarray) -> float:
    array = np.asarray(mask, dtype=bool)
    return float(array.mean()) if len(array) else 0.0


def _column(row: pd.Series, name: str, default: Any = math.nan) -> Any:
    return row[name] if name in row.index else default


def compute_details(initial: pd.DataFrame, relaxed: pd.DataFrame, id_column: str) -> pd.DataFrame:
    if id_column not in initial or id_column not in relaxed:
        raise ValueError(f"Both inputs must contain id column {id_column!r}")
    if initial[id_column].duplicated().any() or relaxed[id_column].duplicated().any():
        raise ValueError(f"{id_column!r} must be unique in both inputs")

    initial = initial.copy()
    relaxed = relaxed.copy()
    initial["_join_id"] = initial[id_column].astype(str)
    relaxed["_join_id"] = relaxed[id_column].astype(str)
    relaxed_by_id = relaxed.set_index("_join_id", drop=False)

    strict_matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        allow_subset=False,
    )
    loose_matcher = StructureMatcher(
        ltol=0.3,
        stol=0.5,
        angle_tol=10,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        allow_subset=False,
    )

    details: list[dict[str, Any]] = []
    for _, initial_row in initial.iterrows():
        join_id = initial_row["_join_id"]
        result: dict[str, Any] = {
            id_column: initial_row[id_column],
            "paired_relaxed_row": join_id in relaxed_by_id.index,
            "initial_structure_ok": False,
            "relaxed_structure_ok": False,
            "structure_pair_ok": False,
            "analysis_error": "",
            "relax_done": False,
            "relax_converged": False,
            "relax_steps": math.nan,
            "relax_error": "",
            "initial_energy_dpa_raw": math.nan,
            "E_tot_pred_dpa_raw": math.nan,
            "initial_max_force_eV_per_A": math.nan,
            "final_max_force_eV_per_A": math.nan,
            "initial_volume_A3": math.nan,
            "final_volume_A3": math.nan,
            "volume_change_fraction": math.nan,
            "is_valid": False,
            "hull_exclude_reason": "",
            "dpa_done": False,
            "mattergen_rmsd_A": math.nan,
            "mattergen_rmsd_alignment_succeeded": False,
            "mattergen_rmsd_used_fallback": False,
            "energy_drop_eV_per_atom": math.nan,
            "retained_strict": False,
            "retained_loose": False,
            "space_group_preserved": False,
        }
        if join_id not in relaxed_by_id.index:
            result["analysis_error"] = "missing_relaxed_row"
            details.append(result)
            continue

        relaxed_row = relaxed_by_id.loc[join_id]
        initial_structure = _read_structure(_column(initial_row, "cif_str"))
        relaxed_structure = _read_structure(_column(relaxed_row, "cif_str"))
        result["initial_structure_ok"] = initial_structure is not None
        result["relaxed_structure_ok"] = relaxed_structure is not None

        for name in (
            "relax_done",
            "relax_converged",
            "relax_steps",
            "relax_error",
            "initial_energy_dpa_raw",
            "E_tot_pred_dpa_raw",
            "initial_max_force_eV_per_A",
            "final_max_force_eV_per_A",
            "initial_volume_A3",
            "final_volume_A3",
            "volume_change_fraction",
            "is_valid",
            "hull_exclude_reason",
            "dpa_done",
        ):
            result[name] = _column(relaxed_row, name, "" if "error" in name else math.nan)

        result["relax_done"] = _bool_value(result["relax_done"])
        result["relax_converged"] = _bool_value(result["relax_converged"])
        result["dpa_done"] = _bool_value(result["dpa_done"])
        result["is_valid"] = _bool_value(result["is_valid"])

        if initial_structure is None or relaxed_structure is None:
            result["analysis_error"] = "unparseable_initial_or_relaxed_cif"
            details.append(result)
            continue

        result["structure_pair_ok"] = True
        result["formula_initial"] = initial_structure.composition.reduced_formula
        result["formula_relaxed"] = relaxed_structure.composition.reduced_formula
        result["n_atoms_initial"] = len(initial_structure)
        result["n_atoms_relaxed"] = len(relaxed_structure)
        result["composition_preserved"] = (
            initial_structure.composition.reduced_composition
            == relaxed_structure.composition.reduced_composition
        )

        initial_energy = _finite_float(result["initial_energy_dpa_raw"])
        final_energy = _finite_float(result["E_tot_pred_dpa_raw"])
        energy_drop = initial_energy - final_energy
        result["energy_drop_eV"] = energy_drop
        result["energy_drop_eV_per_atom"] = (
            energy_drop / len(initial_structure)
            if math.isfinite(energy_drop) and len(initial_structure)
            else math.nan
        )

        initial_force = _finite_float(result["initial_max_force_eV_per_A"])
        final_force = _finite_float(result["final_max_force_eV_per_A"])
        result["max_force_drop_eV_per_A"] = initial_force - final_force
        result["max_force_final_over_initial"] = (
            final_force / initial_force
            if math.isfinite(initial_force)
            and math.isfinite(final_force)
            and initial_force > 0
            else math.nan
        )

        initial_volume = float(initial_structure.volume)
        final_volume = float(relaxed_structure.volume)
        result["volume_change_fraction_recomputed"] = final_volume / initial_volume - 1

        for index, axis in enumerate(("a", "b", "c")):
            before = float(initial_structure.lattice.abc[index])
            after = float(relaxed_structure.lattice.abc[index])
            result[f"initial_lattice_{axis}_A"] = before
            result[f"final_lattice_{axis}_A"] = after
            result[f"lattice_{axis}_change_fraction"] = after / before - 1
        for index, angle in enumerate(("alpha", "beta", "gamma")):
            before = float(initial_structure.lattice.angles[index])
            after = float(relaxed_structure.lattice.angles[index])
            result[f"initial_{angle}_deg"] = before
            result[f"final_{angle}_deg"] = after
            result[f"{angle}_change_abs_deg"] = abs(after - before)

        result["space_group_initial"] = _space_group_number(initial_structure)
        result["space_group_relaxed"] = _space_group_number(relaxed_structure)
        result["space_group_preserved"] = (
            math.isfinite(result["space_group_initial"])
            and result["space_group_initial"] == result["space_group_relaxed"]
        )
        result["retained_strict"] = _retention_match(
            initial_structure, relaxed_structure, strict_matcher
        )
        result["retained_loose"] = _retention_match(
            initial_structure, relaxed_structure, loose_matcher
        )
        try:
            rmsd, aligned = mattergen_rmsd_angstrom(relaxed_structure, initial_structure)
            result["mattergen_rmsd_A"] = rmsd
            result["mattergen_rmsd_alignment_succeeded"] = aligned
            result["mattergen_rmsd_used_fallback"] = not aligned
        except Exception as exc:
            result["mattergen_rmsd_A"] = math.nan
            result["mattergen_rmsd_alignment_succeeded"] = False
            result["mattergen_rmsd_used_fallback"] = False
            result["analysis_error"] = (
                f"mattergen_rmsd:{type(exc).__name__}:{exc}"
            )
        details.append(result)
    return pd.DataFrame(details)


def summarize(
    details: pd.DataFrame,
    initial_path: Path,
    relaxed_path: Path,
    max_steps: int,
) -> dict[str, Any]:
    pairs = details.loc[details["structure_pair_ok"].fillna(False).astype(bool)].copy()
    converged = pairs.loc[pairs["relax_converged"].fillna(False).astype(bool)].copy()
    rmsd = pairs["mattergen_rmsd_A"].dropna()
    aligned = pairs.loc[
        pairs["mattergen_rmsd_alignment_succeeded"].fillna(False).astype(bool)
    ]

    summary: dict[str, Any] = {
        "inputs": {
            "initial_csv": str(initial_path.resolve()),
            "relaxed_csv": str(relaxed_path.resolve()),
        },
        "definitions": {
            "mattergen_rmsd": {
                "unit": "angstrom",
                "initial_preprocess": "refined -> primitive -> LLL reduced",
                "matcher": {
                    "ltol": 0.5,
                    "stol": 0.5,
                    "angle_tol": 10,
                    "primitive_cell": True,
                    "scale": False,
                    "attempt_supercell": True,
                    "allow_subset": False,
                },
                "failed_alignment": (
                    "MatterGen-compatible fallback: "
                    "0.5 * (average_lattice_volume / relaxed_atom_count)^(1/3) Å"
                ),
            },
            "energy_drop_eV_per_atom": (
                "(initial_energy_dpa_raw - E_tot_pred_dpa_raw) / initial_atom_count"
            ),
            "strict_retention": "ltol=0.2, stol=0.3, angle_tol=5",
            "loose_retention": "ltol=0.3, stol=0.5, angle_tol=10",
            "stress": "not computed",
        },
        "counts": {
            "initial_rows": int(len(details)),
            "paired_relaxed_rows": int(details["paired_relaxed_row"].fillna(False).sum()),
            "structure_pairs": int(len(pairs)),
            "relax_done": int(details["relax_done"].fillna(False).sum()),
            "relax_converged": int(details["relax_converged"].fillna(False).sum()),
            "relax_errors": int(
                details["relax_error"].fillna("").astype(str).str.strip().ne("").sum()
            ),
            "dpa_final_energies": int(details["E_tot_pred_dpa_raw"].notna().sum()),
            "rmsd_values": int(rmsd.notna().sum()),
            "rmsd_alignments_succeeded": int(len(aligned)),
            "rmsd_fallbacks": int(
                pairs["mattergen_rmsd_used_fallback"].fillna(False).sum()
            ),
        },
        "rates_over_all_initial": {
            "paired_relaxed_row": _fraction(details["paired_relaxed_row"].fillna(False)),
            "structure_pair": _fraction(details["structure_pair_ok"].fillna(False)),
            "relax_done": _fraction(details["relax_done"].fillna(False)),
            "relax_converged": _fraction(details["relax_converged"].fillna(False)),
            "relax_error": _fraction(
                details["relax_error"].fillna("").astype(str).str.strip().ne("")
            ),
        },
        "rates_over_structure_pairs": {
            "relax_converged": _fraction(pairs["relax_converged"].fillna(False)),
            "mattergen_rmsd_alignment_succeeded": _fraction(
                pairs["mattergen_rmsd_alignment_succeeded"].fillna(False)
            ),
            "mattergen_rmsd_fallback": _fraction(
                pairs["mattergen_rmsd_used_fallback"].fillna(False)
            ),
            "retained_strict": _fraction(pairs["retained_strict"].fillna(False)),
            "retained_loose": _fraction(pairs["retained_loose"].fillna(False)),
            "space_group_preserved": _fraction(
                pairs["space_group_preserved"].fillna(False)
            ),
        },
        "rates_over_converged_structure_pairs": {
            "retained_strict": _fraction(converged["retained_strict"].fillna(False)),
            "retained_loose": _fraction(converged["retained_loose"].fillna(False)),
            "space_group_preserved": _fraction(
                converged["space_group_preserved"].fillna(False)
            ),
        },
        "mattergen_rmsd_A": {
            "all_with_fallback": _distribution(rmsd),
            "alignment_succeeded_only": _distribution(aligned["mattergen_rmsd_A"]),
            "threshold_fractions_all_with_fallback": {
                f"below_{threshold:g}_A": _fraction(rmsd < threshold)
                for threshold in RMSD_THRESHOLDS_ANGSTROM
            },
        },
        "energy_drop_eV_per_atom": {
            "distribution": _distribution(pairs["energy_drop_eV_per_atom"]),
            "fraction_energy_increased": _fraction(
                pairs["energy_drop_eV_per_atom"].dropna() < 0
            ),
            "fraction_drop_below_0_1": _fraction(
                pairs["energy_drop_eV_per_atom"].dropna() < 0.1
            ),
            "fraction_drop_below_0_5": _fraction(
                pairs["energy_drop_eV_per_atom"].dropna() < 0.5
            ),
            "fraction_drop_above_1_0": _fraction(
                pairs["energy_drop_eV_per_atom"].dropna() > 1.0
            ),
        },
        "relaxation_steps": {
            "all_completed": _distribution(pairs["relax_steps"]),
            "converged_only": _distribution(converged["relax_steps"]),
            "max_steps": int(max_steps),
            "fraction_hitting_max_steps": _fraction(
                pairs["relax_steps"].dropna() >= max_steps
            ),
        },
        "forces_eV_per_A": {
            "initial_max": _distribution(pairs["initial_max_force_eV_per_A"]),
            "final_max": _distribution(pairs["final_max_force_eV_per_A"]),
            "drop": _distribution(pairs["max_force_drop_eV_per_A"]),
            "initial_threshold_fractions": {
                f"below_{threshold:g}": _fraction(
                    pairs["initial_max_force_eV_per_A"].dropna() < threshold
                )
                for threshold in (0.1, 0.5, 1.0, 5.0)
            },
        },
        "volume_change_fraction": _distribution(
            pairs["volume_change_fraction_recomputed"]
        ),
        "absolute_lattice_change": {
            "a_fraction": _distribution(
                np.abs(pairs["lattice_a_change_fraction"].astype(float))
            ),
            "b_fraction": _distribution(
                np.abs(pairs["lattice_b_change_fraction"].astype(float))
            ),
            "c_fraction": _distribution(
                np.abs(pairs["lattice_c_change_fraction"].astype(float))
            ),
            "alpha_deg": _distribution(pairs["alpha_change_abs_deg"]),
            "beta_deg": _distribution(pairs["beta_change_abs_deg"]),
            "gamma_deg": _distribution(pairs["gamma_change_abs_deg"]),
        },
    }
    return summary


def summary_markdown(tag: str, summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    rates_all = summary["rates_over_all_initial"]
    rates_pairs = summary["rates_over_structure_pairs"]
    rates_converged = summary["rates_over_converged_structure_pairs"]
    rmsd = summary["mattergen_rmsd_A"]["all_with_fallback"]
    energy = summary["energy_drop_eV_per_atom"]["distribution"]
    steps = summary["relaxation_steps"]["converged_only"]
    initial_force = summary["forces_eV_per_A"]["initial_max"]
    final_force = summary["forces_eV_per_A"]["final_max"]
    volume = summary["volume_change_fraction"]

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    return f"""# {tag}：弛豫前后结构指标

## 样本与收敛

- 初始产物：{counts['initial_rows']}
- 成功配对结构：{counts['structure_pairs']}
- 弛豫收敛：{counts['relax_converged']}（all：{pct(rates_all['relax_converged'])}）
- 弛豫错误：{counts['relax_errors']}（all：{pct(rates_all['relax_error'])}）
- 达到最大步数比例：{pct(summary['relaxation_steps']['fraction_hitting_max_steps'])}

## MatterGen-compatible RMSD

- Mean：{rmsd.get('mean', math.nan):.6f} Å
- Median：{rmsd.get('q50', math.nan):.6f} Å
- P90 / P95：{rmsd.get('q90', math.nan):.6f} / {rmsd.get('q95', math.nan):.6f} Å
- Matcher 对齐成功：{counts['rmsd_alignments_succeeded']}
- MatterGen fallback：{counts['rmsd_fallbacks']}（{pct(rates_pairs['mattergen_rmsd_fallback'])}）

## 能量、优化与力

- 能量下降中位数：{energy.get('q50', math.nan):.6f} eV/atom
- 能量下降 P90：{energy.get('q90', math.nan):.6f} eV/atom
- 收敛样本优化步数中位数：{steps.get('q50', math.nan):.2f}
- 收敛样本优化步数 P90：{steps.get('q90', math.nan):.2f}
- 初始最大原子力中位数：{initial_force.get('q50', math.nan):.6f} eV/Å
- 最终最大原子力中位数：{final_force.get('q50', math.nan):.6f} eV/Å
- 体积相对变化中位数：{volume.get('q50', math.nan):.6f}

## 结构保持率

| 分母 | 严格标准 | 宽松标准 | 空间群保持 |
|---|---:|---:|---:|
| 全部成功配对结构 | {pct(rates_pairs['retained_strict'])} | {pct(rates_pairs['retained_loose'])} | {pct(rates_pairs['space_group_preserved'])} |
| 弛豫收敛的配对结构 | {pct(rates_converged['retained_strict'])} | {pct(rates_converged['retained_loose'])} | {pct(rates_converged['space_group_preserved'])} |

RMSD 使用 MatterGen 官方宽松 matcher；结构保持率使用项目的严格
`(ltol=0.2, stol=0.3, angle_tol=5)` 和宽松
`(ltol=0.3, stol=0.5, angle_tol=10)` 两套标准。应力未计算。
"""


def run(args: argparse.Namespace) -> None:
    initial_path = Path(args.initial)
    relaxed_path = Path(args.relaxed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = pd.read_csv(initial_path)
    relaxed = pd.read_csv(relaxed_path)
    details = compute_details(initial, relaxed, args.id_column)
    summary = summarize(details, initial_path, relaxed_path, args.max_steps)

    details_path = output_dir / f"{args.tag}.relaxation_metrics.csv"
    summary_path = output_dir / f"{args.tag}.relaxation_metrics.summary.json"
    markdown_path = output_dir / f"{args.tag}.relaxation_metrics.md"
    details.to_csv(details_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown_path.write_text(
        summary_markdown(args.tag, summary), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "details": str(details_path.resolve()),
                "summary": str(summary_path.resolve()),
                "markdown": str(markdown_path.resolve()),
                "counts": summary["counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute paired pre/post-relaxation metrics for generated crystals."
    )
    parser.add_argument("--initial", required=True, help="CSV with initial cif_str")
    parser.add_argument("--relaxed", required=True, help="CSV with relaxed cif_str and DPA fields")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="relaxation")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.set_defaults(func=run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
