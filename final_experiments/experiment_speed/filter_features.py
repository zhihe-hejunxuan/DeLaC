"""Pre-relaxation features shared by paper screening experiments."""
import math, time
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element
FEATURES = ['n_atoms','volume_per_atom_A3','min_distance_A','min_lattice_A','max_lattice_A','min_angle_deg','max_angle_deg','lattice_aspect_ratio','n_elements','density_g_cm3','mean_Z','max_Z','mean_X','range_X','mean_atomic_radius']


def structure_features(value: object, with_smact: bool = True) -> dict:
    result = {name: np.nan for name in FEATURES}
    result.update({"parseable": False, "smact_valid": False, "feature_error": ""})
    t0 = time.perf_counter()
    try:
        structure = Structure.from_str(str(value).replace("\\n", "\n"), fmt="cif")
        n = len(structure)
        lengths = np.asarray(structure.lattice.abc, dtype=float)
        angles = np.asarray(structure.lattice.angles, dtype=float)
        distances = np.asarray(structure.distance_matrix, dtype=float).copy()
        np.fill_diagonal(distances, np.inf)
        composition = structure.composition.fractional_composition
        z_values, x_values, radius_values = [], [], []
        for element, fraction in composition.items():
            z_values.append((float(element.Z), float(fraction)))
            x = Element(element.symbol).X
            radius = Element(element.symbol).atomic_radius_calculated
            if x is not None:
                x_values.append((float(x), float(fraction)))
            if radius is not None:
                radius_values.append((float(radius), float(fraction)))
        result.update({
            "parseable": True, "n_atoms": n,
            "n_elements": len(composition.elements),
            "volume_per_atom_A3": float(structure.volume / n),
            "density_g_cm3": float(structure.density),
            "min_distance_A": float(distances.min()) if n > 1 else math.inf,
            "min_lattice_A": float(lengths.min()), "max_lattice_A": float(lengths.max()),
            "min_angle_deg": float(angles.min()), "max_angle_deg": float(angles.max()),
            "lattice_aspect_ratio": float(lengths.max() / lengths.min()),
            "mean_Z": sum(v * f for v, f in z_values),
            "max_Z": max(v for v, _ in z_values),
            "mean_X": sum(v * f for v, f in x_values) / sum(f for _, f in x_values) if x_values else np.nan,
            "range_X": max((v for v, _ in x_values), default=np.nan) - min((v for v, _ in x_values), default=np.nan),
            "mean_atomic_radius": sum(v * f for v, f in radius_values) / sum(f for _, f in radius_values) if radius_values else np.nan,
        })
        if with_smact:
            from evaluation.metrics.structure import is_smact_valid
            smact_started = time.perf_counter()
            try:
                result["smact_valid"] = bool(is_smact_valid(structure))
            except Exception as exc:
                result["smact_valid"] = False
                result["smact_error"] = f"{type(exc).__name__}: {exc}"
            result["smact_seconds"] = time.perf_counter() - smact_started
    except Exception as exc:
        result["feature_error"] = f"{type(exc).__name__}: {exc}"
        result["smact_seconds"] = 0.0
    result["base_feature_seconds"] = time.perf_counter() - t0 - result.get("smact_seconds", 0.0)
    return result

