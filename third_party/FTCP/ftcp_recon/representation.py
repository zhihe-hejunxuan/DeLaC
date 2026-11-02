from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from pymatgen.core import Element, Lattice, Structure


@dataclass(frozen=True)
class FTCPConfig:
    max_sites: int = 20
    max_elms: int = 4
    n_recip: int = 59
    pad_rows: int = 2
    reciprocal_radius: float = 1.297
    reciprocal_radius_fallback: float = 1.4
    niggli: bool = False


class FTCPCodec:
    """Original FTCP real/reciprocal matrix with a deterministic inverse.

    Rows contain element one-hot vectors, lattice, fractional coordinates,
    site occupancies, reciprocal-vector lengths, and elemental descriptors.
    Columns contain element slots followed by reciprocal-space samples.
    """

    def __init__(self, config: FTCPConfig = FTCPConfig(), data_dir: str | Path = "data"):
        self.config = config
        data_dir = Path(data_dir)
        self.elements = json.loads((data_dir / "elements.json").read_text())
        with (data_dir / "atom_init.json").open() as f:
            raw = json.load(f)
        self.element_features = {int(k): np.asarray(v, np.float32) for k, v in raw.items()}
        self.feature_dim = len(next(iter(self.element_features.values())))
        self.atomic_numbers = np.array([Element(e).Z for e in self.elements])
        self.z_to_row = {int(z): i for i, z in enumerate(self.atomic_numbers)}

    @property
    def shape(self) -> tuple[int, int]:
        c = self.config
        rows = len(self.elements) + 2 + 2 * c.max_sites + 1 + self.feature_dim + c.pad_rows
        cols = c.max_elms + c.n_recip
        return rows, cols

    @property
    def layout(self) -> dict[str, slice | int]:
        n, m, f = len(self.elements), self.config.max_sites, self.feature_dim
        return {
            "elements": slice(0, n), "abc": n, "angles": n + 1,
            "coords": slice(n + 2, n + 2 + m),
            "occupancy": slice(n + 2 + m, n + 2 + 2 * m),
            "g": n + 2 + 2 * m,
            "properties": slice(n + 3 + 2 * m, n + 3 + 2 * m + f),
        }

    def supports(self, structure: Structure) -> bool:
        zs = {int(site.specie.Z) for site in structure}
        return (len(structure) <= self.config.max_sites and
                len(zs) <= self.config.max_elms and
                all(z in self.z_to_row and z in self.element_features for z in zs) and
                structure.is_ordered)

    def encode(self, structure: Structure) -> np.ndarray:
        if self.config.niggli:
            structure = structure.get_reduced_structure()
        if not self.supports(structure):
            raise ValueError("structure exceeds max_sites/max_elms, is disordered, or has an unsupported element")
        c, lay = self.config, self.layout
        site_z = np.asarray([int(s.specie.Z) for s in structure])
        _, first = np.unique(site_z, return_index=True)
        slot_z = site_z[np.sort(first)]
        z_slot = {int(z): i for i, z in enumerate(slot_z)}
        slot_of_site = np.asarray([z_slot[int(z)] for z in site_z])

        base_rows = self.shape[0] - c.pad_rows
        real = np.zeros((base_rows, c.max_elms), np.float32)
        for slot, z in enumerate(slot_z):
            real[self.z_to_row[int(z)], slot] = 1.0
        real[lay["abc"], :3] = structure.lattice.abc
        real[lay["angles"], :3] = structure.lattice.angles
        real[lay["coords"].start:lay["coords"].start + len(structure), :3] = structure.frac_coords % 1.0
        for i, slot in enumerate(slot_of_site):
            real[lay["occupancy"].start + i, slot] = 1.0
        prop = np.stack([self.element_features[int(z)] for z in slot_z], axis=1)
        real[lay["properties"], :len(slot_z)] = prop

        recip = np.zeros((base_rows, c.n_recip), np.float32)
        rl = structure.lattice.reciprocal_lattice_crystallographic
        hkl, dist, _, _ = rl.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], c.reciprocal_radius, zip_results=False)
        if len(hkl) - 1 < c.n_recip:
            hkl, dist, _, _ = rl.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], c.reciprocal_radius_fallback, zip_results=False)
        keep = dist != 0
        hkl, dist = hkl[keep].astype(np.int16), dist[keep]
        order = np.lexsort((-hkl[:, 2], -hkl[:, 1], -hkl[:, 0], np.abs(hkl).sum(1)))[:c.n_recip]
        if len(order) < c.n_recip:
            raise ValueError("not enough reciprocal points; increase reciprocal_radius_fallback")
        hkl, dist = hkl[order], dist[order]
        site_prop = np.stack([self.element_features[int(z)] for z in site_z], axis=1)
        phase = np.einsum("ij,kj->ik", hkl, structure.frac_coords)
        recip[lay["g"], :] = dist
        recip[lay["properties"], :] = site_prop @ (np.pi * phase.T)
        return np.pad(np.concatenate([real, recip], axis=1), ((0, c.pad_rows), (0, 0)))

    def decode(self, matrix: np.ndarray, occupancy_threshold: float = 0.5) -> Structure | None:
        c, lay = self.config, self.layout
        x = np.asarray(matrix)
        element_scores = x[lay["elements"], :c.max_elms]
        slot_rows = element_scores.argmax(axis=0)
        slot_conf = element_scores.max(axis=0)
        # A slot is usable only when some site assigns nontrivial occupancy to it.
        occ = x[lay["occupancy"], :c.max_elms]
        active_sites = occ.max(axis=1) >= occupancy_threshold
        if not active_sites.any():
            return None
        slots = occ[active_sites].argmax(axis=1)
        if np.any(slot_conf[slots] <= 0):
            return None
        species = [self.elements[int(slot_rows[s])] for s in slots]
        abc = np.clip(x[lay["abc"], :3], 0.5, None)
        angles = np.clip(x[lay["angles"], :3], 10.0, 170.0)
        coords = x[lay["coords"], :3][active_sites] % 1.0
        try:
            lattice = Lattice.from_parameters(*abc, *angles)
            if not np.isfinite(lattice.matrix).all() or lattice.volume < 1e-3 or np.linalg.cond(lattice.matrix) > 1e8:
                return None
            return Structure(lattice, species, coords,
                             coords_are_cartesian=False, to_unit_cell=True)
        except Exception:
            return None

    def metadata(self) -> dict:
        return {"config": asdict(self.config), "shape": self.shape,
                "elements": self.elements, "feature_dim": self.feature_dim}
