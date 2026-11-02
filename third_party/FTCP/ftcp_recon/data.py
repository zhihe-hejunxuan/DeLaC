from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset


def _parse_structure(value) -> Structure:
    if isinstance(value, Structure):
        return value
    if isinstance(value, dict):
        return Structure.from_dict(value)
    if not isinstance(value, str):
        raise TypeError(f"unsupported structure value: {type(value)}")
    text = value.strip()
    if text.startswith("{"):
        return Structure.from_dict(json.loads(text))
    return Structure.from_str(text, fmt="cif")


def load_structures(path: str | Path, structure_column: str = "structure") -> tuple[list[str], list[Structure], pd.DataFrame]:
    path = Path(path)
    if path.suffix in {".json", ".jsonl"}:
        frame = pd.read_json(path, lines=path.suffix == ".jsonl")
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    else:
        raise ValueError("dataset must be .csv, .json, .jsonl, .pkl or .pickle")
    if structure_column not in frame and structure_column == "structure" and "cif" in frame:
        structure_column = "cif"
    if structure_column not in frame:
        raise KeyError(f"missing structure column {structure_column!r}; columns={list(frame)}")
    ids, structures, keep = [], [], []
    id_col = next((x for x in ("material_id", "mp_id", "id") if x in frame), None)
    for i, value in enumerate(frame[structure_column]):
        try:
            structures.append(_parse_structure(value))
            ids.append(str(frame.iloc[i][id_col]) if id_col else str(i))
            keep.append(i)
        except Exception as exc:
            print(f"warning: skip row {i}: {exc}")
    return ids, structures, frame.iloc[keep].reset_index(drop=True)


class MatrixDataset(Dataset):
    def __init__(self, matrices: np.ndarray, indices: np.ndarray, properties: np.ndarray | None = None):
        self.x = torch.from_numpy(matrices[indices]).float()
        self.indices = torch.from_numpy(indices.astype(np.int64))
        self.properties = None if properties is None else torch.from_numpy(properties[indices]).float()

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        item = {"x": self.x[i], "index": self.indices[i]}
        if self.properties is not None:
            item["properties"] = self.properties[i]
            item["property_mask"] = torch.isfinite(self.properties[i])
            item["properties"] = torch.nan_to_num(item["properties"])
        return item
