"""Label loading shared by the paper's seed-isolated screening experiments."""
from pathlib import Path
import pandas as pd


def truth(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))


def load_split(data_root: Path, split: str) -> pd.DataFrame:
    folder = data_root / "evaluation" / split
    raw = pd.read_csv(folder / "relaxed.csv", low_memory=False)
    labels = pd.read_csv(folder / "sun_labels.csv", usecols=["dataset_id", "stable", "novel", "unique", "SUN"], low_memory=False)
    if raw.dataset_id.duplicated().any() or labels.dataset_id.duplicated().any():
        raise ValueError(f"duplicate dataset_id in {split}")
    frame = raw.merge(labels, on="dataset_id", how="left", validate="one_to_one", indicator=True)
    frame["label_valid"] = frame._merge.eq("both")
    frame = frame.drop(columns="_merge")
    frame["source_split"] = split
    if frame.source_seed.isin((13, 37, 42, 73)).any():
        raise ValueError("formal seed leaked into dataset")
    return frame

