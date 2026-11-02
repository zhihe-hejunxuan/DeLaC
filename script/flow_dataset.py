"""Unconditional latent training data."""
import torch
from torch.utils.data import Dataset
class FlowFeatureDataset(Dataset):
    def __init__(self, pt_path, ef_type="norm"):
        self.features = torch.load(pt_path, map_location="cpu", weights_only=False)["flow_feature"].float()
    def __len__(self): return len(self.features)
    def __getitem__(self, index): return self.features[index], torch.tensor(0.)
