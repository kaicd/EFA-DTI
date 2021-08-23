import math
import os
import pickle
from typing import Callable

import pandas as pd
import torch as th
from ogb.utils import smiles2graph
from torch.utils.data import Dataset
from tqdm import tqdm

from Utility.Preprocess import dgl_graph, get_fingerprint, EmbedProt


class EFA_DTI_Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        reset: bool = False,
        device: int = 0,
        y_transform: Callable = None,
    ):
        self.data_dir = data_dir
        self.data_name = data_name
        self.y_transform = y_transform

        data_path = os.path.join(self.data_dir, self.data_name)
        if self.data_name[-3:] == "ftr":
            self.data = pd.read_feather(data_path)
        elif self.data_name[-3:] == "csv":
            self.data = pd.read_csv(data_path)
        else:
            raise ValueError(f"Invalid File Format : {self.data_name[-4:]}")

        # Graphs(g_emb)
        graphs_path = os.path.join(data_dir, "ligand_graphs.pkl")
        if not os.path.exists(graphs_path) or reset:
            print(
                f"{graphs_path} does not exist!\nProcessing SMILES to graphs...",
                flush=True,
            )
            self.ligand_graphs = {}
            for s in tqdm(self.data["SMILES"]):
                if not s in self.ligand_graphs:
                    self.ligand_graphs[s] = dgl_graph(smiles2graph(s))
            with open(graphs_path, "wb") as f:
                pickle.dump(self.ligand_graphs, f)
        else:
            print("Loading preprocessed graphs...", flush=True)
            with open(graphs_path, "rb") as f:
                self.ligand_graphs = pickle.load(f)

        # Fingerprint(fp_emb)
        fingerprint_path = os.path.join(data_dir, "ligand_fingerprints.pkl")
        if not os.path.exists(fingerprint_path) or reset:
            print(
                f"{fingerprint_path} does not exist!\nProcessing SMILES to fingerprints...",
                flush=True,
            )
            self.ligand_fps = {}
            for s in tqdm(self.data["SMILES"]):
                if not s in self.ligand_fps:
                    self.ligand_fps[s] = get_fingerprint(s)
            with open(fingerprint_path, "wb") as f:
                pickle.dump(self.ligand_fps, f)
        else:
            print("Loading preprocessed fingerprints...", flush=True)
            with open(fingerprint_path, "rb") as f:
                self.ligand_fps = pickle.load(f)

        # ProtTrans(pt_emb)
        prottrans_path = os.path.join(data_dir, "target_prottrans.pkl")
        if not os.path.exists(prottrans_path) or reset:
            print(
                f"{prottrans_path} does not exist!\nProcessing proteins to ProtTrans embedding...",
                flush=True,
            )
            pemb = EmbedProt()
            self.target_pts = {}
            for s in tqdm(self.data["SEQUENCE"]):
                if not s in self.target_pts:
                    self.target_pts[s] = pemb([s], device=device)[0]
            with open(prottrans_path, "wb") as f:
                pickle.dump(self.target_pts, f)
        else:
            print("Loading preprocessed ProtTrans embedding...", flush=True)
            with open(prottrans_path, "rb") as f:
                self.target_pts = pickle.load(f)

    def __getitem__(self, idx):
        # Raw data
        smiles = self.data["SMILES"][idx]
        sequence = self.data["SEQUENCE"][idx]
        ic50 = math.log10(self.data["IC50"][idx])

        # Preprocessed data
        g = self.ligand_graphs[smiles]
        fp = th.as_tensor(self.ligand_fps[smiles], dtype=th.float32).unsqueeze(0)
        pt = th.as_tensor(self.target_pts[sequence], dtype=th.float32).unsqueeze(0)
        y = th.as_tensor(ic50, dtype=th.float32).unsqueeze(0)

        if self.y_transform is not None:
            y = self.y_transform(y)

        return g, fp, pt, y

    def __len__(self):
        return len(self.data)
