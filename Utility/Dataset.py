import os
import pickle
from typing import Union, Callable

import dgl
import numpy as np
import torch as th
from ogb.utils import smiles2graph
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from Utility.Preprocess import dgl_graph, get_fingerprints, EmbedProt


class Interactions(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        mol_featurizer: Callable = smiles2graph,
        reset=False,
        device=0,
        y_transform=None,
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
            raise ValueError("Invalid File Format.")

        # graphs
        graphs_path = os.path.join(data_dir, "ligand_graphs.pkl")
        if not os.path.exists(graphs_path) or reset:
            logging.info("Processing SMILES to graphs...")
            self.ligand_graphs = {}
            for s in self.data["SMILES"]:
                if not s in self.ligand_graphs:
                    self.ligand_graphs[s] = dgl_graph(mol_featurizer(s))
            with open(graph_path, "wb") as f:
                pickle.dump(self.ligand_graphs, f)
        else:
            logging.info("Loading preprocessed graphs...")
            with open(graph_path, "rb") as f:
                self.ligand_graphs = pickle.load(f)

        # fingerprint
        fingerprint_path = os.path.join(data_dir, "ligand_fingerprints.npy")
        if not os.path.exists(fingerprint_path) or reset:
            logging.info("processing fps...")
            self.ligand_fps = get_fingerprints(self.ligands.values())
            np.save(fingerprint_path, self.ligand_fps)
        else:
            self.ligand_fps = np.load(fingerprint_path)

        # ProtTrans(pt)
        pt_path = os.path.join(data_dir, "pt.npy")
        if not os.path.exists(pt_path) or reset:
            logging.info("processing proteins to pt...")
            pemb = EmbedProt()
            self.pt = pemb(self.proteins.values(), device=device)
            np.save(prottrans_path, self.pt)
        else:
            self.pt = np.load(prottrans_path)

    def __getitem__(self, item):
        d, p = self.ids[item]
        fp = th.as_tensor(self.ligand_fps[d], dtype=th.float32).unsqueeze(0)
        pt = th.as_tensor(self.pt[p], dtype=th.float32).unsqueeze(0)
        y = th.as_tensor(self.Y[d, p], dtype=th.float32).unsqueeze(0)

        if self.y_transform is not None:
            y = self.y_transform(y)

        g = self.ligand_graphs[d]
        g = dgl_graph(g)

        return g, fp, pt, y

    def __len__(self):
        return len(self.ids)
