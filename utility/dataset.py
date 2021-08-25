import os
import pickle
from typing import Callable

import pandas as pd
import torch as th
from ogb.utils import smiles2graph
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from utility.preprocess import (
    atom_features,
    dgl_graph,
    pyg_graph,
    smiles_to_graph,
    sequence_to_numerical,
    get_fingerprint,
    EmbedProt,
    pIC50_transform,
)


class EFA_DTI_Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        unit: str = "nM",
        reset: bool = False,
        device: int = 0,
        y_transform: Callable = pIC50_transform,
    ):
        self.data_dir = data_dir
        self.data_name = data_name
        self.unit = unit
        self.y_transform = y_transform

        data_path = os.path.join(self.data_dir, self.data_name)
        if self.data_name[-3:] == "ftr":
            self.data = pd.read_feather(data_path)
        elif self.data_name[-3:] == "csv":
            self.data = pd.read_csv(data_path)
        else:
            raise ValueError(f"Invalid File Format : {self.data_name[-4:]}")

        # Graphs(dgl_graph)
        graphs_path = os.path.join(
            self.data_dir, self.data_name[:-4] + "_ligand_graphs.pkl"
        )
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
        fingerprint_path = os.path.join(
            self.data_dir, self.data_name[:-4] + "_ligand_fingerprints.pkl"
        )
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
        prottrans_path = os.path.join(
            self.data_dir, self.data_name[:-4] + "_target_prottrans.pkl"
        )
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
        ic50 = self.data["IC50"][idx]

        # Preprocessed data
        g = self.ligand_graphs[smiles]
        fp = th.as_tensor(self.ligand_fps[smiles], dtype=th.float32).unsqueeze(0)
        pt = th.as_tensor(self.target_pts[sequence], dtype=th.float32).unsqueeze(0)
        y = th.as_tensor(ic50, dtype=th.float32).unsqueeze(0)

        if self.y_transform is not None:
            y = self.y_transform(y, self.unit)

        return g, fp, pt, y

    def __len__(self):
        return len(self.data)


class GIN_Dataset(InMemoryDataset):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        unit: str = "nM",
        y_transform: Callable = pIC50_transform,
    ):
        super(GIN_Dataset, self).__init__(data_dir)
        self.data_dir = data_dir
        self.data_name = data_name
        self.unit = unit
        self.y_transform = y_transform

        self.processed_path = os.path.join(self.data_dir, self.data_name[:-4] + ".pt")
        if os.path.exists(self.processed_path):
            print("Pre-processed data found: {}, loading ...".format(processed_path))
            self.data, self.slices = torch.load(processed_path)
        else:
            print(
                "Pre-processed data {} not found, doing pre-processing...".format(
                    self.processed_path
                )
            )
            self.process()
            self.data, self.slices = torch.load(self.processed_path)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.data_name[:-4] + ".pt"]

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self):
        data_path = os.path.join(self.data_dir, self.data_name)
        if self.data_name[-3:] == "ftr":
            self.data = pd.read_feather(data_path)
        elif self.data_name[-3:] == "csv":
            self.data = pd.read_csv(data_path)
        else:
            raise ValueError(f"Invalid File Format : {self.data_name[-4:]}")

        # Graphs(pyg_graph)
        graphs_path = os.path.join(
            self.data_dir, self.data_name[:-4] + "_pyg_ligand_graphs.pkl"
        )
        if not os.path.exists(graphs_path):
            print(
                f"{graphs_path} does not exist!\nProcessing SMILES to graphs...",
                flush=True,
            )
            self.ligand_graphs = {}
            smiles_list = set(list(self.data["SMILES"]))
            for s in tqdm(smiles_list):
                self.ligand_graphs[s] = smiles_to_graph(s)
            with open(graphs_path, "wb") as f:
                pickle.dump(self.ligand_graphs, f)
        else:
            print("Loading preprocessed graphs...", flush=True)
            with open(graphs_path, "rb") as f:
                self.ligand_graphs = pickle.load(f)

        # Protein(sequence)
        sequence_path = os.path.join(
            self.data_dir, self.data_name[:-4] + "_target_sequence.pkl"
        )
        if not os.path.exists(sequence_path):
            print(
                f"{graphs_path} does not exist!\nProcessing sequences to numerics...",
                flush=True,
            )
            self.protein_sequences = {}
            sequence_list = set(list(self.data["SEQUENCE"]))
            for s in tqdm(sequence_list):
                self.protein_sequences[s] = sequence_to_numerical(s)
            with open(sequence_path, "wb") as f:
                pickle.dump(self.protein_sequences, f)
        else:
            print("Loading preprocessed sequences...", flush=True)
            with open(sequence_path, "rb") as f:
                self.protein_sequences = pickle.load(f)

        data_list = []
        for i in tqdm(range(len(self.data))):
            smiles = self.data["SMILES"][i]
            target = self.data["SEQUENCE"][i]
            labels = self.data["IC50"][i]

            c_size, features, edge_index = self.ligand_graphs[smiles]
            target = self.protein_sequences[target]
            if self.y_transform is not None:
                labels = self.y_transform(labels, self.unit)

            data = pyg_graph(c_size, features, edge_index, target, labels)
            data_list.append(data)

        print("Graph construction done. Saving to file...", flush=True)
        data, slices = self.collate(data_list)
        th.save((data, slices), self.processed_path)
