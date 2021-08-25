import re
from typing import Union

import dgl
import networkx as nx
import numpy as np
import torch as th
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from torch_geometric import data as DATA
from transformers import AutoTokenizer, AutoModel, pipeline


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "Mg",
                "Na",
                "Ca",
                "Fe",
                "As",
                "Al",
                "I",
                "B",
                "V",
                "K",
                "Tl",
                "Yb",
                "Sb",
                "Sn",
                "Ag",
                "Pd",
                "Co",
                "Se",
                "Ti",
                "Zn",
                "H",
                "Li",
                "Ge",
                "Cu",
                "Au",
                "Ni",
                "Cd",
                "In",
                "Mn",
                "Zr",
                "Cr",
                "Pt",
                "Hg",
                "Pb",
                "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(
            atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + one_of_k_encoding_unk(
            atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + [atom.GetIsAromatic()]
    )


def dgl_collate(batch):
    g, fp, pt, y = zip(*batch)
    return (
        dgl.batch(g),
        th.cat(fp),
        th.cat(pt),
        th.cat(y),
    )


def pyg_graph(c_size, features, edge_index, target, y):
    g = DATA.Data(
        x=th.Tensor(features),
        edge_index=th.LongTensor(edge_index).transpose(1, 0),
        y=th.FloatTensor([y]),
    )
    g.target = th.LongTensor([target])
    g.__setitem__("c_size", th.LongTensor([c_size]))
    return g


def dgl_graph(graph, ndatakey="feat", edatakey="feat") -> dgl.DGLHeteroGraph:
    g = dgl.graph(tuple(graph["edge_index"]), num_nodes=graph["num_nodes"])
    if graph["edge_feat"] is not None:
        g.edata[edatakey] = th.from_numpy(graph["edge_feat"])
    if graph["node_feat"] is not None:
        g.ndata[ndatakey] = th.from_numpy(graph["node_feat"])
    return g


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    c_size = mol.GetNumAtoms()
    features = []

    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index


def get_fingerprint(mol: Union[Chem.Mol, str], r=3, nBits=2048, **kwargs) -> np.ndarray:
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=nBits, **kwargs)
    arr = np.zeros((0,), dtype=np.int8)
    # noinspection PyUnresolvedReferences
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def sequence_to_numerical(seq: str) -> np.ndarray:
    seq_voc = "MALIPDETWVSYGHFKNCQRmgskpdqrclethfanvyiwxX2345678910Ubo-"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    max_seq_len = 1000
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(seq[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


def pIC50_transform(y: float, unit: str = "nM") -> th.Tensor:
    if unit == "nM":
        y = y / 1e9
    elif unit == "uM":
        y = y / 1e6
    elif unit == "M":
        pass
    else:
        raise ValueError("The exact unit must be input(nM, uM, M).")

    return -np.log10(y)


class EmbedProt:
    def __init__(self):
        super(EmbedProt, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Rostlab/prot_bert_bfd", do_lower_case=False
        )
        self.model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")

    def __call__(self, proteins, device=0):
        fe = pipeline(
            "feature-extraction",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )
        seqs = [" ".join(list(x)) for x in proteins]
        seqs = [re.sub(r"[UZOB]", "X", sequence) for sequence in seqs]
        embs = []
        for s in seqs:
            emb = np.array(fe([s])[0])  # (n, 1024)
            cls = emb[0]
            rest = emb[1:].mean(0)
            embs.append(np.concatenate([cls, rest]))
        return embs
