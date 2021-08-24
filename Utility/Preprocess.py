import re
from typing import Union

import dgl
import numpy as np
import torch as th
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from transformers import AutoTokenizer, AutoModel, pipeline


def dgl_collate(batch):
    g, fp, pt, y = zip(*batch)
    return (
        dgl.batch(g),
        th.cat(fp),
        th.cat(pt),
        th.cat(y),
    )


def dgl_graph(graph, ndatakey="feat", edatakey="feat") -> dgl.DGLHeteroGraph:
    g = dgl.graph(tuple(graph["edge_index"]), num_nodes=graph["num_nodes"])
    if graph["edge_feat"] is not None:
        g.edata[edatakey] = th.from_numpy(graph["edge_feat"])
    if graph["node_feat"] is not None:
        g.ndata[ndatakey] = th.from_numpy(graph["node_feat"])
    return g


def get_fingerprint(mol: Union[Chem.Mol, str], r=3, nBits=2048, **kwargs) -> np.ndarray:
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=nBits, **kwargs)
    arr = np.zeros((0,), dtype=np.int8)
    # noinspection PyUnresolvedReferences
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


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
