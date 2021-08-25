import math

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import edge_softmax
from einops import rearrange
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import global_add_pool

from utility.norms import ReZero, GraphNormAndProj, EdgeNormWithGainAndBias

"""
----------------------
Graph Net Layers
----------------------
"""


class ActGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int, act: str = "relu"):
        super(ActGLU, self).__init__()
        self.proj = nn.Linear(d_in, d_out * 2)
        acts = {"gelu": nn.GELU, "relu": nn.ReLU}
        self.act = acts[act]()

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class GraphNetBlock(nn.Module):
    def __init__(
        self,
        d_in: int = 256,
        qk_dim: int = 64,
        v_dim: int = 64,
        n_heads: int = 8,
        dropout: float = 0.2,
        attn_weight_norm="norm",
        act: str = "relu",
        norm_type="gn",
    ):
        """
        Initialize a multi-headed attention block compatible with DGLGraph
        inputs. Given a fully connected input graph with self loops,
        is analogous to original Transformer.

        Args:
            d_in: input dimension
            qk_dim: head dimension
            n_heads: number of heads
            dropout: dropout probability
            attn_weight_norm: attention pooling method, 'norm' or 'softmax'
        """
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.n_heads = n_heads
        self.d_hidden = n_heads * v_dim

        self.attn_weight_norm = {
            "norm": EdgeNormWithGainAndBias(n_heads),
            "softmax": edge_softmax,
        }[attn_weight_norm]
        self.attn_dropout = nn.Dropout(dropout)

        def pwff():
            return nn.Sequential(
                ActGLU(d_in, d_in * 2, act),
                nn.Dropout(dropout),
                nn.Linear(d_in * 2, d_in),
            )

        self.node_rezero = ReZero()
        self.edge_rezero = ReZero()
        self.node_ff = pwff()
        self.edge_ff = pwff()
        self.node_ff2 = pwff()
        self.edge_ff2 = pwff()

        self.q_proj = nn.Linear(d_in, self.qk_dim * self.n_heads)
        self.k_proj = nn.Linear(d_in, self.qk_dim * self.n_heads)
        self.v_proj = nn.Linear(d_in, self.v_dim * self.n_heads)
        self.eq_proj = nn.Linear(d_in, self.qk_dim * self.n_heads)
        self.ek_proj = nn.Linear(d_in, self.qk_dim * self.n_heads)
        self.ev_proj = nn.Linear(d_in, self.v_dim * self.n_heads)

        self.mix_nodes = GraphNormAndProj(
            n_heads * v_dim, d_in, act, dropout, norm_type
        )

    def forward(self, g: dgl.DGLGraph, n: th.Tensor, e: th.Tensor):
        # convection
        n = n + self.node_rezero(self.node_ff(n))
        e = e + self.edge_rezero(self.edge_ff(e))

        # diffusion (attn)
        q = rearrange(self.q_proj(n), "b (h qk) -> b h qk", h=self.n_heads)
        k = rearrange(self.k_proj(n), "b (h qk) -> b h qk", h=self.n_heads)
        v = rearrange(self.v_proj(n), "b (h v) -> b h v", h=self.n_heads)
        eq = rearrange(self.eq_proj(e), "b (h qk) -> b h qk", h=self.n_heads)
        ek = rearrange(self.ek_proj(e), "b (h qk) -> b h qk", h=self.n_heads)
        ev = rearrange(self.ev_proj(e), "b (h v) -> b h v", h=self.n_heads)

        g.ndata.update({"q": q, "k": k, "v": v})
        g.edata.update({"eq": eq, "ek": ek, "ev": ev})

        g.apply_edges(fn.v_dot_u("q", "k", "n2n"))  # n2n
        g.apply_edges(fn.v_dot_e("q", "ek", "n2e"))  # n2e
        g.apply_edges(fn.e_dot_u("eq", "k", "e2n"))  # e2n
        if self.attn_weight_norm == "softmax":
            scale = math.sqrt(self.qk_dim)
            g.edata["n2n"] /= scale
            g.edata["n2e"] /= scale
            g.edata["e2n"] /= scale
        n2n_attn = self.attn_dropout(self.attn_weight_norm(g, g.edata["n2n"]))
        n2e_attn = self.attn_dropout(self.attn_weight_norm(g, g.edata["n2e"]))
        e2n_attn = self.attn_dropout(self.attn_weight_norm(g, g.edata["n2e"]))

        # aggregate normalized weighted values per node
        g.apply_edges(
            lambda edge: {
                "wv": n2n_attn * edge.src["v"]
                + n2e_attn * edge.data["ev"]
                + e2n_attn * edge.src["v"]
            }
        )
        g.update_all(fn.copy_e("wv", "wv"), fn.sum("wv", "z"))

        n = n + self.node_rezero(
            self.mix_nodes(g.ndata["z"].view(-1, self.d_hidden), g.batch_num_nodes())
        )

        # convection
        n = n + self.node_rezero(self.node_ff2(n))
        e = e + self.edge_rezero(self.edge_ff2(e))

        return g, n, e


class GraphNet(nn.Module):
    def __init__(
        self,
        features: int = 256,
        qk_dim: int = 32,
        v_dim: int = 64,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.2,
        attn_weight_norm: str = "norm",
        act: str = "relu",
        norm_type: str = "gn",
        pool_type: str = "deepset",
    ):
        super(GraphNet, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attn_layers = nn.ModuleList()
        self.atom_enc = AtomEncoder(features)
        self.bond_enc = BondEncoder(features)

        for _ in range(n_layers):
            self.attn_layers.append(
                GraphNetBlock(
                    d_in=features,
                    qk_dim=qk_dim,
                    v_dim=v_dim,
                    dropout=dropout,
                    act=act,
                    norm_type=norm_type,
                    attn_weight_norm=attn_weight_norm,
                )
            )

        if pool_type == "deepset":
            self.readout = DeepSet(features, features, dropout=dropout)
        elif pool_type == "mean_max":
            self.readout = MeanMaxPool(features * 2)

    def forward(self, g: dgl.DGLGraph):
        n = self.atom_enc(g.ndata["feat"])
        e = self.bond_enc(g.edata["feat"])
        for i in range(self.n_layers):
            g, n, e = self.attn_layers[i](g, n, e)
        out = self.readout(g, n)
        return out


class GINNet(nn.Module):
    def __init__(
        self,
        mol_dim: int = 78,
        hidden_dim: int = 32,
        output_dim: int = 128,
        n_layers: int = 5,
        act: str = "relu",
        dropout: float = 0.2,
    ):
        super(GINNet, self).__init__()
        acts = {"gelu": nn.GELU, "relu": nn.ReLU}
        self.act = acts[act]()
        self.dropout = dropout
        self.n_layers = n_layers

        self.conv_list, self.bn_list = [], []
        for i in range(self.n_layers):
            net = nn.Sequential(
                nn.Linear(mol_dim if i == 0 else hidden_dim, hidden_dim),
                self.act,
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv_list.append(net)
            self.bn_list.append(nn.BatchNorm1d(hidden_dim))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        for i in range(self.n_layers):
            x = self.bn_list[i](self.act(self.conv_list[i](x, edge_index)))
        x = F.dropout(
            self.act(self.fc(global_add_pool(x, batch))),
            p=self.dropout,
            training=self.training,
        )

        return x


"""
----------------------
MLP Layers
----------------------
"""


class MLP_IC(nn.Sequential):
    def __init__(
        self, *dims, norm: bool = True, dropout: float = 0.1, act: str = "relu"
    ):
        acts = {"gelu": nn.GELU, "relu": nn.ReLU}
        act = acts[act]
        l = []
        for i in range(len(dims) - 2):
            l.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    act(),
                    nn.BatchNorm1d(dims[i + 1]) if norm else nn.Identity(),
                    nn.Dropout(dropout),
                ]
            )
        l.append(nn.Linear(dims[-2], dims[-1]))
        super(MLP_IC, self).__init__(*l)


"""
----------------------
Graph Pooling Layers
----------------------
"""


class DeepSet(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super(DeepSet, self).__init__()
        self.glu = nn.Sequential(nn.Linear(d_in, d_in * 2), nn.GLU())
        self.agg = nn.Sequential(
            nn.BatchNorm1d(d_in), nn.Dropout(dropout), nn.Linear(d_in, d_out)
        )

    def forward(self, g, n):
        g.ndata["out"] = self.glu(n)
        readout = self.agg(dgl.readout_nodes(g, "out", op="sum"))

        return readout


class MeanMaxPool(nn.Module):
    def __init__(self, dim: int):
        super(MeanMaxPool, self).__init__()
        self.gain = nn.Parameter(th.ones(dim))
        self.bias = nn.Parameter(th.zeros(dim))

    def forward(self, g, n, key="out"):
        g.ndata[key] = n
        max = dgl.readout_nodes(g, key, op="max")
        mean = dgl.readout_nodes(g, key, op="mean")
        out = th.cat([max, mean], dim=-1)
        return out * self.gain + self.bias
