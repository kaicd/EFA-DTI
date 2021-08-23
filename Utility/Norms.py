import dgl
import torch as th
from dgl.backend.pytorch.sparse import _gsddmm, _gspmm, gspmm, gsddmm
from torch import nn


class ReZero(nn.Module):
    def __init__(self):
        super(ReZero, self).__init__()
        self.g = nn.Parameter(th.zeros(1))

    def forward(self, x):
        return x * self.g


class GraphNorm(nn.Module):
    def __init__(self, norm_type, hidden_dim=300, print_info=None):
        super(GraphNorm, self).__init__()
        assert norm_type in ["bn", "gn", "ln", None]
        self.norm = None
        self.print_info = print_info
        if norm_type == "bn":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "ln":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "gn":
            self.norm = norm_type
            self.gnw = nn.Parameter(th.ones(hidden_dim))
            self.gnb = nn.Parameter(th.zeros(hidden_dim))

    def forward(self, x, batch_num_nodes):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(x)
        elif self.norm is None:
            return x
        batch_size = len(batch_num_nodes)
        batch_list = th.as_tensor(batch_num_nodes, dtype=th.long, device=x.device)
        batch_index = th.arange(batch_size, device=x.device).repeat_interleave(
            batch_list
        )
        batch_index = batch_index.view((-1,) + (1,) * (x.dim() - 1)).expand_as(x)
        mean = th.zeros(batch_size, *x.shape[1:], device=x.device)
        mean = mean.scatter_add_(0, batch_index, x)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)
        sub = x - mean
        std = th.zeros(batch_size, *x.shape[1:], device=x.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        # return sub / std
        return self.gnw * sub / std + self.gnb


class GraphNormAndProj(nn.Module):
    def __init__(self, d_in, d_out, act, dropout, norm_type):
        super(GraphNormAndProj, self).__init__()
        self.norm = GraphNorm(norm_type, d_in)
        self.activation = act()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_in, d_out)

    def forward(self, tensor, batch_num_graphs):
        x = self.norm(tensor, batch_num_graphs)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out(x)
        return x


class EdgeNorm(th.autograd.Function):
    r"""Apply normalization over signals of incoming edges.

    For a node :math:`i` of :math:`N`, head :math:`m`, EdgeNorm is an
    operation computing

    .. math::
        \textbf{\textit{a}}^i_m = \text{normalize}([l^{i,1}_m, ... , l^{i,N}_m])

        \text{normalize}(\textbf{\textit{x}})^j = g \cdot
        \frac{x^j - \mu_{x}}{\sigma_x} + b

    Adapted from dgl implementation of EdgeSoftmax:
    https://github.com/dmlc/dgl/blob/b36b6c268efb59b59046a74976067050141b1d6e/python/dgl/backend/pytorch/sparse.py#L180
    """

    @staticmethod
    def forward(ctx, gidx, score, eids=dgl.base.ALL):
        r"""
        Args:
            ctx: context to save cache intermediate values to
            gidx: graph index object
            score: edata shaped scores to normalize, first dimension should
                match length of eids
            eids: ids of edges to normalize

        Returns:
            edge score values normalized by destination node grouping.
        """
        # save graph to backward cache
        if not dgl.base.is_all(eids):
            gidx = gidx.edge_subgraph([eids], True).graph

        # graph statistics aggregated by target node: mu and stdev
        score_sums = _gspmm(gidx, "copy_rhs", "sum", None, score)[0]
        score_counts = _gspmm(gidx, "copy_rhs", "sum", None, th.ones_like(score))[0]
        means = score_sums / score_counts.clamp_min(1)
        residual = _gsddmm(gidx, "sub", score, means, "e", "v")
        var = th.pow(residual, 2)
        stdev = th.sqrt(_gspmm(gidx, "copy_rhs", "sum", None, var)[0] / score_counts)
        inv_stdev = 1.0 / stdev.clamp_min(1e-5)

        # rescale residuals
        normalized = _gsddmm(gidx, "mul", residual, inv_stdev, "e", "v")

        ctx.graph_cache = gidx  # cache non tensor obj
        ctx.save_for_backward(residual, inv_stdev)  # save tensors in ctx
        return normalized

    @staticmethod
    def backward(ctx, output_grad):
        """
        Args:
            ctx: cached context from forward pass
            output_grad: upstream derivatives, shape: (num_E,)

        Returns:
            As many elements as output_grad
        """
        gidx = ctx.graph_cache
        residual, inv_stdev = ctx.saved_tensors
        edge_counts = gspmm(gidx, "copy_rhs", "sum", None, th.ones_like(output_grad))
        dres = gsddmm(gidx, "mul", output_grad, inv_stdev, "e", "v")
        dinv_std = gspmm(gidx, "copy_rhs", "sum", None, output_grad * residual)
        dsqerr = dinv_std * -th.pow(inv_stdev, 2) * inv_stdev / 2 / edge_counts
        dres2 = 2 * gsddmm(gidx, "mul", residual, dsqerr, "e", "v")
        dresidual = -gspmm(gidx, "copy_rhs", "sum", None, dres + dres2)
        out = gsddmm(gidx, "add", dres + dres2, dresidual / edge_counts, "e", "v")

        return None, out, None


def edge_norm(gidx, scores, eids=dgl.base.ALL):
    return EdgeNorm.apply(gidx._graph, scores, eids)


class EdgeNormWithGainAndBias(th.nn.Module):
    """
    Edge normalization with gain and bias per head from Richter and
    Wattenhofer, 2020. https://arxiv.org/abs/2005.09561, adapted for graph
    input structures.
    """

    def __init__(self, nheads=1):
        super().__init__()
        # trainable gain and bias per head.
        self.gain = th.nn.Parameter(th.ones(nheads, 1))
        self.bias = th.nn.Parameter(th.zeros(nheads, 1))

    def forward(self, g, edge_scores, eids=dgl.base.ALL):
        eweights = edge_norm(g, edge_scores, eids)
        return self.gain * eweights + self.bias
