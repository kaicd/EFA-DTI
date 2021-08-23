import pytorch_lightning as pl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score

from Utility.Layers import MLP_IC, GraphNet


class EFA_DTI_Module(pl.LightningModule):
    def __init__(
        self,
        mol_dim=196,
        mol_n_layers=6,
        mol_n_heads=6,
        mol_attn="norm",
        act=nn.ReLU,
        attn_dropout=0.1,
        dropout=0.3,
        graph_norm_type="gn",
        fp_dims=(2048, 512, 128),
        prottrans_dims=(2048, 1024, 512),
        output_dims=(2048, 512, 1),
        graph_pool="deepset",
        batch_size=512,
        lr=2e-3,
        lr_anneal_epochs=200,
        weight_decay=1e-2,
        eps=1e-16,
        data_dir=None,
    ):
        super(EFA_DTI_Module, self).__init__()
        self.save_hyperparameters()

        self.mol_enc = GraphNet(
            features=mol_dim,
            qk_dim=int(mol_dim // mol_n_heads),
            v_dim=max(64, int(mol_dim // mol_n_heads)),
            n_layers=mol_n_layers,
            n_heads=mol_n_heads,
            dropout=attn_dropout,
            act=act,
            attn_weight_norm=mol_attn,
            norm_type=graph_norm_type,
            pool_type=graph_pool,
        )
        self.fingerprint_enc = MLP_IC(*fp_dims, dropout=dropout, act=act)
        self.prottrans_enc = MLP_IC(*prottrans_dims, dropout=dropout, act=act)
        outd = (
            mol_dim * (1 if graph_pool == "deepset" else 2)
            + fp_dims[-1]
            + prottrans_dims[-1]
        )
        self.output = MLP_IC(outd, *output_dims, dropout=dropout, act=act)

    def forward(self, g, fp, pt):
        mol = self.mol_enc(g)
        molfp = self.fingerprint_enc(fp)
        prottrans = self.prottrans_enc(pt)

        return self.output(th.cat([mol, molfp, prottrans], -1))

    def training_step(self, batch, _=None):
        y = batch[-1]
        g, fp, pt, _ = batch
        yhat = self(g, fp, pt).squeeze()
        mse = F.mse_loss(yhat, y)
        self.log("train_mse", mse)
        return mse

    def validation_step(self, batch, _=None):
        y = batch[-1]
        g, fp, pt, _ = batch
        yhat = self(g, fp, pt).squeeze()
        return {"yhat": yhat, "y": y}

    def validation_epoch_end(self, outputs):
        yhats = []
        ys = []
        for o in outputs:
            yhats.append(o["yhat"])
            ys.append(o["y"])
        yhat = th.cat(yhats).detach().cpu()
        y = th.cat(ys).detach().cpu()

        self.log_dict(
            {
                "val_mse": th.as_tensor(F.mse_loss(yhat, y), device=self.device),
                "ci": th.as_tensor(concordance_index(y, yhat), device=self.device),
                "r2": th.as_tensor(r2_score(y, yhat), device=self.device),
            }
        )

    def configure_optimizers(self):
        opt = AdaBelief(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.eps,
        )

        def lr_lambda(epoch):
            n = self.hparams.lr_anneal_epochs
            return max(1e-7, 1 - epoch / n)

        sched = {
            "scheduler": th.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda),
            "reduce_on_plateau": False,
            "interval": "epoch",
            "frequency": 1,
        }
        return [opt], [sched]
