from typing import List

import pytorch_lightning as pl
import torch as th
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score

from Utility.Layers import MLP_IC, GraphNet


class EFA_DTI_Module(pl.LightningModule):
    def __init__(
        self,
        mol_dim: int = 196,
        mol_n_layers: int = 6,
        mol_n_heads: int = 6,
        mol_attn: str = "norm",
        act: str = "relu",
        attn_dropout: float = 0.1,
        dropout: float = 0.3,
        graph_norm_type: str = "gn",
        fp_dims: List = [2048, 512, 128],
        prottrans_dims: List = [2048, 1024, 512],
        output_dims: List = [2048, 512, 1],
        graph_pool: str = "deepset",
        lr: float = 2e-3,
        lr_anneal_epochs: int = 200,
        weight_decay: float = 1e-2,
        eps: float = 1e-16,
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
        g_emb = self.mol_enc(g)
        fp_emb = self.fingerprint_enc(fp)
        pt_emb = self.prottrans_enc(pt)
        yhat = self.output(th.cat([g_emb, fp_emb, pt_emb], -1)).squeeze()
        return yhat

    def sharing_step(self, batch, _=None):
        y = batch[-1]
        g, fp, pt, _ = batch
        yhat = self(g, fp, pt)
        return yhat, y

    def training_step(self, batch, _=None):
        yhat, y = self.sharing_step(batch)
        mse = F.mse_loss(yhat, y)
        self.log("train_mse", mse)
        return mse

    def validation_step(self, batch, _=None):
        y, yhat = self.sharing_step(batch)
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
                "valid_mse": th.as_tensor(F.mse_loss(yhat, y), device=self.device),
                "valid_ci": th.as_tensor(
                    concordance_index(y, yhat), device=self.device
                ),
                "valid_r2": th.as_tensor(r2_score(y, yhat), device=self.device),
            }
        )

    def configure_optimizers(self):
        optimizer = AdaBelief(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=float(self.hparams.weight_decay),
            eps=float(self.hparams.eps),
        )
        scheduler = {
            "scheduler": th.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=0.01,
                steps_per_epoch=self.num_training_steps,
                epochs=self.hparams.lr_anneal_epochs,
                anneal_strategy="cos",
            ),
            "reduce_on_plateau": False,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs
