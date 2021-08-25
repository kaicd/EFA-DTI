import pytorch_lightning as pl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score

from utility.layers import GINNet, MLP_IC


class GIN_Module(pl.LightningModule):
    def __init__(
        self,
        mol_dim=78,
        hidden_dim=32,
        n_layers=5,
        prot_dim=25,
        embed_dim=128,
        in_channels=1000,
        out_channels=32,
        kernel_size=8,
        output_dims=[1024, 256, 1],
        dropout=0.2,
        act="relu",
        lr=0.0005,
        lr_anneal_epochs=500,
        weight_decay=1e-8,
        eps=1e-16,
        scheduler="Lambda",
    ):
        super(GIN_Module, self).__init__()
        self.save_hyperparameters()

        self.gin_conv = GINNet(
            mol_dim=mol_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            n_layers=n_layers,
            act=act,
            dropout=dropout,
        )
        self.embedding = nn.Embedding(prot_dim + 1, embed_dim)
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.conv_fc = nn.Linear(
            out_channels * (embed_dim - kernel_size + 1), embed_dim
        )
        self.output = MLP_IC(embed_dim * 2, *output_dims, dropout=dropout, act=act)

    def forward(self, data):
        x, edge_index, target, batch = data.x, data.edge_index, data.target, data.batch

        mol_emb = self.gin_conv(x, edge_index, batch)
        prot_emb = self.conv(self.embedding(target)).view(
            -1,
            self.hparams.out_channels
            * (self.hparams.embed_dim - self.hparams.kernel_size + 1),
        )
        yhat = self.output(th.cat[mol_emb, prot_emb], -1).squeeze()
        return yhat

    def sharing_step(self, data, _=None):
        y = data.y
        yhat = self(data)
        return yhat, y

    def training_step(self, data, _=None):
        yhat, y = self.sharing_step(data)
        mse = F.mse_loss(yhat, y)
        self.log("train_mse", mse)
        return mse

    def validation_step(self, batch, _=None):
        y, yhat = self.sharing_step(data)
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
        scheduler_type = {
            "Lambda": th.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: max(
                    1e-7, 1 - epoch / self.hparams.lr_anneal_epochs
                ),
            ),
            "OneCycle": th.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=0.01,
                steps_per_epoch=self.num_training_steps,
                epochs=self.hparams.lr_anneal_epochs,
                anneal_strategy="cos",
            ),
        }
        scheduler = {
            "scheduler": scheduler_type[self.hparams.scheduler],
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
