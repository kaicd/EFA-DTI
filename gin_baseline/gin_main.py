import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from gin_data_module import GIN_DataModule
from gin_module import GIN_Module

# Load configuration file(.yaml)
with open("config/gin_config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Set parameters
project_params = cfg["project_params"]
module_params = cfg["module_params"]
dataset_params = cfg["dataset_params"]

# Set model and dataloader
net = GIN_Module(**module_params)
data = GIN_DataModule(**dataset_params)

# Set wandb
pl.seed_everything(project_params["seed"])
trainer = pl.Trainer(
    logger=WandbLogger(
        project=project_params["project"],
        entity=project_params["entity"],
        log_model=True,
    ),
    gpus=project_params["gpus"],
    accelerator=project_params["accelerator"],
    max_epochs=project_params["max_epochs"],
    callbacks=[
        EarlyStopping(
            patience=project_params["patience"], monitor=project_params["monitor"]
        ),
        LearningRateMonitor(logging_interval=project_params["logging_interval"]),
        ModelCheckpoint(
            dirpath=project_params["save_path"],
            filename="GIN_best_mse",
            monitor=project_params["monitor"],
            save_top_k=1,
            mode="min",
        ),
    ],
)
trainer.fit(net, data)
