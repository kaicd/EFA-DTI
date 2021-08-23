import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from EFA_DTI_DataModule import EFA_DTI_DataModule
from EFA_DTI_Module import EFA_DTI_Module

# Load configuration file(.yaml)
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Set parameters
project_params = cfg["project_params"]
module_params = cfg["module_params"]
dataset_params = cfg["dataset_params"]

# Set model and dataloader
net = EFA_DTI_Module(**module_params)
data = EFA_DTI_DataModule(**dataset_params)

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
    ],
)
trainer.fit(net, data)
