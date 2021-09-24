import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from .efa_dti_data_module import EFA_DTI_DataModule
from .efa_dti_module import EFA_DTI_Module


def load_configuration():
    # Load configuration file(.yaml)
    with open("config/efa_dti_config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Set parameters
    project_params = cfg["project_params"]
    module_params = cfg["module_params"]
    dataset_params = cfg["dataset_params"]

    return project_params, module_params, dataset_params


def load_net_and_data(module_params, dataset_params):
    # Set model and dataloader
    net = EFA_DTI_Module(**module_params)
    data = EFA_DTI_DataModule(**dataset_params)
    return net, data


def load_trainer(project_params) -> pl.Trainer:
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
                filename="EFA_DTI_best_mse",
                monitor=project_params["monitor"],
                save_top_k=1,
                mode="min",
            ),
        ],
    )
    return trainer


def main():
    project_params, module_params, dataset_params = load_configuration()

    net, data = load_net_and_data(module_params, dataset_params)

    trainer = load_trainer(project_params)
    trainer.fit(net, data)


if __name__ == "__main__":
    main()
