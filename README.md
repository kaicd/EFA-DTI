## Installation

### Modify prefix:
1. Open KAICD.yaml
2. prefix: /home/lhs/anaconda3/envs/KAICD to prefix: {your_anaconda3_path}/envs/{env_name}

### Create a conda environment:
```sh
conda env create -f KAICD.yaml
```

### Activate the environment:
```sh
conda activate {env_name}
```

## Usage

### Create wandb account(Recommended):
https://wandb.ai/

### Modify data file path:
1. Open config/efi_dti_config.yaml
   1. project params
      1. entity : wandb account name
      2. project : wandb project name(customize)
   2. dataset params
      1. save_path: Path to save model
      2. data_dir : Path to directory with raw data files
      3. data_name : Path to raw data files(.csv or .ftr)


## Runs
```sh
python efa_dti/efa_dti_main.py
```
