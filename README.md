## Installation

### Download repository:
```sh
git clone https://github.com/kaicd/EFA-DTI.git && cd EFA-DTI
```

### Modify prefix:
1. Open KAICD.yaml
2. Change i to ii
   1. prefix: /home/lhs/anaconda3/envs/KAICD
   2. prefix: {your_anaconda3_path}/envs/{env_name}

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

### Raw data file format:
|**SMILES**|**SEQUENCE**|**IC50**|
|---|---|---|
|Cc1nc(CN2CCN(CC2)c2c...|MALIPDLAMETWLLL...|10000.0|
|...|...|...|
|...|...|...|
- **The IC50 unit must be unified(nM or uM or M).**
- The unit must be specified in the config file (dataset_params -> unit)

### Modify data file path:
1. Open config/efi_dti_config.yaml
   1. project params
      1. entity : wandb account name
      2. project : wandb project name
   2. dataset params
      1. save_path: Path to save model
      2. data_dir : Path to directory with raw data files and pre-processed data files
      3. data_name : Path to raw data files(.csv or .ftr)


## Runs
```sh
PYTHONPATH="./" python efa_dti/efa_dti_main.py
```
