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

# Alternative Approach

## Requirements

The only requirement to run this application is nvidia-docker. To install nvidia-docker
follow the guide found here:

- [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Setup

```sh
# Clone the repository
$ git clone https://github.com/kaicd/EFA-DTI.git && cd EFA-DTI
# Create the storage directory (will be later mounted into the Docker container)
$ mkdir -p storage/data
# Download the dataset
$ (cd storage/data && wget https://bindingdb-ic50-1m.s3.eu-central-1.amazonaws.com/BindingDB_IC50_filtering.ftr)
```

## Usage

Any code will be executed in the Docker container. We provide 3 scripts.

### dev.sh

The `dev/dev.sh` script will trigger the build of the Docker image before starting
a container with the image and running `/bin/bash` instead of the containers default
command. As soon as you are in the running container you should run
`conda activate KAICD` to enable the Conda environment. Afterwards you can run the
training using `python efa_dti/efa_dti_main.py`.

> **Note**
>
> Because of the code structure we need to set PYTHONPATH="./" which we do in the
> Dockerfile already while building the image.

### Adding dependencies

If you are planning to add a new dependency follow this process to ensure a env file
which is as small as possible.

1. Run `conda install my-dep=1.2.3` (replace my-dep accordingly)
2. Run `conda env export | grep my-dep` to get the exact string to copy
3. Add the line to the `KAICD.yaml` file

### train.sh

The `dev/train.sh` script will trigger the build of the Docker image before starting
a container with the image. The container will have the storage directory mounted into
the container so that dataset can be used. All preprocessed files will also land in
the storage directory and can be reused in multiple invocations.

### build.sh

The `dev/build.sh` script will trigger the build of the Docker image. All required
dependencies will be installed in the image using Conda. The Conda environment /
dependencies are specified in KAICD.yaml. If you like to install a new dependency use
the `dev/dev.sh` script.

## Weights & Biases

For logging we are using Weights & Biases. You will have to setup a wandb account.

### Create wandb account(Recommended)

https://wandb.ai/

## Data

### Raw data file format

|**SMILES**|**SEQUENCE**|**IC50**|
|---|---|---|
|Cc1nc(CN2CCN(CC2)c2c...|MALIPDLAMETWLLL...|10000.0|
|...|...|...|
|...|...|...|

- **The IC50 unit must be unified(nM or uM or M).**
- The unit must be specified in the config file (dataset_params -> unit)

### Modify data file path

1. Open config/efi_dti_config.yaml
   1. Project params
      1. entity : wandb account name
      2. project : wandb project name
   2. Dataset params
      1. save_path: Path to save model
      2. data_dir : Path to directory with raw data files and pre-processed data files
      3. data_name : Path to raw data files(.csv or .ftr)
