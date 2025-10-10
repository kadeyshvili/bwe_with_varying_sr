# Generative Adversarial Networks for Audio Super-Resolution with Varying Sample Rate

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#license">License</a>
</p>


## About

This repository contains the implementation for my diploma research on Generative Adversarial Networks for Audio Super-Resolution with Varying Sample Rate. In the current branch you can find implementation of NU-Wave blocks, MRF and classic HiFi++



## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## Usage


### Training a Model
The basic command for training model with NU-Wave blocks

```bash
python3 train.py -cn=train_config HYDRA_CONFIG_ARGUMENTS
```
The basic command for training model with MRF 

```bash
python3 train.py -cn=train_config_with_mrf HYDRA_CONFIG_ARGUMENTS
```

The basic command for training classic HiFi++ model without SpectralMaskNet

```bash
python3 train.py -cn=train_config_hifiplusplus HYDRA_CONFIG_ARGUMENTS
```
### Configuration

The model uses Hydra for configuration management. Key configuration parameters include:

- *datasets.train.dataset_split_file*: Path to training dataset split
- *datasets.val.dataset_split_file*: Path to validation dataset split
- *datasets.train.vctk_wavs_dir_lr*: Directory containing low-resolution training audio
- *datasets.val.vctk_wavs_dir_lr*: Directory containing low-resolution validation audio
- *datasets.train.vctk_wavs_dir_hr*: Directory containing high-resolution training audio
- *datasets.val.vctk_wavs_dir_hr*: Directory containing high-resolution validation audio
- *model.initial_sr*: Initial sample rate (e.g. 4000)
- *model.target_sr*: Traget sample rate (e.g. 16000)


### How to run inference
To evaluate a trained model:
```bash
python3 inference.py -cn="inference_config" inferencer.from_pretrained="path_to_pretrained_model" datasets.test.split=True datasets.test.vctk_wavs_dir=<path_to_dir_with_wavs> dataloader.test.batch_size=4 datasets.test.dataset_split_file=<path_to_split_file/test.txt> 
```



## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)