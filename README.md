![GitHub Logo](https://socialify.git.ci/HenryZhou19/Pytorch-Template/image?description=1&font=Source%20Code%20Pro&forks=1&language=1&name=1&owner=1&pattern=Charlie%20Brown&stargazers=1&theme=Light)

# Pytorch Template
A simple template for Pytorch projects.

## Requirements
* './requirements.txt' is only a hint for the packages (especially the versions of python and torch) you need to install.
* Some common packages are not listed in the file.
* You may try different versions of the packages.

## Introduction to files & folders in this repo
### configs
* the folder for "xxx.yaml" files
    * some contains main configs for train or inference in a certain setting
    * others are used as inheritances specified by "config.additional" in main yaml files
* most of the things can be configured easily in "xxx.yaml" files

### data
* the folder for data(images, videos, texts, etc.) files, which will not be included by git

### outputs*
* folders for all outputs, which will not be included by git
* all folders start with "output" are for the same purpose
* the suffix may be used for distinguishing between different tasks, which can be set in "xxx.yaml" files as "info.output_dir"

### scripts
* the folder for bash scripts, which are used for the entrance of all main functions of the code
* common usage: "bash scripts/xxx.sh -args ..."
    * <u>"bash scripts/train.sh -d 0,1"</u> to train on GPU0 and GPU1 with config file "configs/train.yaml"
    * <u>"bash scripts/train.sh -c train_more -d 0"</u> to train on GPU0 with config file "configs/train_more.yaml"
    * <u>"bash scripts/train.sh -d 0,1,2,3 data.batch_size_per_rank=16"</u> to train on GPU0~3 with config file "configs/train.yaml", in which the default "batch_size_per_rank" will be changed to 16
    * ……

### src
* the folder for all python scripts except for train.py and inference.py
#### src/criterions
* nn.Module used for calculating the loss, metrics
* all inherit the CriterionBase class (src/criterions/modules/criterion_base.py)
* other useful functions for losses and metrics
#### src/datasets
* DataModule which creates and returns train/val/test datasets 
* all inherit the DataModuleBase class (src/datasets/modules/data_module_base.py)
* other useful functions for loading and saving multimedia data
#### src/gears
* Trainer and Tester providing all necessary features such as forward_one_iter, save_checkpoint, etc.
* all inherit the TrainerBase and TesterBase class (src/gears/modules/gear_base.py)
#### src/models
* nn.Module as the main models and sub-modules
* all the main models inherit the ModelBase class (src/models/modules/model_base.py)
* other useful functions
#### src/utils
* just utils
* some things in src/utils/misc.py may be handy
    * "DistMisc.is_main_process()" returns True if rank 0
    * "DistMisc.avoid_print_mess()" makes sure that all ranks print things in order
    * "LoggerMisc.block_wrapper(input)" prints the input with decorations
    * "TensorMisc.GradCollector(x)" collects the grad of Tensor x
    * "with TimeMisc.TimerContext(block_name):..." shows the time it takes to execute a particular block of code
    * ……
#### engine.py
* implementation of the main loop of the dataloader in train_one_epoch / evaluate / test

