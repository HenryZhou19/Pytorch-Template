![GitHub Logo](https://socialify.git.ci/HenryZhou19/Pytorch-Template/image?description=1&font=Source%20Code%20Pro&forks=1&language=1&name=1&owner=1&pattern=Charlie%20Brown&stargazers=1&theme=Light)


# Pytorch Template
A simple template for Pytorch projects.

## Easy Start
### Create a new Python environment and install the requirements
 ```
 conda create -n NEW_ENV_NAME python=3.9
 pip install -r requirements.txt
 ```
### Run the LeNet model to check if this template works well on your device
* Important notices: 
    * **Tensorboard** (offline logger) and **wandb** (online logger) are enabled by default.  
        So you may pre-register a wandb account to log in when the task is starting.

    * Loggers can be configured in 'yaml' config files, or you can disable all of them by running the bash command with:
        ```
        bash scripts/train.sh -d 0 -c template_train_lenet special.no_logger=True
        ```
    * Most of the configs can be modified in the bash command line by adding 'X.Y.Z...=...' intuitively.

* run on CPU, GPU and multiple GPUs by standalone DDP
    ```
    # Example(s):
    # CPU
    bash scripts/train.sh -d cpu -c template_train_lenet

    # GPU device_id=0
    bash scripts/train.sh -d 0 -c template_train_lenet

    # GPU device_id=0,1 (DDP)
    bash scripts/train.sh -d 0,1 -c template_train_lenet
    ```

* (optional) DDP across multiple machines 
    * Firstly, make sure all nodes (machines) have the same code files and have access to the same data files.
    * Secondly, enable password-free Secure Shell (ssh) communication between multiple machines which wilol be used later.
    * Thirdly, get the IP address of the master node (machine) and a free port, and call them MASTER_ADDR and MASTER_PORT respectively.
    ```
    # Example(s):
    # GPU device_id=0,1,2,3 on node 0 (master) and device_id=2,3 on node 1 (slave_1)
    # i.e. 4+2 GPUs

    # node 0 (master)
    bash scripts/multi_machine_train.sh -d 0,1,2,3 -a MASTER_ADDR -p MASTER_PORT -nn 2 -n 0 -c template_train_lenet

    # node 1 (slave_1)
    bash scripts/multi_machine_train.sh -d 2,3 -a MASTER_ADDR -p MASTER_PORT -nn 2 -n 1 -c template_train_lenet

    # '-a' means '--master_addr' in torchrun.
    # '-p' means '--master_port' in torchrun.
    # '-nn' means '--nnodes' in torchrun, which is 2 here for all nodes.
    # '-n' means '--node_rank' in torchrun, 0 for master and 1 for slave_1.
    ```

### outputs of the running task
* You can find the outputs where the path is defined in the corresponding 'yaml' file in './configs' (info.output_dir)
    * current_project files
    * checkpoints of the latest and the best model (the latest one includes the state of optimizer and scheduler etc., so it can be used to resume the task)
    * logs.txt with model's structure, summary...
    * ...

### Test the trained model if test_dataset exists
* get the absolute or relative (to the main path of this project) path of the cfg.yaml file in your training outputs as **TRAIN_CFG_PATH**
    ```
    bash scripts/inference.sh -d (cpu; 0; 0,1 ...) -p TRAIN_CFG_PATH
    # or just the same
    bash scripts/inference.sh -d (cpu; 0; 0,1 ...) tester.train_cfg_path=TRAIN_CFG_PATH
    ```
* Notes: 
    * It will use 'template_inference.yaml' as configs for inference by default.

    * The config 'tester.use_best' can control whether to use the best model or the latest model in the outputs folder.

    * Other configs will be the same as in **TRAIN_CFG_PATH** if not specified in the bash command or the inference 'yaml' file.

### Migrate or create your datasets, models, and tasks

#### ①These files and folders may not require any changes in most cases:
* **scripts/\***
* **train.py**; **inference.py**
* **src/engine.py**
* **src/utils/\*** (except for **src/utils/optimizer/\***, if you need some special optimizers and lr_schedulers)
* **src/datasets/modules/\*** (you may add some new files); **src/datasets/\_\_init\_\_.py**
* **src/models/modules/\*** (you may add some new files); **src/models/\_\_init\_\_.py**
* **src/criterions/modules/\*** (you may add some new files); **src/criterions/\_\_init\_\_.py**
* **src/gears/modules/\*** (you may add some new files); **src/gears/\_\_init\_\_.py**

#### ②Prepare for datasets:
1. Create a new 'py' file for your dataset (named ImageNet for example) in **src/datasets** as **src/datasets/imagenet_dataset.py**.

* In your new file, do the following steps just as in **src/datasets/template_dataset.py**.

2. Create a torch.utils.data.Dataset class for your Dataset, make sure the **\_\_getitem\_\_** returns a sample in the form of a nested dict like this:
    ```
    class ImageNetDataset(Dataset)
        def __init__(self, ...)
            super().__init__()
            ...

        def __len__(self):
            ...

        def __getitem__(self, index)
            ...
            return {
                'inputs': {
                    'INPUT_1_NAME': xxx,
                    'INPUT_2_NAME': xxx,
                    ...
                },
                'targets': {
                    'TARGET_1_NAME': xxx,
                    'TARGET_2_NAME': xxx,
                    ...
                },
            }

        ...
    ```
    * **'xxx'** above can be one of [torch.Tensor, np.ndarray, str, dict, int, float, bool]
    * **'inputs'** will go into the **Model** by default (can be modified in your own **Gear**)
    * **'targets'** will go into the **Criterion** by default, together with the **output dict of the Model** (can be modified in your own **Gear**)
    
3. Create a DataModule to make a connection between your Dataset and this template like this:
    * 'cfg', including all configs, will be in DataModule's \_\_init\_\_ method
    * 'cfg.data.dataset=DATASET_NAME' chooses the dataset as @data_module_register('DATASET_NAME')
    ```
    @data_module_register('imagenet')
    class ImageNetDataModule(DataModuleBase):
        def __init__(self, cfg, ...)
            super().__init__(cfg)
            # use 'cfg' to make some necessary preparations for your Dataset
            # or just save it by 'self.cfg=cfg' to leave this preparation procedure to the following methods
            ...

        def build_train_dataset(self)
            return Dataset(args_for_train)

        def build_val_dataset(self):
            return Dataset(args_for_val)
            
        def build_test_dataset(self):
            # if this dataset has test split...
            # it will be used in 'inference.py'
            return Dataset(args_for_test)
            
        ...
    ```

#### ③Prepare for models:
1. Create a new 'py' file for your model (named ResNet for example) in **src/models** as **src/models/resnet_model.py**.

* In your new file, do the following steps just as in **src/models/template_model.py**.

2. Create a Model(ModelBase) class for your Model like this:
    * 'cfg', including all configs, will be in Model's \_\_init\_\_ method
    * 'cfg.model.model_choice=MODEL_NAME' chooses the Model as @model_register('MODEL_NAME')
    ```
    @model_register('resnet')
    class ResNetModel(ModelBase):
        def __init__(self, cfg):
            super().__init__(cfg)
            ...

        def forward(self, inputs: dict) -> dict:
            x = inputs['INPUT_1_NAME']
            y = inputs['INPUT_2_NAME']

            ...

            return {
                'OUTPUT_1_NAME': xx
                'OUTPUT_2_NAME': yy
                ...
            }

        ...
    ```

#### ④Prepare for criterions (losses and metrics):
1. Create a new 'py' file for your criterion (still named ResNet for example) in **src/criterions** as **src/criterions/resnet_criterion.py**.

* In your new file, do the following steps just as in **src/criterions/template_criterion.py**.

2. Create a Criterion(CriterionBase) class for your Model, make sure the **\_\_getitem\_\_** returns a sample in the form of a nested dict like this:
    * 'cfg', including all configs, will be in Criterion's \_\_init\_\_ method
    * 'cfg.criterion.criterion_choice=CRITERION_NAME' chooses the ModelBase as @criterion_register('CRITERION_NAME')
        * if cfg.criterion.criterion_choice is not specified (default),
        **CRITERION_NAME=MODEL_NAME**
    ```
    @criterion_register('resnet')
    class ResNetCriterion(CriterionBase):
        def __init__(self, cfg):
            super().__init__(cfg)
            ...
            # if there's some metrics that can not be calculated by averaging the metrics of one sample
            # for example: accuracy
            self.epoch_correct_count = 0  # init
            self.epoch_sample_count = 0  # init
            
        def forward(self, outputs, targets, infer_mode=False):
            super().forward(outputs, targets, infer_mode)

            output_1 = outputs['OUTPUT_1_NAME']
            ...
            target_1 = targets['TARGET_1_NAME']   
            ...

            loss = ...  # no need to backward here
            self.epoch_correct_count += ...
            self.epoch_sample_count += ...
            ...

            return loss, {
                # all metrics here are scalars (float, torch's scalar, numpy's scalar) 
                # referring to one sample with reduction='mean'
                # all DDP things will be done outside for these metrics
                'METRIC_1_NAME': metric_1  
                'METRIC_2_NAME': metric_2
                ...
            }  # loss, metrics_dict

        # if some metrics can not be calculated by averaging the metrics of one sample (called automatically when an epoch ends)
        def get_epoch_metrics_and_reset(self):
            import torch.distributed as dist
            from src.utils.misc import DistMisc

            # get all results from DDP if needed
            if DistMisc.is_dist_avail_and_initialized():
                self.epoch_sample_count = torch.tensor(self.epoch_sample_count).cuda()
                self.epoch_correct_count = torch.tensor(self.epoch_correct_count).cuda()
                dist.all_reduce(self.epoch_sample_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.epoch_correct_count, op=dist.ReduceOp.SUM)
            
            accuracy = self.epoch_correct_count / self.epoch_sample_count
            self.epoch_correct_count = 0  # reset
            self.epoch_sample_count = 0  # reset
            return {
                'accuracy': accuracy,
                ...
                }
        ...
    ```

#### ⑤(Optional) Prepare for gears (trainers and testers):
* You may not need to create your trainers or testers in most easy tasks.
* But if you want to control training, validation, and testing procedures better:
1. Create a new 'py' file for your task (still named ResNet for example) in **src/gears** as **src/gears/resnet_gear.yaml**.

* In your new file, do the following steps just as in **src/gears/default_gear.py**.

2. Create a Trainer(TrainerBase) class for your Model like this:
    * 'cfg.trainer.trainer_choice=TRAINER_NAME' chooses the Trainer as @trainer_register('TRAINER_NAME').
    * Do not forget to **super** the corresponding method if you want to add some new features to existing methods.
    ```
    @trainer_register('default')
    class MyTrainer(TrainerBase):
        ...
    ```

3. (Further optional) Create a TesterBase class...

#### ⑥Prepare for config files:
1. Create a new 'yaml' file for your task (still named ResNet for example) in **configs** as **configs/train_resnet.yaml**.

* In your new file, do the following steps just as in **src/models/template_model.py**.
* Maybe just copying one of the 'template_train_XXX.yaml' files will help your work greatly.

2. 'config.additional' is the list of default config files in which all configs are inherited (and overridden by the latter one in the list **and this main yaml file**).

3. Just change or add any configs you need. All of them can be accessed by nested namespace as 'cfg.A.B.C...' in Python files.

4. Sweep the hyper-parameters
    * Example(s):
    ```
    sweep:
        sweep_enabled: True
        sweep_params:  # use '//' as the connector for sub-params 
            trainer//optimizer//lr_default: [2.0e-4, 4.0e-4]
            trainer//scheduler//scheduler_choice: [linear, cosine]
    ```
    * so if you run one task using this 'yaml' file, there will be four experiments in total:
        1. trainer.scheduler.scheduler_choice=linear  
           trainer.optimizer.lr_default=2.0e-4
        2. trainer.scheduler.scheduler_choice=cosine  
           trainer.optimizer.lr_default=2.0e-4
        3. trainer.scheduler.scheduler_choice=linear  
           trainer.optimizer.lr_default=4.0e-4
        4. trainer.scheduler.scheduler_choice=cosine  
           trainer.optimizer.lr_default=4.0e-4

5. You may explore the usage of other configs by reading the comments in 'default_xxx.yaml' and 'template_xxx.yaml' files.

#### ⑦Run the task:

 * Just use the 'yaml' file name of your task and run it by any device configurations (CPU, GPU, GPUs) mentioned earlier.
    ```
    bash scripts/train.sh -d 0,1,2,3 -c train_resnet
    ```

---
-> Easy Start ends here. Following are some more hints.

## Requirements
* './requirements.txt' is only an imperfect recommendation for the packages (especially the versions of Python and Pytorch) you need to install.
* Some common packages are not listed in the file.
* You may try different versions of the packages.

## Introduction to files & folders in this repo
### configs
* the folder for "xxx.yaml" files
    * some contain main configs for train or inference in a certain setting
    * others are used as inheritances specified by "config.additional" in main yaml files
* most of the things can be configured easily in "xxx.yaml" files

### data
* the folder for data (images, videos, texts, etc.) files, which will not be included by git

### outputs*
* folders for all outputs, which will not be included by git
* all folders start with "output" are for the same purpose
* the suffix may be used for distinguishing between different tasks, which can be set in "xxx.yaml" files as "info.output_dir"

### scripts
* the folder for bash scripts, which are used for the entrance of all main functions of the code
* common usage: "bash scripts/xxx.sh -args ..."
    * <u>"bash scripts/train.sh -d 0,1"</u> to train on GPU0 and GPU1 with config file "configs/template_train.yaml"
    * <u>"bash scripts/train.sh -c template_train_lenet -d 0"</u> to train on GPU0 with config file "configs/template_train_lenet.yaml"
    * <u>"bash scripts/train.sh -d 0,1,2,3 data.batch_size_per_rank=16"</u> to train on GPU0~3 with config file "configs/template_train.yaml", in which the default "batch_size_per_rank" will be changed to 16
    * ……

### src
* the folder for all Python scripts except for train.py and inference.py
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
