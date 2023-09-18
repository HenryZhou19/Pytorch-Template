import os
from argparse import Namespace

import cv2
import numpy as np
import torch

from src.utils.misc import TesterMisc, TrainerMisc


class Trainer(TrainerMisc):
    @staticmethod
    def before_one_epoch(cfg, trainer_status, **kwargs):
        super(Trainer, Trainer).before_one_epoch(cfg, trainer_status, **kwargs)
        
    @staticmethod
    def after_training_before_validation(cfg, trainer_status, **kwargs):
        super(Trainer, Trainer).after_training_before_validation(cfg, trainer_status, **kwargs)

    @staticmethod
    def after_validation(cfg, trainer_status, **kwargs):
        super(Trainer, Trainer).after_validation(cfg, trainer_status, **kwargs)
        
    @staticmethod
    def after_all_epochs(cfg, trainer_status, **kwargs):
        super(Trainer, Trainer).after_all_epochs(cfg, trainer_status, **kwargs)


class Tester(TesterMisc):
    @staticmethod
    def before_inference(cfg, tester_status, **kwargs):
        super(Tester, Tester).before_inference(cfg, tester_status, **kwargs)

    @staticmethod
    def after_inference(cfg, tester_status, **kwargs):
        super(Tester, Tester).after_inference(cfg, tester_status, **kwargs)
        