from .modules.tester_base import TesterBase, tester_register
from .modules.trainer_base import TrainerBase, trainer_register


@trainer_register('default')
class Trainer(TrainerBase):
    pass


@tester_register('default')
class Tester(TesterBase):
    pass
    