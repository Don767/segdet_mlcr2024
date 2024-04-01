from typing import Union

from mmengine import PARAM_SCHEDULERS
from mmengine.optim import _ParamScheduler, BaseOptimWrapper
from torch.optim import Optimizer


@PARAM_SCHEDULERS.register_module()
class LinearWarmupScheduler(_ParamScheduler):
    def __init__(self,
                 optimizer: Union[Optimizer, BaseOptimWrapper],
                 param_name: str,
                 start_lr: float,
                 end_lr: float,
                 begin: int = 0,
                 end: int = 250,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_iters = end - begin
        self.lr_slope = (end_lr - start_lr) / self.total_iters
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    def _get_value(self):
        dx = (self.last_step - self.begin) / self.total_iters
        # Before lr start
        if dx < 0:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        # lr is a line with slope `lr_slope` intersecting x=0 at self.start_lr
        lr = self.start_lr + self.lr_slope * dx
        if self.last_step == 0:
            return [
                lr
                for group in self.optimizer.param_groups
            ]
