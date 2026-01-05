import datetime
from collections import defaultdict
from math import nan
from typing import List

import numpy as np
import torch
import torch.distributed as dist

from .misc import ConfigMisc, DistMisc, LoggerMisc, TimeMisc

__all__ = [
    'ValueMetric',
    'MetricLogger',
    'LossGuard',
    ]


class ValueMetric:
    def __init__(self, format=None, final_format=None, high_prior=False, low_prior=False, no_print=False, no_sync=False):
        if format is None:  # show current value and average when running
            format = '{value:.4f} ({avg:.4f})'
        if final_format is None:  # only show average when one epoch finished (synced if needed)
            final_format = '({avg:.4f})'
        self.value_now = 0.0
        self.sample_count = 0
        self.total = 0.0
        self.format = format
        self.final_format = final_format
        self.high_prior = high_prior
        self.low_prior = low_prior
        assert not (high_prior and low_prior), 'high_prior and low_prior cannot be True at the same time.'
        self.no_print = no_print
        self.require_sync = not no_sync
        self.synced = False

    def append_one_value(self, value, sample_count=1):
        # maybe the mean of one batch, so sample_count can be larger than 1
        self.value_now = value
        self.sample_count += sample_count
        self.total += value * sample_count
        
    def prepare_sync_metrics(self):
        assert not self.synced, 'metrics have been synced.'
        summary = torch.as_tensor([self.sample_count, self.total], dtype=torch.float64, device='cuda')
        return summary
    
    def write_synced_metrics(self, summary):
        summary = summary.tolist()
        self.sample_count = int(summary[0])
        self.total = summary[1]
    
    @property
    def avg(self):
        return self.total / self.sample_count if self.sample_count > 0 else nan

    @property
    def value(self):
        return self.value_now

    def get_str(self, final=False, synced=True):
        if final:
            f = self.final_format
        else:
            f = self.format
        return f.format(
            value=self.value if 'value' in f else None,
            avg=self.avg if 'avg' in f else None,
        )


class MetricLogger:
    def __init__(self, cfg, loggers, pbar=None, delimiter='  ', header='', epoch_str=''):
        self.print_freq=cfg.info.cli_log_freq
        self.debug=cfg.special.debug
        self.global_tqdm=cfg.info.global_tqdm if not ConfigMisc.is_inference(cfg) else False
        
        self.log_file=loggers.log_file
        
        self.pbar: LoggerMisc.MultiTQDM = pbar
        self.delimiter = delimiter
        self.header = header
        self.epoch_str = epoch_str
        
        self.metrics = defaultdict(ValueMetric)
        self.synced = False
        
    def add_metrics(self, metrics):
        for metric in metrics:
            if isinstance(metric, str):
                self.metrics[metric] = ValueMetric()
            elif isinstance(metric, dict):
                self.metrics.update(metric)

    def update_metrics(self, sample_count=1, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                v = v.item()
            assert isinstance(v, (float, int)), f'{v} is {type(v)}, not float or int.'
            # self.metrics[k] = SmoothedValue()  # as default
            self.metrics[k].append_one_value(v, sample_count)
            
    def add_epoch_metrics(self, *args, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, (torch.Tensor, np.ndarray, np.number)):
                v = v.item()
            assert isinstance(v, (float, int)), f'{v} is {type(v)}, not float or int.'
            self.metrics[k] = ValueMetric(
                format='({value:.4f})',
                final_format='({value:.4f})',
                no_sync=True,
                )
            self.metrics[k].append_one_value(v)

    def __getattr__(self, attr):
        if attr in self.metrics:
            return self.metrics[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f'"{type(self).__name__}" object has no attribute "{attr}"')

    def metrics_str(self, final=False, synced=True):
        high_prior_metric_s = []
        metrics_s = []
        low_prior_metrics_s = []
        for name, metric in self.metrics.items():
            if metric.no_print:  # skip no_print metrics
                continue
            metric_str = f'{name}: {metric.get_str(final, synced)}'
            if metric.high_prior:
                high_prior_metric_s.append(metric_str)
            elif metric.low_prior:
                low_prior_metrics_s.append(metric_str)
            else:
                metrics_s.append(metric_str)
        return self.delimiter.join(high_prior_metric_s + metrics_s + low_prior_metrics_s)
    
    def _synchronize_all_processes(self):
        if not DistMisc.is_dist_avail_and_initialized():
            return
        assert not self.synced, 'metrics have been synced.'
        summary_list = []
        try:
            for metric in self.metrics.values():
                if metric.require_sync:
                    summary = metric.prepare_sync_metrics()
                    summary_list.append(summary)
            summary_stack = torch.stack(summary_list, dim=0)
            dist.barrier()
            dist.all_reduce(summary_stack)
            require_sync_metric_idx = 0
            for metric in self.metrics.values():
                if metric.require_sync:
                    metric.write_synced_metrics(summary_stack[require_sync_metric_idx])
                    require_sync_metric_idx += 1
        except Exception as e:
            rank = DistMisc.get_rank()
            print(f'Rank: {rank}, error in MetricLogger._synchronize_all_processes():', force=True)
            for name, metric in self.metrics.items():
                print(f'\tRank: {rank}, metric name: {name}\n\t\ttype: {type(metric.value)}, sample_count: {metric.sample_count}, total: {metric.total}\n\t\trequire_sync: {metric.require_sync}, synced: {metric.synced}', force=True)
            raise e
        self.synced = True

    def log_every(self, iterable):
        self.iter_len = len(iterable)

        iter_time = ValueMetric(format='{value:.4f} ({avg:.4f})')
        data_time = ValueMetric(format='{value:.4f} ({avg:.4f})')
        model_time = ValueMetric(format='{value:.4f} ({avg:.4f})')
        
        if self.pbar is not None:
            if self.global_tqdm:
                post_msg = '\033[33m' + self.epoch_str \
                    + '\033[32m' + ' [{0}/{1}] eta: {eta} ' \
                    + '\033[30m' + ' t_data: {data_time}  t_model: {model_time}\033[0m'
            else:
                post_msg = '\033[30m' + ' t_data: {data_time}  t_model: {model_time}\033[0m'
                self.pbar.set_description_str(self.header + ' ' + self.epoch_str, refresh=False)
            postlines_msg = self.delimiter.join([
                # '\t{metrics}',
                '    \033[30m{metrics}\033[0m',
                # 'data_time: {data_time}',
                # 'iter_time: {iter_time}',
            ])

        self.timer = TimeMisc.Timer()
        for idx, obj in enumerate(iterable, start=1):
            data_time.append_one_value(self.timer.info['last'])
            yield obj
            iter_time.append_one_value(self.timer.info['last'])
            model_time.append_one_value(iter_time.value_now - data_time.value_now)
            
            if self.pbar is not None:
                if idx % self.print_freq == 0 or idx == self.iter_len:
                    
                    if self.global_tqdm:
                        eta_second = iter_time.avg * (self.iter_len - idx)
                        eta_string = str(datetime.timedelta(seconds=int(eta_second)))
                        self.pbar.set_postfix_str(post_msg.format(idx, self.iter_len, eta=eta_string, data_time=data_time.get_str(), model_time=model_time.get_str()), refresh=False)
                    else:
                        self.pbar.set_postfix_str(post_msg.format(data_time=data_time.get_str(), model_time=model_time.get_str()), refresh=False)
                        
                    last_infos = postlines_msg.format(
                        metrics=self.metrics_str(),
                        # data_time=data_time.get_str(), 
                        # iter_time=iter_time.get_str(),
                    )
                    
                    self.pbar.set_postlines_str([last_infos], refresh=False)  # len(list()) == self.pbar.postlines
                    if idx % self.print_freq == 0:
                        step = self.print_freq        
                    else:
                        step = int(self.iter_len % self.print_freq)
                    self.pbar.update(n=step)
                    self.pbar.refresh()

            # DEBUG
            if self.debug == 'one_iter' and idx % self.print_freq == 0:
                break

            self.timer.press()
                 
    def output_dict(self, no_avg_list=[], sync=False, final_print=False):
        if sync:
            self._synchronize_all_processes()
        if final_print:
            self._final_print(print_time=True, synced=sync)
        
        return_dict = {}         
        if 'all' in no_avg_list:
            return_dict.update({
                k: v.value
                for k, v in self.metrics.items()
                })
        else:
            return_dict.update({
                k: getattr(v, 'avg') if k not in no_avg_list else v.value
                for k, v in self.metrics.items()
                })
        return return_dict
                     
    def _final_print(self, print_time=False, synced=True):       
        final_msg = self.delimiter.join([
            self.header + ' ' + self.epoch_str + ' finished. Summary of All Ranks:'
            '\n    {metrics}',
        ]).format(metrics=self.metrics_str(final=True, synced=synced))
        if print_time:
            total_time = self.timer.info['all']
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            final_msg += f'\n    Elapsed time: {total_time_str} ({total_time / self.iter_len:.4f} sec / batch)\n'
        print(
            final_msg, '\n',
            file=self.log_file
        )
        self.log_file.flush()
        if self.pbar is not None:
            print(
                '\n' * (self.pbar.postlines + 1) + '\033[34m' + final_msg, '\033[0m\n'
            )


class LossGuard:
    def __init__(self, scalers: List[torch.cuda.amp.GradScaler] | List[None], tolerance: int = 5):
        '''
        Args:
            mlogger: MetricLogger, to show all metrics when nan/inf detected.
            tolerance: (int) Number of consecutive nan/inf before aborting.
            scalers: List[torch.cuda.amp.GradScaler] | None
        '''
        self.tolerance = tolerance
        self.valid_scalers = [scaler for scaler in scalers if scaler is not None]
        
        self.nan_inf_count = 0
        self.prev_scales = [None] * len(self.valid_scalers)

    def reset(self):
        self.nan_inf_count = 0
        self.prev_scales = [None] * len(self.valid_scalers)
        
    def _check_one_value_safe(self, loss):
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            return False
        return True

    @torch.no_grad()
    def check(self, loss_dict: dict, mlogger: MetricLogger, model: torch.nn.Module | None = None):
        '''
        loss_dict: dict of losses to be checked
        model: torch.nn.Module | None, to check model parameters and gradients
        '''
        safe_to_backward = True
        
        for k, loss in loss_dict.items():
            if not self._check_one_value_safe(loss):
                safe_to_backward = False
                print(LoggerMisc.block_wrapper(f'Rank {DistMisc.get_rank()}: [LossGuard] `{k}` has nan/inf: {loss}, already counts: {self.nan_inf_count}/{self.tolerance}', '#'), force=True)

        if not safe_to_backward:
            # check amp scale and set safe_to_backward to True if scale is reduced
            if len(self.valid_scalers) > 0:
                all_scalers_safe = True
                for idx, scaler in enumerate(self.valid_scalers):
                    prev_scale = self.prev_scales[idx]
                    curr_scale = scaler.get_scale()
                    if prev_scale is not None:
                        if curr_scale < prev_scale:
                            print(f'Rank {DistMisc.get_rank()}: [LossGuard] AMP scale in scaler No.{idx + 1} reduced from {prev_scale} to {curr_scale}, safe to backward.', force=True)
                        else:
                            print(f'Rank {DistMisc.get_rank()}: [LossGuard] AMP scale in scaler No.{idx + 1} not reduced: {prev_scale} -> {curr_scale}.', force=True)
                            all_scalers_safe = False
                    self.prev_scales[idx] = curr_scale
                if all_scalers_safe:
                    safe_to_backward = True
        
        if not safe_to_backward:
            self.nan_inf_count += 1
            print(f'Rank {DistMisc.get_rank()}: [LossGuard] All metrics:', force=True)
            print('\t' + mlogger.metrics_str(synced=False).replace(mlogger.delimiter, '\n\t'), force=True)
            if model is not None:
                for n, p in model.named_parameters():
                    if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                        print(f'[LossGuard] Parameter `{n}` has nan/inf!', force=True)
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        print(f'[LossGuard] Grad `{n}` has nan/inf!', force=True)
                        
            if self.nan_inf_count > self.tolerance:
                LoggerMisc.get_wandb_pid(kill_all=True)
                raise RuntimeError(f'[LossGuard] NAN/INF detected {self.nan_inf_count} consecutive steps — aborting.')
        else:
            self.nan_inf_count = 0
        
        return safe_to_backward