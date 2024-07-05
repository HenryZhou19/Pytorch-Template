import datetime
import statistics
from collections import defaultdict, deque
from math import nan

import numpy as np
import torch
import torch.distributed as dist

from .misc import ConfigMisc, DistMisc, LoggerMisc, TimeMisc

__all__ = [
    'MetricLogger',
    'ValueMetric',
    ]

class ValueMetric(object):
    def __init__(self, window_size=None, format=None, final_format=None, high_prior=False, low_prior=False, no_print=False, no_sync=False):
        if format is None:  # show current value and average when running
            format = '{value:.4f} ({avg:.4f})'
        if final_format is None:  # show average, min, max, std when one epoch finished
            final_format = '({avg:.4f} ± {std:.4f}) [{min:.4f}, {max:.4f}]'
        self.value_now = 0.0
        self.deque = deque(maxlen=window_size)
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
        self.deque.append(value)
        self.value_now = value
        self.sample_count += sample_count
        self.total += value * sample_count
        
    def prepare_sync_metrics(self):
        assert not self.synced, 'metrics have been synced.'
        queue = torch.as_tensor(list(self.deque), dtype=torch.float64, device='cpu')
        summary = torch.as_tensor([self.sample_count, self.total], dtype=torch.float64, device='cuda')
        return queue, summary
    
    def write_synced_metrics(self, queue, summary):
        queue = queue.tolist()
        summary = summary.tolist()
        self.deque.clear()
        self.deque += list(queue)
        self.sample_count = int(summary[0])
        self.total = summary[1]
    
    @property
    def std(self):
        try:
            return statistics.stdev(self.deque) if len(self.deque) > 1 else nan
        except:
            return nan
    
    @property
    def avg(self):
        return self.total / self.sample_count if self.sample_count > 0 else nan
    
    @property
    def min(self):
        return min(self.deque) if len(self.deque) > 0 else nan
    
    @property
    def max(self):
        return max(self.deque) if len(self.deque) > 0 else nan

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
            min=self.min if 'min' in f else None,
            max=self.max if 'max' in f else None,
            std=self.std if 'std' in f else None,
        )


class MetricLogger(object):
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
            assert isinstance(v, (float, int)), f'v is {type(v)}, not float or int.'
            # self.metrics[k] = SmoothedValue()  # as default
            self.metrics[k].append_one_value(v, sample_count)
            
    def add_epoch_metrics(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                v = v.item()
            assert isinstance(v, (float, int)), f'v is {type(v)}, not float or int.'
            self.metrics[k] = ValueMetric(
                window_size=1,
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
        self.synced = True
        queue_list = []
        summary_list = []
        for metric in self.metrics.values():
            if metric.require_sync:
                queue, summary = metric.prepare_sync_metrics()
                queue_list.append(queue)
                summary_list.append(summary)
        queue_stack = torch.stack(queue_list, dim=0)
        summary_stack = torch.stack(summary_list, dim=0)
        dist.barrier()
        dist.all_reduce(summary_stack)
        gathered_queue_stack = [None] * DistMisc.get_world_size()
        dist.all_gather_object(gathered_queue_stack, queue_stack)
        queue_stack = torch.cat(gathered_queue_stack, dim=1)
        require_sync_metric_idx = 0
        for metric in self.metrics.values():
            if metric.require_sync:
                metric.write_synced_metrics(queue_stack[require_sync_metric_idx], summary_stack[require_sync_metric_idx])
                require_sync_metric_idx += 1

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
                        step = self.iter_len % self.print_freq
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
