import datetime
import statistics
from collections import defaultdict, deque
from math import nan

import torch
import torch.distributed as dist

from .misc import ConfigMisc, DistMisc, LoggerMisc, TimeMisc


class SmoothedValue(object):
    def __init__(self, window_size=None, format=None, final_format=None, prior=False, no_print=False, no_sync=False):
        if format is None:  # show current value and average when running
            format = '{value:.4f} ({avg:.4f})'
        if final_format is None:  # show average, min, max, std when one epoch finished
            final_format = '({avg:.4f} ± {std:.4f}) [{min:.4f}, {max:.4f}]'
        self.value_now = 0.0
        self.deque = deque(maxlen=window_size)
        self.count = 0
        self.total = 0.0
        self.format = format
        self.final_format = final_format
        self.prior = prior
        self.no_print = no_print
        self.require_sync = not no_sync
        self.synced = False

    def append_one_value(self, value):
        # assert n==1, 'n != 1 is not supported yet.'
        self.deque.append(value)
        self.value_now = value
        self.count += 1
        self.total += value
        
    def prepare_sync_meters(self):
        assert not self.synced, 'Meters have been synced.'
        d = torch.as_tensor(list(self.deque), dtype=torch.float64, device='cpu')
        t = torch.as_tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        return d, t
    
    def write_synced_meters(self, d, t):
        d = d.tolist()
        t = t.tolist()
        self.deque.clear()
        self.deque += list(d)
        self.count = int(t[0])
        self.total = t[1]
    
    @property
    def std(self):
        return statistics.stdev(self.deque) if len(self.deque) > 1 else nan
    
    @property
    def avg(self):
        return self.total / self.count if self.count > 0 else nan
    
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
            value=self.value,
            avg=self.avg,
            min=self.min,
            max=self.max,
            std=self.std,
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
        
        self.meters = defaultdict(SmoothedValue)
        self.synced = False
        
    def add_meters(self, meters):
        for meter in meters:
            if isinstance(meter, str):
                self.meters[meter] = SmoothedValue()
            elif isinstance(meter, dict):
                self.meters.update(meter)

    def update_meters(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            # self.meters[k] = SmoothedValue()  # as default
            self.meters[k].append_one_value(v)
            
    def add_epoch_meters(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k] = SmoothedValue(
                window_size=1,
                format='{value:.4f}',
                final_format='{value:.4f}',
                no_sync=True,
                )
            self.meters[k].append_one_value(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f'"{type(self).__name__}" object has no attribute "{attr}"')

    def meters_str(self, final=False, synced=True):
        prior_meter_s = []
        meters_s = []
        for name, meter in self.meters.items():
            if meter.no_print:  # skip no_print meters
                continue
            if meter.prior:
                prior_meter_s.append(
                    f'{name}: {meter.get_str(final, synced)}'
                )
            else:
                meters_s.append(
                    f'{name}: {meter.get_str(final, synced)}'
                )
        return self.delimiter.join(prior_meter_s + meters_s)
    
    def _synchronize_between_processes(self):
        if not DistMisc.is_dist_avail_and_initialized():
            return
        assert not self.synced, 'Meters have been synced.'
        self.synced = True
        d_list = []
        t_list = []
        for meter in self.meters.values():
            if meter.require_sync:
                d, t = meter.prepare_sync_meters()
                d_list.append(d)
                t_list.append(t)
        ds = torch.stack(d_list, dim=0)
        ts = torch.stack(t_list, dim=0)
        dist.barrier()
        dist.all_reduce(ts)
        gathered_ds = [None] * DistMisc.get_world_size()
        dist.all_gather_object(gathered_ds, ds)
        ds = torch.cat(gathered_ds, dim=1)
        line_idx = 0
        for meter in self.meters.values():
            if meter.require_sync:
                meter.write_synced_meters(ds[line_idx], ts[line_idx])
                line_idx += 1

    def log_every(self, iterable):
        self.iter_len = len(iterable)

        iter_time = SmoothedValue(format='{value:.4f} ({avg:.4f})')
        data_time = SmoothedValue(format='{value:.4f} ({avg:.4f})')
        model_time = SmoothedValue(format='{value:.4f} ({avg:.4f})')
        
        if self.pbar is not None:
            if self.global_tqdm:
                post_msg = '\033[33m' + self.epoch_str \
                    + '\033[32m' + ' [{0}/{1}] eta: {eta} ' \
                    + '\033[30m' + ' t_data: {data_time}  t_model: {model_time}\033[0m'
            else:
                post_msg = '\033[30m' + ' t_data: {data_time}  t_model: {model_time}\033[0m'
                self.pbar.set_description_str(self.header + ' ' + self.epoch_str, refresh=False)
            postlines_msg = self.delimiter.join([
                # '\t{meters}',
                '    \033[30m{meters}\033[0m',
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
                        meters=self.meters_str(),
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
            self._synchronize_between_processes()
        if final_print:
            self._final_print(print_time=True, synced=sync)

        return_dict = {}         
        if 'all' in no_avg_list:
            return_dict.update({
                k: v.value
                for k, v in self.meters.items()
                })
        else:
            return_dict.update({
                k: getattr(v, 'avg') if k not in no_avg_list else v.value
                for k, v in self.meters.items()
                })
        return return_dict
                     
    def _final_print(self, print_time=False, synced=True):       
        final_msg = self.delimiter.join([
            self.header + ' ' + self.epoch_str + ' finished. Summary of All Ranks:'
            '\n    {meters}',
        ]).format(meters=self.meters_str(final=True, synced=synced))
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
