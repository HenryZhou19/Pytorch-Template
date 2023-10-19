import datetime
import statistics
import sys
from collections import defaultdict, deque
from math import nan

import torch
import torch.distributed as dist

from .misc import DistMisc, TesterMisc, TimeMisc


class SmoothedValue(object):
    def __init__(self, window_size=None, fmt=None, final_fmt=None, final_fmt_no_sync=None, prior=False, no_print=False, no_sync=False):
        if fmt is None:
            fmt = '{value:.4f} ({avg:.4f})'
        if final_fmt is None:
            final_fmt = '({global_avg:.4f} ± {global_std:.4f}) [{global_min:.4f}, {global_max:.4f}]'
        if final_fmt_no_sync is None:
            final_fmt_no_sync = '({avg:.4f} ± {std:.4f}) [{min:.4f}, {max:.4f}]'
        self.value_now = 0.0
        self.deque = deque(maxlen=window_size)
        self.synced_deque = deque()
        self.count = 0
        self.synced_count = 0
        self.total = 0.0
        self.synced_total = 0.0
        self.fmt = fmt
        self.final_fmt = final_fmt
        self.final_fmt_no_sync = final_fmt_no_sync
        self.prior = prior
        self.no_print = no_print
        self.require_sync = not no_sync

    def update(self, value, n=1):
        assert n==1, 'n != 1 is not supported yet.'
        self.deque.append(value)
        self.value_now = value
        self.count += 1
        self.total += value
        
    def prepare_sync_meters(self):
        d = torch.as_tensor(list(self.deque), dtype=torch.float64, device='cpu')
        t = torch.as_tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        return d, t
    
    def write_synced_meters(self, d, t):
        d = d.tolist()
        t = t.tolist()
        self.synced_deque += list(d)
        self.synced_count = self.synced_count + int(t[0])
        self.synced_total = self.synced_total + t[1]
        self.deque = deque()
        self.count = 0
        self.total = 0.0
    
    @property
    def global_std(self):
        return statistics.stdev(self.synced_deque) if len(self.synced_deque) > 1 else nan

    @property
    def global_avg(self):
        return self.synced_total / self.synced_count if self.synced_count > 0 else nan
    
    @property
    def global_min(self):
        return min(self.synced_deque) if len(self.synced_deque) > 0 else nan
    
    @property
    def global_max(self):
        return max(self.synced_deque) if len(self.synced_deque) > 0 else nan
    
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
            if synced:
                f = self.final_fmt
            else:
                f = self.final_fmt_no_sync
        else:
            f = self.fmt
        return f.format(
            value=self.value,
            avg=self.avg,
            min=self.min,
            max=self.max,
            std=self.std,
            global_min=self.global_min,
            global_max=self.global_max,
            global_avg=self.global_avg,
            global_std=self.global_std,
        )


class MetricLogger(object):
    def __init__(self, cfg=None, log_file=sys.stdout, print_freq=1, debug=False, global_tqdm=False, pbar=None, delimiter='  ', header='', epoch_str=''):
        if cfg is not None:
            self.log_file=cfg.info.log_file
            self.print_freq=cfg.info.cli_log_freq
            self.debug=cfg.special.debug
            self.global_tqdm=cfg.info.global_tqdm if not TesterMisc.is_inference(cfg) else False
        else:
            self.log_file = log_file
            self.print_freq = print_freq
            self.debug = debug
            self.global_tqdm = global_tqdm

        self.pbar = pbar
        self.delimiter = delimiter
        self.header = header
        self.epoch_str = epoch_str
        
        self.meters = defaultdict(SmoothedValue)
        self.synced = False

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def meters_str(self, final=False, synced=True):
        prior_meter_s = []
        meters_s = []
        for name, meter in self.meters.items():
            if meter.no_print:  # skip no_print meters
                continue
            if meter.prior:
                prior_meter_s.append(
                    '{}: {}'.format(name, meter.get_str(final, synced))
                )
            else:
                meters_s.append(
                    '{}: {}'.format(name, meter.get_str(final, synced))
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
        gathered_ds = [None]*DistMisc.get_world_size()
        dist.all_gather_object(gathered_ds, ds)
        ds = torch.cat(gathered_ds, dim=1)
        line_idx = 0
        for meter in self.meters.values():
            if meter.require_sync:
                meter.write_synced_meters(ds[line_idx], ts[line_idx])
                line_idx += 1

    def add_meters(self, meters):
        for meter in meters:
            if isinstance(meter, str):
                self.meters[meter] = SmoothedValue()
            elif isinstance(meter, dict):
                self.meters.update(meter)

    def log_every(self, iterable):
        self.iter_len = len(iterable)

        i = 0
        iter_time = SmoothedValue(fmt='{value:.4f} ({avg:.4f})')
        loader_time = SmoothedValue(fmt='{value:.4f} ({avg:.4f})')
        
        if self.pbar is not None:
            if self.global_tqdm:
                post_msg = self.epoch_str + ' [{0}/{1}]' + ' eta: {eta}'
            else:
                self.pbar.set_description_str(self.header + ' ' + self.epoch_str, refresh=False)
            postlines_msg = self.delimiter.join([
                # '\t{meters}',
                '    {meters}',
                'loader_time: {loader_time}',
                # 'iter_time: {iter_time}',
            ])

        last_infos = '\n'
        self.timer = TimeMisc.Timer()
        for obj in iterable:
            i += 1
            loader_time.update(self.timer.info['last'])
            yield obj
            iter_time.update(self.timer.info['last'])
            
            if self.pbar is not None:
                if i % self.print_freq == 0 or i == self.iter_len:
                    
                    if self.global_tqdm:
                        eta_second = iter_time.avg * (self.iter_len - i)
                        eta_string = str(datetime.timedelta(seconds=int(eta_second)))
                        self.pbar.set_postfix_str(post_msg.format(i, self.iter_len, eta=eta_string), refresh=False)

                    last_infos = postlines_msg.format(
                        meters=self.meters_str(), loader_time=loader_time.get_str(), 
                        # iter_time=iter_time.get_str(),
                    )

                    self.pbar.set_postlines_str([last_infos], refresh=False)  # len(list()) == self.pbar.postlines
                    if i % self.print_freq == 0:
                        step = self.print_freq        
                    else:
                        step = self.iter_len % self.print_freq
                    self.pbar.update(n=step)
                    self.pbar.refresh()

            # DEBUG
            if self.debug and i % self.print_freq == 0:
                break

            self.timer.press()
                 
    def output_dict(self, no_avg_list=[], sync=False, final_print=False):
        if sync:
            self._synchronize_between_processes()
            avg_name = 'global_avg'
        else:
            avg_name = 'avg'
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
                k: getattr(v, avg_name) if k not in no_avg_list else v.value
                for k, v in self.meters.items()
                })
        return return_dict
                     
    def _final_print(self, print_time=False, synced=True):       
        final_msg = self.delimiter.join([
            self.header + ' ' + self.epoch_str + ' finished. Summary of All Ranks:'
            '\n\t{meters}',
        ]).format(meters=self.meters_str(final=True, synced=synced))
        if print_time:
            total_time = self.timer.info['all']
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            final_msg += f'\n\tElapsed time: {total_time_str} ({total_time / self.iter_len:.4f} sec / batch)\n'
        print(
            final_msg, '\n',
            file=self.log_file
        )
        self.log_file.flush()
        if self.pbar is not None:
            print(
                '\n' * (self.pbar.postlines + 1) + final_msg, '\n'
            )
