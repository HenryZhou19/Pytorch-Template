import bisect
import functools
import math
from typing import Any, Callable, Dict, List


class MovingFn:
    FUNCTIONS = {
        'keep_start': lambda idx, total, **kwargs: 0.0,
        'keep_end': lambda idx, total, **kwargs: 1.0,
        'linear': lambda idx, total, **kwargs: float(idx) / total,
        'exponential': lambda idx, total, gamma=5.0, **kwargs: math.exp(gamma * float(idx) / total - gamma),
        'cosine': lambda idx, total, **kwargs: 0.5 * (1.0 - math.cos(math.pi * float(idx) / total)),
    }

    @staticmethod
    def get_moving_fn(warmup_type: str, **kwargs) -> Callable:
        if warmup_type not in MovingFn.FUNCTIONS:
            raise ValueError(f"Unknown moving function type: {warmup_type}")
        fn = MovingFn.FUNCTIONS[warmup_type]
        return functools.partial(fn, **kwargs)


class BasicScheduler:
    def __init__(self, T_max=1, current_index=-1):
        self.T_max = T_max
        self.current_index = current_index

    def _calc_value(self, index):
        raise NotImplementedError

    def reset_index(self, index=-1):
        self.current_index = index

    def __getitem__(self, index):
        return self._calc_value(index)

    def __call__(self):
        return self.step()

    def __next__(self):
        return self.step()

    def step(self):
        self.current_index += 1
        if self.current_index >= self.T_max:
            # raise StopIteration
            return self[self.T_max - 1]
        return self[self.current_index]

    def get_all_as_list(self):
        return [self._calc_value(i) for i in range(self.T_max)]

    def state_dict(self):
        return {'current_index': self.current_index}

    def load_state_dict(self, state_dict):
        self.current_index = state_dict.get('current_index', -1)


class DummyScheduler(BasicScheduler):
    def __init__(self, dummy_value=0.0):
        super().__init__(T_max=1, current_index=-1)
        self.dummy_value = dummy_value
    
    def __getitem__(self, index):
        return self.dummy_value


class ArbitraryScheduler(BasicScheduler):
    def __init__(
        self,
        T_list: List[int],
        key_value_list: List[float],
        moving_type_list: List[str],
        moving_kwargs_list: List[Dict[str, Any]] = None,
        current_index: int = -1,
    ):
        '''
        An arbitrary scheduler that can have multiple phases with different moving functions.
        If N phases is wanted,
            `T_list`, `moving_type_list` should all have length `N`,
            and `key_value_list` should have length `N+1`.
        
        Example: a simple linear warmup + cosine annealing scheduler will have 2 phases:
            T_list = [T_warmup, T_max]
            key_value_list = [start_value, base_value, end_value]
            moving_type_list = ['linear', 'cosine']
        
        T_list: a list of integers, each integer is the end index (exclusive) of each phase
            so the values in T_list should be strictly increasing and the last value will be T_max
        key_value_list: a list of floats, each float is the key value at the start of each phase,
            and the last float is the end value of the last phase
        moving_type_list: a list of strings, each string is the moving function type for each phase
            Options: 'keep_start', 'keep_end', 'linear', 'exponential', 'cosine', etc. as defined in MovingFn class
        current_index: the current index of the scheduler, (default: -1) means before the first step
        '''
        self.N_phases = len(T_list)
        if len(key_value_list) != self.N_phases + 1:
            raise ValueError(f'key_value_list should have length {self.N_phases + 1}, got {len(key_value_list)}')
        if len(moving_type_list) != self.N_phases:
            raise ValueError(f'moving_type_list should have length {self.N_phases}, got {len(moving_type_list)}')
        if not all(T_list[i] < T_list[i+1] for i in range(self.N_phases - 1)):
            raise ValueError(f'T_list should be strictly increasing, got {T_list}')
        
        T_max = T_list[-1]
        super().__init__(T_max, current_index=current_index)
        self.T_list = T_list
        self.key_value_list = key_value_list
        self.moving_type_list = moving_type_list
        if moving_kwargs_list is None:
            moving_kwargs_list = [{} for _ in range(self.N_phases)]
        elif len(moving_kwargs_list) != self.N_phases:
            raise ValueError(f'moving_kwargs_list should have length {self.N_phases}, got {len(moving_kwargs_list)}')
        self.moving_fn_list = [
            MovingFn.get_moving_fn(moving_type, **kwargs)
            for moving_type, kwargs in zip(moving_type_list, moving_kwargs_list)
        ]
        
        self.phase_idx = None
        self.phase_start = 0
        
    def _update_phase_idx(self, index):
        if self.phase_idx is None or not (self.phase_start <= index < self.T_list[self.phase_idx]):
            self.phase_idx = bisect.bisect_right(self.T_list, index)
            self.phase_start = 0 if self.phase_idx == 0 else self.T_list[self.phase_idx - 1]

    def _calc_value(self, index):
        assert 0 <= index < self.T_max, f'Index: {index} out of range: [0, {self.T_max})'
        self._update_phase_idx(index)
        
        start_T = self.phase_start
        start_value = self.key_value_list[self.phase_idx]
        end_T = self.T_list[self.phase_idx]
        end_value = self.key_value_list[self.phase_idx + 1]
        moving_fn = self.moving_fn_list[self.phase_idx]
        phase_total = end_T - start_T
        phase_idx_in = index - start_T
        scale = moving_fn(phase_idx_in, phase_total)
        value = start_value + scale * (end_value - start_value)
        return value
