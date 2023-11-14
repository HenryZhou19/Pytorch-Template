import matplotlib.pyplot as plt
import numpy as np


def plot_curve_with_std_min_max(y_mean, y_std, y_min, y_max, x=None, metric_name='Curve', save_file_path=None):
    """
    Args:
        y_mean (np.ndarray or list)
        y_std (np.ndarray or list)
        y_min (np.ndarray or list)
        y_max (np.ndarray or list) 
        x (np.ndarray or list, optional): Defaults to None.
    """
    if x is None:
        x = np.arange(1, len(y_mean) + 1)
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y_mean, list):
        y_mean = np.array(y_mean)
    if isinstance(y_std, list):
        y_std = np.array(y_std)
    if isinstance(y_min, list):
        y_min = np.array(y_min)
    if isinstance(y_max, list):
        y_max = np.array(y_max)

    _, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(x, y_mean, label='Mean', color='blue')
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.2, label='± Std')
    ax.fill_between(x, y_min, y_max, color='gray', alpha=0.2, label='Min/Max')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Values')
    ax.set_title(f'{metric_name}')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.savefig(save_file_path)
    plt.close()