import os
import sys

from metric_curve import plot_curve_with_std_min_max
from read_log import get_metrics_from_log

if __name__ == '__main__':
    log_file_path = sys.argv[1]
    log_dir = os.path.dirname(log_file_path)
    metric_list = get_metrics_from_log(log_file_path)
    for metric_name, metric_data in metric_list.items():
        plot_curve_with_std_min_max(
            y_mean=metric_data['mean'],
            y_std=metric_data['std'],
            y_min=metric_data['min'],
            y_max=metric_data['max'],
            metric_name=metric_name,
            save_file_path=f'{log_dir}/{metric_name}.png'
            )