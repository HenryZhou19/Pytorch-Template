import os
import sys

from metric_curve import *
from read_log import get_metrics_from_log

if __name__ == '__main__':
    log_file_path = sys.argv[1]
    log_dir = os.path.dirname(log_file_path)
    metric_list = get_metrics_from_log(log_file_path)
    
    for metric_name, metric_data in metric_list.items():
        if 'mean' in metric_data and 'std' in metric_data and 'min' in metric_data and 'max' in metric_data:
            plot_curve_with_std_min_max(
                y_mean=metric_data['mean'],
                y_std=metric_data['std'],
                y_min=metric_data['min'],
                y_max=metric_data['max'],
                metric_name=metric_name,
                save_file_path=f'{log_dir}/{metric_name}.png'
                )
        elif 'min' in metric_data and 'max' in metric_data:
            plot_curve_only_min_max(
                y_min=metric_data['min'],
                y_max=metric_data['max'],
                metric_name=metric_name,
                save_file_path=f'{log_dir}/{metric_name}.png'
                )
        elif 'value' in metric_data:
            plot_curve(
                y=metric_data['value'],
                metric_name=metric_name,
                save_file_path=f'{log_dir}/{metric_name}.png'
                )
        else:
            Warning(f'Unknown metric type: {metric_name}')