import re


def get_metrics_from_log(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    pattern_full = re.compile(r'(\w+): \(([\d.]+) ± ([\d.]+)\) \[([\d.]+), ([\d.]+)\]')  # 'name': (mean ± std) [min, max]
    pattern_minmax = re.compile(r'(\w+): \[([\d.]+), ([\d.]+)\]')  #  'name': [min, max]
    pattern_single = re.compile(r'(\w+): \(([\d.]+)\)')  # 'name': value

    metric_list = {}
    for block in log_content.split('\n\n'):
        block = block.strip('\n')
        if block.startswith('Train') or block.startswith('Eval'):
            prefix = 'train_' if block.startswith('Train') else 'eval_'
            
            matches_full = pattern_full.findall(block)
            matches_minmax = pattern_minmax.findall(block)
            matches_single = pattern_single.findall(block)
            
            for data in matches_full:
                metric_name = prefix + data[0]
                if metric_name not in metric_list:
                    metric_list[metric_name] = {
                        'mean': [],
                        'std': [],
                        'min': [],
                        'max': [],
                        }
                metric_list[metric_name]['mean'].append(float(data[1]))
                metric_list[metric_name]['std'].append(float(data[2]))
                metric_list[metric_name]['min'].append(float(data[3]))
                metric_list[metric_name]['max'].append(float(data[4]))
                
            for data in matches_minmax:
                metric_name = prefix + data[0]
                if metric_name not in metric_list:
                    metric_list[metric_name] = {
                        'min': [],
                        'max': [],
                        }
                metric_list[metric_name]['min'].append(float(data[1]))
                metric_list[metric_name]['max'].append(float(data[2]))
                
            for data in matches_single:
                metric_name = prefix + data[0]
                if metric_name not in metric_list:
                    metric_list[metric_name] = {
                        'value': [],
                        }
                metric_list[metric_name]['value'].append(float(data[1]))
            
    return metric_list
        