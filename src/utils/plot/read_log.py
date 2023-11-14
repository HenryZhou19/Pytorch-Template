import re


def get_metrics_from_log(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    pattern = re.compile(r'(\w+): \(([\d.]+) ± ([\d.]+)\) \[([\d.]+), ([\d.]+)\]')

    metric_list = {}
    for block in log_content.split('\n\n'):
        block = block.strip('\n')
        if block.startswith('Train') or block.startswith('Eval'):
            prefix = 'train_' if block.startswith('Train') else 'eval_'
            
            matches = pattern.findall(block)
            for data in matches:
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
                
    return metric_list
        