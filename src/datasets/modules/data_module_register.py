registered_data_modules = {}

def register(data_module_name):
    def decorator(data_module_class):
        registered_data_modules[data_module_name] = data_module_class
        return data_module_class
    return decorator

def get_data_module(data_module_name):
    if data_module_name not in registered_data_modules:
        raise ValueError(f'Try to get data_module: dataset {data_module_name} not implemented.')
    return registered_data_modules.get(data_module_name)
