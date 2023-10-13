registered_models = {}

def register(model_name):
    def decorator(model_class):
        registered_models[model_name] = model_class
        return model_class
    return decorator

def get_model(model_name):
    if model_name not in registered_models:
        raise ValueError(f'Try to get model: architecture {model_name} not implemented.')
    return registered_models.get(model_name)
