registered_criterions = {}

def register(criterion_name):
    def decorator(criterion_class):
        registered_criterions[criterion_name] = criterion_class
        return criterion_class
    return decorator

def get_criterion(criterion_name):
    if criterion_name not in registered_criterions:
        raise ValueError(f'Try to get criterion: architecture {criterion_name} not implemented.')
    return registered_criterions.get(criterion_name)
