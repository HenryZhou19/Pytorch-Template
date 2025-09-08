class Register:
    def __init__(self, name=''):
        self.registered_classes = {}
        self.name = name

    def __call__(self, class_name):
        def decorator(_class):
            self.registered_classes[class_name] = _class
            _class.registered_name = class_name
            return _class
        return decorator

    def get(self, class_name):
        if class_name not in self.registered_classes:
            raise ValueError(f'Try to get {self.name}: {class_name} not implemented.')
        return self.registered_classes.get(class_name)
