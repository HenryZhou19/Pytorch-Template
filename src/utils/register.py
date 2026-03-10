import ast
import importlib
import os


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


def scan_register_classes(py_dir: str, register_type: str):
    '''
    Not using register instance, but scan the source code to find all classes decorated with @register('name').
    Just for speed and convenience, so that we don't need to import all modules to get the register table.
    '''
    results = {}
    for fn in os.listdir(py_dir):
        if not fn.endswith('.py') or fn.startswith('_'):
            continue
        file_path = os.path.join(py_dir, fn)
        module_name = os.path.splitext(fn)[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        try:
            tree = ast.parse(source, filename=fn)
        except Exception as e:
            print(f'Parse error in {fn}: {e}')
            continue
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                for deco in node.decorator_list:
                    if (isinstance(deco, ast.Call) and
                        isinstance(deco.func, ast.Name) and
                        deco.func.id == register_type and
                        len(deco.args) >= 1 and
                        isinstance(deco.args[0], ast.Constant) and
                        isinstance(deco.args[0].value, str)):
                        register_name = deco.args[0].value
                        class_name = node.name
                        if register_name in results:
                            raise ValueError(f'Duplicate register name: `{register_name}` in {register_type}')
                        results[register_name] = (module_name, class_name)
    return results


def get_registered_class(register_dict: dict, name: str, package: str = None):
    '''
    Get the registered class by name, and import the module if needed.
    '''
    module, class_name = register_dict[name]
    mod = importlib.import_module(f'.{module}', package=package)
    return getattr(mod, class_name)
