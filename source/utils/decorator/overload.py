import inspect
import typing
from typing import Any, OrderedDict

def _parse_func_signature(func_signature: OrderedDict):
    r"""
    Parse function signature to get the parameter types.
    """
    parsed = list()
    for value in func_signature.values():
        if value.annotation is not inspect._empty:
            parsed.append(value.annotation)
        else:
            parsed.append(Any)
    return tuple(parsed)

class Namespace:
    __instance = None
    def __init__(self):
        if not Namespace.__instance:
            # function_map = {func_class: {func_module: {func_name: list(tuple(func_param_dtype, func), ...)}}}
            self.function_map = dict()
            Namespace.__instance = self
        else:
            raise Exception("You cannot create another Singleton class")
    
    @staticmethod
    def get_instance():
        if Namespace.__instance is None:
            Namespace()
        return Namespace.__instance
    
    def register(self, func_class, func_module, func_name, func_param_dtype, func):
        r"""
        Register function to the function map.
        """
        func_module_dict = self.function_map.setdefault(func_class, dict())
        func_name_dict = func_module_dict.setdefault(func_module, dict())
        param_dtype_list = func_name_dict.setdefault(func_name, [])

        already_registered = False
        for param_dtype, _ in param_dtype_list:
            if param_dtype == func_param_dtype:
                already_registered = True
                raise Exception(f"Function '{func_name}' of parameter types {param_dtype} already registered")

        if not already_registered:
            param_dtype_list.append(tuple([func_param_dtype, func]))

    @staticmethod
    def _match_func(func_params, search_func_param_dtypes_list):
        r"""
        Match function param dtype with search_func_param_dtypes_list
        """
        for search_func_param_dtypes, search_func in search_func_param_dtypes_list:
            is_match = True
            for param, search_func_param_dtype in zip(func_params, search_func_param_dtypes):
                if search_func_param_dtype is Any:
                    continue
                if not isinstance(param, search_func_param_dtype):
                    is_match = False
                    break
            if is_match:
                return search_func
        return None

    def get(self, func_class, func_module, func_name, func_params):
        r"""
        Traverse the function map to get the matching function.
        """
        current_dict = self.function_map
        browse_sequence = [func_class, func_module, func_name]

        for key in browse_sequence:
            current_dict = current_dict.get(key)
            if current_dict is None:
                return None

            if key == func_name:
                search_func_param_dtypes_list = current_dict
                return self._match_func(func_params, search_func_param_dtypes_list)

        return None
class Overload:
    r"""
    A decorator class that allows overloading of functions based on their parameter types.
    ATTENTION: This decorator class does not support method overloading.
    """
    def __init__(self, func):
        self._func_class = func.__class__
        self._func_module = func.__module__
        self._func_name = func.__name__
        self._func_param_dtype = _parse_func_signature(
            inspect.signature(func).parameters.copy())
        self._func = func

        Namespace.get_instance().register(
            self._func_class,
            self._func_module,
            self._func_name,
            self._func_param_dtype,
            self._func
        )

    def __call__(self, *args, **kwargs):
        func = Namespace.get_instance().get(
            func_class=self._func_class,
            func_module=self._func_module,
            func_name=self._func_name,
            func_params=args
        )
        if func is None:
            raise Exception("No such function")
        return func(*args, **kwargs)


class Foo:
    def __init__(self):
        self.one = 1
        self.two = 2
    
    @Overload
    def number(self, a: int):
        print(self.one)
    
    @Overload
    def number(self, b:str):
        print(self.two)

if __name__ == "__main__":
    foo = Foo()
    foo.number(1)
    foo.number("a")
    print(Namespace.get_instance().function_map)