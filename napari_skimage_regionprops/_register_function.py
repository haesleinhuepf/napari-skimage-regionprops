"""
This module mimicks yhe napari_tools_menu.register_function decorator, but it
simply discards the menu argument and then executes the decorated function.It
is meant as a fall-back option for when napari_tools_menu cannot be imported.
"""


try:
    from napari_tools_menu import register_function
except ModuleNotFoundError as e:
    from typing import Callable
    from toolz import curry
    import logging
    logging.warning(e)
    logging.warning("Replace napari_tools_menu.register_function with custom decorator")
    #from ._custom_register_function_decorator import register_function
    @curry
    def register_function(func: Callable, menu:str, *args, **kwargs) -> Callable:
        return func
