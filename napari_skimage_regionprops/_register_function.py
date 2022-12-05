"""
This module replaces the napari_tools_menu.register_function decorator with a
custom one, when the import fails. The new decorator simply discards the menu
argument and then executes the decorated function as is.
"""

try:
    from napari_tools_menu import register_function
except ModuleNotFoundError as e:
    from typing import Callable
    from toolz import curry
    import logging

    logging.warning(e)
    logging.warning("Replace napari_tools_menu.register_function with custom decorator")

    @curry
    def register_function(func: Callable, menu:str, *args, **kwargs) -> Callable:
        return func
