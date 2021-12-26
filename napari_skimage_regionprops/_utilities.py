import numpy as np
from napari_plugin_engine import napari_hook_implementation
from napari.types import ImageData
from napari import Viewer
from napari_tools_menu import register_function

@napari_hook_implementation
def napari_experimental_provide_function():
    return [duplicate_current_frame]

@register_function(menu="Utilities > Duplicate current timepoint")
def duplicate_current_frame(image: ImageData, napari_viewer : Viewer, axis : int = 0) -> ImageData:
    current_dim_value = napari_viewer.dims.current_step[axis]
    return np.take(image, current_dim_value, axis)
