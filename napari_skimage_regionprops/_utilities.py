import numpy as np
from napari_plugin_engine import napari_hook_implementation
from ._register_function import register_function

from typing_extensions import Annotated
from napari.layers import Image, Labels, Layer
LayerInput = Annotated[Layer, {"label": "Image"}]

@napari_hook_implementation
def napari_experimental_provide_function():
    return [duplicate_current_frame]

@register_function(menu="Utilities > Duplicate current timepoint (nsr)")
def duplicate_current_frame(layer : LayerInput, napari_viewer : "napari.Viewer", axis : int = 0) -> Layer:
    image = layer.data
    current_dim_value = napari_viewer.dims.current_step[axis]
    new_image = np.take(image, current_dim_value, axis)
    new_name = layer.name + "[t=" + str(current_dim_value) + "]"

    if isinstance(layer, Labels):
        result = Labels(new_image, name=new_name)
    else:
        result = Image(new_image, name=new_name)

    return result

def isimage(value):
    return isinstance(value, np.ndarray) or str(type(value)) in ["<class 'cupy._core.core.ndarray'>",
                                                          "<class 'dask.array.core.Array'>",
                                                          "<class 'pyclesperanto_prototype._tier0._pycl.OCLArray'>"]
