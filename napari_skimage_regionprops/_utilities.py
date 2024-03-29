import numpy as np
try:
    from napari_plugin_engine import napari_hook_implementation
    from napari.types import ImageData
    import napari
    from napari.layers import Image, Labels, Layer
    from typing_extensions import Annotated
    LayerInput = Annotated[Layer, {"label": "Image"}]
except Exception as e:
    import warnings
    warnings.warn(str(e))
    LayerInput = None

from napari_tools_menu import register_function


@napari_hook_implementation
def napari_experimental_provide_function():
    return [duplicate_current_frame]

@register_function(menu="Utilities > Duplicate current timepoint (nsr)")
def duplicate_current_frame(layer : LayerInput, napari_viewer: "napari.Viewer", axis : int = 0) -> "napari.layers.Layer":
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
