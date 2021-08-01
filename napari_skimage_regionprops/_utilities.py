import warnings

import numpy as np
from magicgui.widgets import Table
from napari_plugin_engine import napari_hook_implementation
from napari.types import ImageData, LabelsData, LayerDataTuple
from napari import Viewer
from pandas import DataFrame
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from skimage.measure import regionprops_table

@napari_hook_implementation
def napari_experimental_provide_function():
    return [duplicate_current_frame]

def duplicate_current_frame(image: ImageData, napari_viewer : Viewer, axis : int = 0) -> ImageData:
    current_dim_value = napari_viewer.dims.current_step[axis]
    return np.take(image, current_dim_value, axis)
