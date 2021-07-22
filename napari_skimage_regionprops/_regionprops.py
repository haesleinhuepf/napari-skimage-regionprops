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
    return [regionprops, duplicate_current_frame]

def duplicate_current_frame(image: ImageData, napari_viewer : Viewer, axis : int = 0) -> ImageData:
    current_dim_value = napari_viewer.dims.current_step[axis]
    return np.take(image, current_dim_value, axis)

def regionprops(image: ImageData, labels: LabelsData, napari_viewer : Viewer, size : bool = True, intensity : bool = True, perimeter : bool = False, shape : bool = False, position : bool = False, moments : bool = False):
    """
    Adds a table widget to a given napari viewer with quantitative analysis results derived from an image-labelimage pair.
    """

    if image is not None and labels is not None:

        # deal with dimensionality of data
        if len(image.shape) > len(labels.shape):
            dim = 0
            subset = ""
            while len(image.shape) > len(labels.shape):
                current_dim_value = napari_viewer.dims.current_step[dim]
                dim = dim + 1
                image = image[current_dim_value]
                subset = subset + ", " + str(current_dim_value)
            warnings.warn("Not the full image was analysed, just the subset [" + subset[2:] + "] according to selected timepoint / slice.")


        properties = ['label']
        extra_properties = []

        if size:
            properties = properties + ['area', 'bbox_area', 'convex_area', 'equivalent_diameter']

        if intensity:
            properties = properties + ['max_intensity', 'mean_intensity', 'min_intensity']

            # arguments must be in the specified order, matching regionprops
            def standard_deviation_intensity(region, intensities):
                return np.std(intensities[region])

            extra_properties.append(standard_deviation_intensity)

        if perimeter:
            properties = properties + ['perimeter', 'perimeter_crofton']

        if shape:
            properties = properties + ['major_axis_length', 'minor_axis_length', 'orientation', 'solidity', 'eccentricity', 'extent', 'feret_diameter_max', 'local_centroid']
            # euler_number

        if position:
            properties = properties + ['centroid', 'bbox', 'weighted_centroid']

        if moments:
            properties = properties + ['moments', 'moments_central', 'moments_hu', 'moments_normalized']

        # todo:
        # weighted_local_centroid
        # weighted_moments
        # weighted_moments_central
        # weighted_moments_hu
        # weighted_moments_normalized

        # quantitative analysis using scikit-image's regionprops
        table = regionprops_table(np.asarray(labels).astype(int), intensity_image=np.asarray(image),
                                  properties=properties, extra_properties=extra_properties)

        # turn table into a widget
        dock_widget = table_to_widget(table)

        # add widget to napari
        napari_viewer.window.add_dock_widget(dock_widget, area='right')
    else:
        warnings.warn("Image and labels must be set.")

def table_to_widget(table: dict) -> QWidget:
    """
    Takes a table given as dictionary with strings as keys and numeric arrays as values and returns a QWidget which
    contains a QTableWidget with that data.
    """
    view = Table(value=table)

    copy_button = QPushButton("Copy to clipboard")

    @copy_button.clicked.connect
    def copy_trigger():
        view.to_dataframe().to_clipboard()

    save_button = QPushButton("Save as csv...")

    @save_button.clicked.connect
    def save_trigger():
        filename, _ = QFileDialog.getSaveFileName(save_button, "Save as csv...", ".", "*.csv")
        view.to_dataframe().to_csv(filename)

    widget = QWidget()
    widget.setWindowTitle("region properties")
    widget.setLayout(QGridLayout())
    widget.layout().addWidget(copy_button)
    widget.layout().addWidget(save_button)
    widget.layout().addWidget(view.native)

    return widget