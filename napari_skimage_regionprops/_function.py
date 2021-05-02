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
    return [regionprops]

def regionprops(image: ImageData, labels: LabelsData, napari_viewer : Viewer, size : bool = True, intensity : bool = True, perimeter : bool = False, shape : bool = False, position : bool = False, moments : bool = False):
    """
    Adds a table widget to a given napari viewer with quantitative analysis results derived from an image-labelimage pair.
    """

    if image is not None and labels is not None:

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
            properties = properties + ['moments', 'moments_central', 'moments_hu']

        # todo:
        # moments_normalized
        # weighted_local_centroid
        # weighted_moments
        # weighted_moments_central
        # weighted_moments_hu
        # weighted_moments_normalized

        table = regionprops_table(np.asarray(labels).astype(int), intensity_image=np.asarray(image),
                                  properties=properties, extra_properties=extra_properties)
        dock_widget = table_to_widget(table)
        napari_viewer.window.add_dock_widget(dock_widget, area='right')
    else:
        warnings.warn("Image and labels must be set.")

def table_to_widget(table: dict) -> QWidget:
    """
    Takes a table given as dictionary with strings as keys and numeric arrays as values and returns a QWidget which
    contains a QTableWidget with that data.
    """

    copy_button = QPushButton("Copy to clipboard")
    def copy_trigger():
        dataframe = DataFrame(table)
        dataframe.to_clipboard()
    copy_button.clicked.connect(copy_trigger)

    save_button = QPushButton("Save as csv...")
    def save_trigger():
        filename, _ = QFileDialog.getSaveFileName(save_button, "Save as csv...", ".", "*.csv")
        dataframe = DataFrame(table)
        dataframe.to_csv(filename)
    save_button.clicked.connect(save_trigger)


    #view = Table(value=table)

    view = QTableWidget(len(next(iter(table.values()))), len(table))
    for i, column in enumerate(table.keys()):
        view.setItem(0, i, QTableWidgetItem(column))
        for j, value in enumerate(table.get(column)):
            view.setItem(j + 1, i, QTableWidgetItem(str(value)))

    widget = QWidget()
    widget.setWindowTitle("region properties")
    widget.setLayout(QGridLayout())
    widget.layout().addWidget(copy_button)
    widget.layout().addWidget(save_button)
    widget.layout().addWidget(view)

    return widget