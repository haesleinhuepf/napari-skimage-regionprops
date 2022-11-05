import time

import napari
from pandas import DataFrame
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QTableWidget, QHBoxLayout, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from napari_tools_menu import register_function

import pandas as pd
from typing import Union
import numpy as np


class TableWidget(QWidget):
    """
    The table widget represents a table inside napari.
    Tables are just views on `properties` of `layers`.
    """
    def __init__(self, layer: napari.layers.Layer, viewer:napari.Viewer = None):
        super().__init__()

        self._layer = layer
        self._viewer = viewer

        self._view = QTableWidget()
        self._view.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        if hasattr(layer, "properties"):
            self.set_content(layer.properties)
        else:
            self.set_content({})

        self._view.clicked.connect(self._clicked_table)
        self._view.horizontalHeader().sectionDoubleClicked.connect(self._double_clicked_table)
        layer.mouse_drag_callbacks.append(self._clicked_labels)

        copy_button = QPushButton("Copy to clipboard")
        copy_button.clicked.connect(self._copy_clicked)

        save_button = QPushButton("Save as csv...")
        save_button.clicked.connect(self._save_clicked)

        self.setWindowTitle("Properties of " + layer.name)
        self.setLayout(QGridLayout())
        action_widget = QWidget()
        action_widget.setLayout(QHBoxLayout())
        action_widget.layout().addWidget(copy_button)
        action_widget.layout().addWidget(save_button)
        self.layout().addWidget(action_widget)
        self.layout().addWidget(self._view)
        action_widget.layout().setSpacing(3)
        action_widget.layout().setContentsMargins(0, 0, 0, 0)

    def _clicked_table(self):
        if "label" in self._table.keys():
            row = self._view.currentRow()
            label = self._table["label"][row]
            print("Table clicked, set label", label)
            self._layer.selected_label = label

            frame_column = _determine_frame_column(self._table)
            if frame_column is not None and self._viewer is not None:
                frame = self._table[frame_column][row]
                current_step = list(self._viewer.dims.current_step)
                if len(current_step) >= 4:
                    current_step[-4] = frame
                    self._viewer.dims.current_step = current_step

    def _double_clicked_table(self):
        if "label" in self._table.keys():
            selected_column = list(self._table.keys())[self._view.currentColumn()]
            print("Selected column", selected_column)
            if selected_column is not None:
                if isinstance(self._layer, napari.layers.Labels):
                    from ._parametric_images import visualize_measurement_on_labels
                    new_layer = self._viewer.add_image(visualize_measurement_on_labels(self._layer, selected_column, self._viewer),
                                           name=selected_column + " in " + self._layer.name)
                    new_layer.contrast_limits = [np.min(self._table[selected_column]), np.max(self._table[selected_column])]
                    new_layer.colormap = "jet"
                elif isinstance(self._layer, napari.layers.Points):
                    features = self._layer.features
                    new_layer = self._viewer.add_points(self._layer.data,
                                                        features=features,
                                                        face_color=selected_column,
                                                        face_colormap="jet",
                                                        size=self._layer.size,
                                                        name=selected_column + " in " + self._layer.name
                                                        )
                    new_layer.contrast_limits = [np.min(self._table[selected_column]), np.max(self._table[selected_column])]

        if "vertex_index" in self._table.keys():
            selected_column = list(self._table.keys())[self._view.currentColumn()]
            print("Selected column (T)", selected_column)
            if selected_column is not None and isinstance(self._layer, napari.layers.Surface):
                values = np.asarray(self._table[selected_column])
                data = self._layer.data
                data = [np.asarray(data[0]).copy(), np.asarray(data[1]).copy(), values]

                new_layer = self._viewer.add_surface(data,
                    name=selected_column + " in " + self._layer.name)
                new_layer.contrast_limits = [np.min(self._table[selected_column]), np.max(self._table[selected_column])]
                if "annotation" in selected_column or "CLUSTER_ID" in selected_column:
                    new_layer.colormap = "hsv"
                else:
                    new_layer.colormap = "jet"

    def _after_labels_clicked(self):
        if "label" in self._table.keys() and hasattr(self._layer, "selected_label"):
            row = self._view.currentRow()
            label = self._table["label"][row]

            frame_column = _determine_frame_column(self._table)
            if frame_column is not None and self._viewer is not None:
                current_step = list(self._viewer.dims.current_step)
                if len(current_step) >= 4:
                    frame = current_step[-4]

            if label != self._layer.selected_label:
                if frame_column is not None and self._viewer is not None:
                    for r, (l, f) in enumerate(zip(self._table["label"], self._table[frame_column])):
                        if l == self._layer.selected_label and f == frame:
                            self._view.setCurrentCell(r, self._view.currentColumn())
                            break
                else:
                    for r, l in enumerate(self._table["label"]):
                        if l == self._layer.selected_label:
                            self._view.setCurrentCell(r, self._view.currentColumn())
                            break

    # We need to run this later as the labels_layer.selected_label isn't changed yet.
    def _clicked_labels(self, event, event1): QTimer.singleShot(200, self._after_labels_clicked)

    def _save_clicked(self, event=None, filename=None):
        if filename is None: filename, _ = QFileDialog.getSaveFileName(self, "Save as csv...", ".", "*.csv")
        DataFrame(self._table).to_csv(filename)

    def _copy_clicked(self): DataFrame(self._table).to_clipboard()

    def set_content(self, table : dict):
        """
        Overwrites the content of the table with the content of a given dictionary.
        """
        if table is None:
            table = {}

        # Workaround to fix wrong row display in napari status bar
        # https://github.com/napari/napari/issues/4250
        # https://github.com/napari/napari/issues/2596
        if "label" in table.keys() and "index" not in table.keys():
            table["index"] = table["label"]

            # workaround until this issue is fixed:
            # https://github.com/napari/napari/issues/4342
            if len(np.unique(table['index'])) != len(table['index']):
                # indices exist multiple times, presumably because it's timelapse data
                def get_status(
                        position,
                        *,
                        view_direction=None,
                        dims_displayed=None,
                        world: bool = False,
                ) -> str:
                    value = self._layer.get_value(
                        position,
                        view_direction=view_direction,
                        dims_displayed=dims_displayed,
                        world=world,
                    )

                    from napari.utils.status_messages import generate_layer_status
                    msg = generate_layer_status(self._layer.name, position, value)
                    return msg

                self._layer.get_status = get_status

                import warnings
                warnings.warn('Status bar display of label properties disabled because labels/indices exist multiple times (napari-skimage-regionprops)')

        self._table = table

        self._layer.properties = table

        self._view.clear()
        try:
            self._view.setRowCount(len(next(iter(table.values()))))
            self._view.setColumnCount(len(table))
        except StopIteration:
            pass

        for i, column in enumerate(table.keys()):

            self._view.setHorizontalHeaderItem(i, QTableWidgetItem(column))
            for j, value in enumerate(table.get(column)):
                self._view.setItem(j, i, QTableWidgetItem(str(value)))

    def get_content(self) -> dict:
        """
        Returns the current content of the table
        """
        return self._table

    def update_content(self):
        """
        Read the content of the table from the associated labels_layer and overwrites the current content.
        """
        self.set_content(self._layer.properties)

    def append_content(self, table: Union[dict, DataFrame], how: str = 'outer'):
        """
        Append data to table.

        Parameters
        ----------
        table : Union[dict, DataFrame]
            New data to be appended.
        how : str, OPTIONAL
            Method how to join the data. See also https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html
        Returns
        -------
        None.
        """
        # Check input type
        if not isinstance(table, DataFrame):
            table = DataFrame(table)

        _table = DataFrame(self._table)

        # Check whether there are common columns and switch merge type accordingly
        common_columns = np.intersect1d(table.columns, _table.columns)
        if len(common_columns) == 0:
            table = pd.concat([table, _table])
        else:
            table = pd.merge(table, _table, how=how, copy=False)

        self.set_content(table.to_dict('list'))


@register_function(menu="Measurement > Show table (nsr)")
def add_table(labels_layer: napari.layers.Layer, viewer:napari.Viewer) -> TableWidget:
    """
    Add a table to a viewer and return the table widget. The table will show the `properties` of the given layer.
    """
    dock_widget = get_table(labels_layer, viewer)
    if dock_widget is None:
        dock_widget = TableWidget(labels_layer, viewer)
        # add widget to napari
        viewer.window.add_dock_widget(dock_widget, area='right', name="Properties of " + labels_layer.name)
    else:
        dock_widget.set_content(labels_layer.properties)
        if not dock_widget.parent().isVisible():
            dock_widget.parent().setVisible(True)

    return dock_widget

def get_table(labels_layer: napari.layers.Layer, viewer:napari.Viewer) -> TableWidget:
    """
    Searches inside a viewer for a given table and returns it. If it cannot find it,
    it will return None.
    """
    import warnings
    # see: https://github.com/napari/napari/issues/3944
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for widget in list(viewer.window._dock_widgets.values()):
            potential_table_widget = widget.widget()
            if isinstance(potential_table_widget, TableWidget):
                if potential_table_widget._layer is labels_layer:
                    return potential_table_widget

    return None

def _determine_frame_column(table):
    candidates = ["Frame", "frame"]
    for c in candidates:
        if c in table.keys():
            return c
    return None