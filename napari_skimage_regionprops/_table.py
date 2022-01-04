import napari
from pandas import DataFrame
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QTableWidget, QHBoxLayout, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from napari_tools_menu import register_function

from typing import Union


class TableWidget(QWidget):
    """
    The table widget represents a table inside napari.

    Tables are just views on `properties` of `layers` but can be extended so
    that data from other layers can be appended.
    """

    def __init__(self, layer: napari.layers.Layer):
        super().__init__()

        self._layer = layer
        self._name = "Properties of " + layer.name

        self._view = QTableWidget()
        self.set_content(layer.properties)

        self._view.clicked.connect(self._clicked_table)
        layer.mouse_drag_callbacks.append(self._clicked_labels)

        copy_button = QPushButton("Copy to clipboard")
        copy_button.clicked.connect(self._copy_clicked)

        save_button = QPushButton("Save as csv...")
        save_button.clicked.connect(self._save_clicked)

        self.setWindowTitle(self._name)
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

    def _after_labels_clicked(self):
        if "label" in self._table.keys() and hasattr(self._layer,
                                                     "selected_label"):
            row = self._view.currentRow()
            label = self._table["label"][row]
            print("labels clicked, set table", label)
            if label != self._layer.selected_label:
                for r, lbl in enumerate(self._table["label"]):
                    if lbl == self._layer.selected_label:
                        self._view.setCurrentCell(r,
                                                  self._view.currentColumn())
                        break

    # We need to run this later as the labels_layer.selected_label isn't
    # changed yet.
    def _clicked_labels(self, event, event1):
        QTimer.singleShot(200, self._after_labels_clicked)

    def _save_clicked(self, filename=None):
        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(self,"Save as csv...", ".", "*.csv")
        DataFrame(self._table).to_csv(filename)

    def _copy_clicked(self): DataFrame(self._table).to_clipboard()

    def set_content(self, table: dict):
        """
        Overwrite the content of the table.

        Overwrites the content of the internal `_table` data container with the
        content of a given dictionary.
        """
        if table is None:
            table = {}
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
        """Return the current content of the table."""
        return self._table

    def update_content(self):
        """Overwrite current content with data from associated labels layer."""
        self.set_content(self._layer.properties)


@register_function(menu="Measurement > Show table (nsr)")
def add_table(labels_layer: napari.layers.Layer,
              viewer: napari.Viewer) -> TableWidget:
    """
    Add a table to a viewer and return the table widget.

    The table will show the `properties` of the given layer.
    """
    dock_widget = get_table(labels_layer, viewer)
    if dock_widget is None:
        dock_widget = TableWidget(labels_layer)
        # add widget to napari
        viewer.window.add_dock_widget(dock_widget,
                                      area='right',
                                      name="Properties of " + labels_layer.name)
    else:
        dock_widget.set_content(labels_layer.properties)
        if not dock_widget.parent().isVisible():
            dock_widget.parent().setVisible(True)

    return dock_widget


def append_table(table_widget: QTableWidget, table: dict):
    """
    Append data to a currently active table widget in the viewer.

    The function finds the table widget in the `viewer`, the name of which
    matches `name`. The data in `table` is then appended to this table and
    displayed. If the provided data to be added to the table widget fails to
    provide some of the fields, the field is filled with 'NaN'

    Parameters
    ----------
    table_widget : QTableWidget
        Table widget instance in the viewer.
    table : dict
        Data to be appended to the table widget.

    Returns
    -------
    None.

    """
    _table = table_widget._table

    for key in _table.keys():
        if key in table.keys():

            _table[key] += list(table[key])
        else:
            _table[key] += list('NaN')

    table_widget.set_content(_table)

    return None


def get_table(labels_layer: Union[napari.layers.Layer, str],
              viewer: napari.Viewer) -> TableWidget:
    """
    Search for a table widget in the viewer.

    Parameters
    ----------
    labels_layer : typing.Union(napari.layers.Layer, str)
        Can be an label layer that is associated with a table if a
        napari.layers.layer is passed as argument. If the argument is a string,
        the function looks for a table widget with a given name.
    viewer : napari.Viewer
        Instance of the napari viewer.

    Returns
    -------
    TableWidget
        Handle of a discovered TableWidget.

    """

    for widget in list(viewer.window._dock_widgets.values()):
        potential_table_widget = widget.widget()
        if isinstance(potential_table_widget, TableWidget):

            # If table is referred by layer
            if type(labels_layer) == napari.layers.Layer:
                if potential_table_widget._layer is labels_layer:
                    return potential_table_widget

            # If table is referred by name
            elif type(labels_layer) == str:
                if potential_table_widget._name == labels_layer:
                    return potential_table_widget
    return None
