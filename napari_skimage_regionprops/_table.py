import napari
from pandas import DataFrame
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QTableWidget, QHBoxLayout, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from napari_tools_menu import register_function

class TableWidget(QWidget):
    """
    The table widget represents a table inside napari.
    Tables are just views on `properties` of `layers`.
    """
    def __init__(self, labels_layer: napari.layers.Labels):
        super().__init__()

        self._labels_layer = labels_layer

        self._view = QTableWidget()
        self.set_content(labels_layer.properties)

        @self._view.clicked.connect
        def clicked_table():
            row = self._view.currentRow()
            label = self._table["label"][row]
            print("Table clicked, set label", label)
            labels_layer.selected_label = label

        def after_labels_clicked():
            row = self._view.currentRow()
            label = self._table["label"][row]
            print("labels clicked, set table", label)
            if label != labels_layer.selected_label:
                for r, l in enumerate(self._table["label"]):
                    if l == labels_layer.selected_label:
                        self._view.setCurrentCell(r, self._view.currentColumn())
                        break

        @labels_layer.mouse_drag_callbacks.append
        def clicked_labels(event, event1):
            # We need to run this later as the labels_layer.selected_label isn't changed yet.
            QTimer.singleShot(200, after_labels_clicked)

        copy_button = QPushButton("Copy to clipboard")

        @copy_button.clicked.connect
        def copy_trigger():
            DataFrame(self._table).to_clipboard()

        save_button = QPushButton("Save as csv...")

        @save_button.clicked.connect
        def save_trigger():
            filename, _ = QFileDialog.getSaveFileName(save_button, "Save as csv...", ".", "*.csv")
            DataFrame(self._table).to_csv(filename)

        self.setWindowTitle("Properties of " + labels_layer.name)
        self.setLayout(QGridLayout())
        action_widget = QWidget()
        action_widget.setLayout(QHBoxLayout())
        action_widget.layout().addWidget(copy_button)
        action_widget.layout().addWidget(save_button)
        self.layout().addWidget(action_widget)
        self.layout().addWidget(self._view)
        action_widget.layout().setSpacing(3)
        action_widget.layout().setContentsMargins(0, 0, 0, 0)

    def set_content(self, table : dict):
        """
        Overwrites the content of the table with the content of a given dictionary.
        """
        self._table = table
        if self._table is None:
            self._table = {}

        self._labels_layer.properties = table

        self._view.clear()
        self._view.setRowCount(len(next(iter(table.values()))))
        self._view.setColumnCount(len(table))

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
        self.set_content(self._labels_layer.properties)

@register_function(menu="Measurement > Show table (nsr)")
def add_table(labels_layer: napari.layers.Labels, viewer:napari.Viewer) -> TableWidget:
    """
    Add a table to a viewer and return the table widget. The table will show the `properties` of the given layer.
    """

    dock_widget = get_table(labels_layer, viewer)
    if dock_widget is None:
        dock_widget = TableWidget(labels_layer)
        # add widget to napari
        viewer.window.add_dock_widget(dock_widget, area='right', name="Properties of " + labels_layer.name)
    else:
        dock_widget.set_content(labels_layer.properties)
        if not dock_widget.parent().isVisible():
            dock_widget.parent().setVisible(True)

    return dock_widget

def get_table(labels_layer: napari.layers.Labels, viewer:napari.Viewer) -> TableWidget:
    """
    Searches inside a viewer for a given table and returns it. If it cannot find it,
    it will return None.
    """
    for widget in list(viewer.window._dock_widgets.values()):
        potential_table_widget = widget.widget()
        if isinstance(potential_table_widget, TableWidget):
            if potential_table_widget._labels_layer is labels_layer:
                return potential_table_widget
    return None
