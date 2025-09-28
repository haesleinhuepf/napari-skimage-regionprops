import numpy as np
from napari_tools_menu import register_function
from napari.utils.notifications import show_warning

try:
    import napari
except Exception as e:
    import warnings
    warnings.warn(str(e))


@register_function(menu="Measurement > Load from CSV (nsr)")
def load_csv(csv_filename:"magicgui.types.PathLike", labels_layer: "napari.layers.Layer", show_table: bool = True, viewer: "napari.Viewer" = None):
    """Save contents of a CSV file into a given layer's properties"""
    import pandas as pd
    # load region properties from csv file
    reg_props = pd.read_csv(csv_filename)
    try:
        edited_reg_props = reg_props.drop(["Unnamed: 0"], axis=1)
    except KeyError:
        edited_reg_props = reg_props

    if "label" not in edited_reg_props.keys().tolist():
        label_column = pd.DataFrame(
            {"label": np.array(range(1, (len(edited_reg_props) + 1)))}
        )
        edited_reg_props = pd.concat([label_column, edited_reg_props], axis=1)
    # Add 'properties' and 'features' attributes if layer is Surface
    if isinstance(labels_layer, napari.layers.Surface):
        # Check if features and properties are already defined
        if not hasattr(labels_layer, "features"):
            labels_layer.features = {}
        if not hasattr(labels_layer, "properties"):
            labels_layer.properties = {}
    elif isinstance(labels_layer, napari.layers.Points):
        # Table shape must match number of points
        if edited_reg_props.shape[0] < labels_layer.data.shape[0]:
            # fill in missing rows with NaNs
            show_warning("Number of rows in CSV file is less than number of points in the layer. Filling in missing rows with NaNs.")
            edited_reg_props = pd.concat([edited_reg_props, pd.DataFrame(np.nan, index=range(labels_layer.data.shape[0] - edited_reg_props.shape[0]), columns=edited_reg_props.columns)]).reset_index(drop=True)
        elif edited_reg_props.shape[0] > labels_layer.data.shape[0]:
            # truncate if necessary
            show_warning("Number of rows in CSV file is greater than number of points in the layer. Truncating to match number of points.")
            edited_reg_props = edited_reg_props.iloc[:labels_layer.data.shape[0],:]
    if hasattr(labels_layer, "properties"):
        labels_layer.properties = edited_reg_props.to_dict(orient="list")
    if hasattr(labels_layer, "features"):
        labels_layer.features = edited_reg_props

    if viewer is not None and show_table:
        from ._table import add_table
        add_table(labels_layer, viewer)
