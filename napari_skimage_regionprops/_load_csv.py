import numpy as np
from napari_tools_menu import register_function

try:
    import napari
except Exception as e:
    import warnings
    warnings.warn(str(e))

@register_function(menu="Measurement > Load from CSV (nsr)")
def load_csv(csv_filename:"magicgui.types.PathLike", labels_layer: "napari.layers.Labels", show_table: bool = True, viewer: "napari.Viewer" = None):
    """Save contents of a CSV file into a given layer's properties"""
    import pandas as pd

    # preload to get find datatypes:
    preload_reg_probs = pd.read_csv(csv_filename, nrows=2)
    dtypes={}
    for c_i, c in enumerate(preload_reg_probs.columns):
        if preload_reg_probs.dtypes[c_i] == np.float64:
            dtypes[c] = np.single
        else:
            dtypes[c] = preload_reg_probs.dtypes[c_i]


    #load region properties from csv file
    reg_props = pd.read_csv(csv_filename, dtype=dtypes)
    try:
        reg_props = reg_props.drop(["Unnamed: 0"], axis=1)
    except KeyError:
        reg_props = reg_props

    if "label" not in reg_props.keys().tolist():
        label_column = pd.DataFrame(
            {"label": np.array(range(1, (len(reg_props) + 1)))}
        )
        reg_props = pd.concat([label_column, reg_props], axis=1)

    if hasattr(labels_layer, "properties"):
        labels_layer.properties = reg_props
    if hasattr(labels_layer, "features"):
        labels_layer.features = reg_props
    if show_table is False:
        labels_layer.metadata["limit_number_rows"] = 0
    else:
        labels_layer.metadata["limit_number_rows"] = 500

    if viewer is not None:
        from ._table import add_table
        add_table(labels_layer, viewer)
