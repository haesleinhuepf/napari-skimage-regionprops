import numpy as np
from napari_tools_menu import register_function

@register_function(menu="Measurement > Load from CSV (nsr)")
def load_csv(csv_filename:"magicgui.types.PathLike", labels_layer:"napari.layers.Labels", viewer:"napari.Viewer"=None):
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

    if hasattr(labels_layer, "properties"):
        labels_layer.properties = edited_reg_props
    if hasattr(labels_layer, "features"):
        labels_layer.features = edited_reg_props

    if viewer is not None:
        from ._table import add_table
        add_table(labels_layer, viewer)
