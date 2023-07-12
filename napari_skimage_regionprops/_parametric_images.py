import numpy as np
from napari_tools_menu import register_function
import numpy
from deprecated import deprecated


def create_feature_map(layer: "napari.layers.Layer",
                       selected_column: str,
                       colormap: str = 'jet'
                       ) -> "napari.layers.Layer":
    """
    Create feature map from layer and column name.

    Parameters
    ----------
    layer : "napari.layers.Layer"
        Layer to create feature map from.
    column_name : str
        Column name to create feature map from.

    Returns
    -------
    "napari.layers.Layer"
        Feature map.
    """
    # Label layers
    from napari.layers import Layer, Labels, Points, Vectors, Surface
    properties = {}
    if isinstance(layer, Labels):
        if "label" not in layer.properties.keys():
            raise ValueError("Layer does not have a 'label' property.")
        if selected_column is None:
            return None

        print("Selected column", selected_column)

        data = map_measurements_on_labels(
            layer, selected_column)

        properties['contrast_limits'] = [np.min(layer.features[selected_column]),
                                         np.max(layer.features[selected_column])]
        properties['colormap'] = colormap
        properties['interpolation'] = 'nearest'
        layertype = 'image'

    elif isinstance(layer, Points):
        data = layer.data
        properties['face_color'] = selected_column
        properties['face_colormap'] = colormap
        properties['features'] = {selected_column: layer.features[selected_column].values}
        layertype = 'points'

    elif isinstance(layer, Vectors):
        data = layer.data
        properties['features'] = {selected_column: layer.features[selected_column].values}
        properties['edge_color'] = selected_column
        properties['edge_colormap'] = colormap
        layertype = 'vectors'

    # Surface layer
    elif isinstance(layer, Surface):
        data = list(layer.data)

        # We may have stored features in the metadata to avoid napari complaining
        if not hasattr(layer, "features") and 'features' not in layer.metadata.keys():
            raise ValueError("Layer does not have a 'features' property.")

        if not hasattr(layer, "features") and "features" in layer.metadata.keys():
            layer.features = layer.metadata["features"]
            layer.metadata.pop("features")

        data[2] = np.asarray(layer.features[selected_column].values)

        properties['colormap'] = colormap
        properties['contrast_limits'] = [np.min(layer.features[selected_column]),
                                         np.max(layer.features[selected_column])]
        if "annotation" in selected_column or "CLUSTER_ID" in selected_column:
            properties.colormap = "hsv"
        layertype = 'surface'

    properties['affine'] = layer.affine
    properties['scale'] = layer.scale
    properties['rotate'] = layer.rotate

    return Layer.create(data, properties, layertype)


@register_function(menu="Measurement maps > Measurements on labels (nsr)")
@register_function(menu="Visualization > Measurements on labels (nsr)")
def map_measurements_on_labels(labels_layer:"napari.layers.Labels", column:str = "label", viewer:"napari.Viewer" = None) -> "napari.types.ImageData":
    """
    Visualize a quantiative measurement on a label image by replacing the label IDs with specified table colum values.
    """
    import pandas as pd
    import dask.array as da
    from dask import delayed
    from functools import partial

    labels = labels_layer.data
    table = pd.DataFrame(labels_layer.properties)

    # special treatment for time series
    if len(labels.shape) == 4:
        # determine how the Frame column is called; in case there is any
        frame_column = None
        for potential_frame_column in ['frame', 'Frame']:
            if potential_frame_column in table.keys():
                frame_column = potential_frame_column
                break

        # Relabel one timepoint
        output_sample = relabel_timepoint_with_map_array(labels, table, column, frame_column, 0)

        lazy_arrays = []
        for i in range(labels.shape[0]):
            # build a delayed function call for each timepoint
            lazy_processed_image = delayed(
                partial(relabel_timepoint_with_map_array, labels, table, column, frame_column, i)
            )
            lazy_arrays.append(
                lazy_processed_image()
            )

        # build an array of delayed arrays
        dask_arrays = [
            [da.from_delayed(
                delayed_reader,
                shape=output_sample.shape,
                dtype=output_sample.dtype)]
            if len(output_sample.shape) == 2
            else da.from_delayed(
                delayed_reader,
                shape=output_sample.shape,
                dtype=output_sample.dtype
            )
            for delayed_reader in lazy_arrays
        ]
        # Stack into one large dask.array
        stack = da.stack(
            dask_arrays,
            axis=0)
        return stack
    else:
        label_list = np.asarray(table['label']).tolist()
        measurement_list = np.asarray(table[column]).tolist()

        return relabel_with_map_array(labels, label_list, measurement_list)
    


@deprecated("visualize_measurement_on_labels() is deprecated. Use map_measurements_on_labels() instead")
def visualize_measurement_on_labels(labels_layer:"napari.layers.Labels", column:str = "label", viewer:"napari.Viewer" = None) -> "napari.types.ImageData":
    """
    Visualize a quantiative measurement on a label image by replacing the label IDs with specified table colum values.
    """
    import pandas as pd
    import dask.array as da
    from dask import delayed
    from functools import partial
    from napari.utils import notifications
    if viewer is not None:
        notifications.show_warning("This function is deprecated! To adhere to future behavior and suppress this warning, use 'map_measurements_on_labels' instead (from 'Tools -> Measurement maps -> Measurements on labels (nsr)'.")
    labels = labels_layer.data
    table = pd.DataFrame(labels_layer.properties)

    # special treatment for time series
    if len(labels.shape) == 4:
        # determine how the Frame column is called; in case there is any
        frame_column = None
        for potential_frame_column in ['frame', 'Frame']:
            if potential_frame_column in table.keys():
                frame_column = potential_frame_column
                break

        # Relabel one timepoint
        output_sample = relabel_timepoint(labels, table, column, frame_column, 0)

        lazy_arrays = []
        for i in range(labels.shape[0]):
            # build a delayed function call for each timepoint
            lazy_processed_image = delayed(
                partial(relabel_timepoint, labels, table, column, frame_column, i)
            )
            lazy_arrays.append(
                lazy_processed_image()
            )

        # build an array of delayed arrays
        dask_arrays = [
            [da.from_delayed(
                delayed_reader,
                shape=output_sample.shape,
                dtype=output_sample.dtype)]
            if len(output_sample.shape) == 2
            else da.from_delayed(
                delayed_reader,
                shape=output_sample.shape,
                dtype=output_sample.dtype
            )
            for delayed_reader in lazy_arrays
        ]
        # Stack into one large dask.array
        stack = da.stack(
            dask_arrays,
            axis=0)
        return stack
    else:
        measurements = np.asarray(table[column]).tolist()
        return relabel(labels, measurements)


def relabel_timepoint_with_map_array(labels, table, column, frame_column, timepoint):
    labels_one_timepoint = labels[timepoint]
    if frame_column is not None:
        table_one_timepoint = table[table[frame_column] == timepoint]
    else:
        table_one_timepoint = table

    label_list = np.asarray(table_one_timepoint['label']).tolist()
    measurement_list = np.asarray(table_one_timepoint[column]).tolist()
    return relabel_with_map_array(labels_one_timepoint, label_list, measurement_list)

def relabel_with_map_array(image, label_list, measurement_list):
    """
    Produce parametric map image from a label image, a list of labels and a list of measurements.
    The two lists must provide labels and corresponding values in the same order.

    See also
    --------
    https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.map_array
    """
    from skimage.util import map_array
    return map_array(np.asarray(image), np.asarray(label_list), np.array(measurement_list))

@deprecated("relabel_timepoint() is deprecated. Use relabel_timepoint_with_map_array() instead")
def relabel_timepoint(labels, table, column, frame_column, timepoint):
    labels_one_timepoint = labels[timepoint]
    if frame_column is not None:
        table_one_timepoint = table[table[frame_column] == timepoint]
    else:
        table_one_timepoint = table
    measurements = np.asarray(table_one_timepoint[column]).tolist()
    return relabel(labels_one_timepoint, measurements)

@deprecated("relabel() is deprecated. Use skimage.util.map_array() instead")
def relabel(image, measurements):
    import importlib
    loader = importlib.find_loader("pyclesperanto_prototype")
    found = loader is not None

    if found and len(image.shape) < 4:
        return relabel_cle(image, measurements)
    else:
        return relabel_numpy(image, measurements)

@deprecated("relabel_cle() is deprecated. Use skimage.util.map_array() instead")
def relabel_cle(image, measurements):
    import pyclesperanto_prototype as cle
    return cle.pull(cle.replace_intensities(image, np.insert(np.array(measurements), 0, 0)))

@deprecated("relabel_numpy() is deprecated. Use skimage.util.map_array() instead")
def relabel_numpy(image, measurements):
    return numpy.take(np.insert(np.array(measurements), 0, 0), image)
