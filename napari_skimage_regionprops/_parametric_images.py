import numpy as np
from napari_tools_menu import register_function
import numpy
from deprecated import deprecated

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
        measurements = np.asarray(table[column]).tolist()
        
        # Ensure background measurements are present in measurements
        # if also present in labels
        if (0 not in table['label'].values) and (0 in labels):
            measurements.insert(0, 0)
        return relabel_skimage(labels, measurements)
    


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
    measurements = np.asarray(table_one_timepoint[column]).tolist()
    return relabel_skimage(labels_one_timepoint, measurements)

def relabel_skimage(image, measurements):
    from skimage.util import map_array
    return map_array(image, np.unique(image), np.array(measurements))

def relabel_timepoint(labels, table, column, frame_column, timepoint):
    labels_one_timepoint = labels[timepoint]
    if frame_column is not None:
        table_one_timepoint = table[table[frame_column] == timepoint]
    else:
        table_one_timepoint = table
    measurements = np.asarray(table_one_timepoint[column]).tolist()
    return relabel(labels_one_timepoint, measurements)

def relabel(image, measurements):
    import importlib
    loader = importlib.find_loader("pyclesperanto_prototype")
    found = loader is not None

    if found and len(image.shape) < 4:
        return relabel_cle(image, measurements)
    else:
        return relabel_numpy(image, measurements)

def relabel_cle(image, measurements):
    import pyclesperanto_prototype as cle
    return cle.pull(cle.replace_intensities(image, np.insert(np.array(measurements), 0, 0)))

def relabel_numpy(image, measurements):
    return numpy.take(np.insert(np.array(measurements), 0, 0), image)
