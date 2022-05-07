from napari_tools_menu import register_function
import numpy

@register_function(menu="Visualization > Measurements on labels (nsr)")
def visualize_measurement_on_labels(labels_layer:"napari.layers.Labels", column:str = "label", viewer:"napari.Viewer" = None) -> "napari.types.ImageData":
    import pandas as pd

    labels = labels_layer.data
    table = pd.DataFrame(labels_layer.properties)

    if len(labels.shape) == 4:
        current_timepoint = viewer.dims.current_step[0]
        labels = labels[current_timepoint]

        if "frame" in table.keys():
            table = table[table['frame'] == current_timepoint]

    measurements = table[column]

    if isinstance(measurements, numpy.ndarray):
        measurements = measurements.tolist()

    try:
        import pyclesperanto_prototype as cle
        return cle.pull(cle.replace_intensities(labels, numpy.asarray([0] + measurements)))
    except ImportError:
        return relabel_numpy(labels, measurements)
    except ModuleNotFoundError:
        return relabel_numpy(labels, measurements)


def relabel_numpy(image, measurements):
    return numpy.take(numpy.array([0] + measurements), image)
