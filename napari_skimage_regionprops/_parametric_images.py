import numpy as np
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

    measurements = np.asarray(table[column]).tolist()

    import importlib
    loader = importlib.find_loader("pyclesperanto_prototype")
    found = loader is not None

    if found:
        import pyclesperanto_prototype as cle
        return cle.pull(cle.replace_intensities(labels, numpy.asarray([0] + measurements)))
    else:
        return relabel_numpy(labels, measurements)


def relabel_numpy(image, measurements):
    return numpy.take(numpy.array([0] + measurements), image)
