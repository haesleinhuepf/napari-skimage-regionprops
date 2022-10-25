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
    
    return relabel(labels, measurements)

def relabel(image, measurements):
    import importlib
    loader = importlib.find_loader("pyclesperanto_prototype")
    found = loader is not None

    if found:
        return relabel_cle(image, measurements)
    else:
        return relabel_numpy(image, measurements)

def relabel_cle(image, measurements):
    import pyclesperanto_prototype as cle
    return cle.pull(cle.replace_intensities(image, np.insert(np.array(measurements), 0, 0)))

def relabel_numpy(image, measurements):
    return numpy.take(np.insert(np.array(measurements), 0, 0), image)
