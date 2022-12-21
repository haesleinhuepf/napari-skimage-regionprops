import numpy as np
import napari
from napari_skimage_regionprops import napari_regionprops_map_channels_table

seed = 42
np.random.seed(seed)
multichannel_image = np.random.randint(0, 20, size = 72, ).reshape(2, 6, 6)

ref_labels = np.zeros((6,6), dtype=int)
ref_labels[2:5, 2:5] = 1

probe_labels = np.zeros((6,6), dtype=int)
probe_labels[3:5, 2:4] = 2
probe_labels[4, 1] = 2
probe_labels[4, 4] = 1
probe_labels[1, 1] = 3

multichannel_labels = np.stack([ref_labels, probe_labels], axis=0)

viewer = napari.Viewer()
viewer.add_image(multichannel_image, channel_axis=0, colormap = 'gray')
viewer.add_labels(multichannel_labels[0])
viewer.add_labels(multichannel_labels[1])

# widget = magicgui(napari_regionprops_map_channels_table)
widget = napari_regionprops_map_channels_table()

viewer.window.add_dock_widget(widget)
