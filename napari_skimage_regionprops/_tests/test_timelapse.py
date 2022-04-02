import numpy as np
import napari
from napari_skimage_regionprops import regionprops_table_all_frames, regionprops_table
import pandas as pd

def test_timelapse_analyse_single_timepoint():
    image = np.asarray([
        [[[1, 2], [3, 4]]],
        [[[5, 6], [7, 8]]]
    ])

    labels = np.asarray([
        [[[1, 2], [3, 4]]],
        [[[1, 2], [3, 4]]]
    ])

    stats = regionprops_table(image[0], labels[0], size=False)
    df = pd.DataFrame(stats)

    assert df.shape[0] == 4


def test_timelapse_analyse_all_timepoints():
    image = np.asarray([
        [[[1, 2], [3, 4]]],
        [[[5, 6], [7, 8]]]
    ])

    labels = np.asarray([
        [[[1, 2], [3, 4]]],
        [[[1, 2], [3, 4]]]
    ])

    stats = regionprops_table_all_frames(image, labels, size=False)
    df = pd.DataFrame(stats)

    assert df.shape[0] == 8


def test_timelapse_analyse_single_timepoint_with_viewer(make_napari_viewer):
    viewer = make_napari_viewer()

    image = np.asarray([
        [[[1, 2], [3, 4]]],
        [[[5, 6], [7, 8]]]
    ])

    labels = np.asarray([
        [[[1, 2], [3, 4]]],
        [[[1, 2], [3, 4]]]
    ])

    viewer.add_image(image)
    labels_layer = viewer.add_labels(labels)

    regionprops_table(image, labels, size=False, napari_viewer=viewer)

    df = pd.DataFrame(labels_layer.properties)

    assert df.shape[0] == 4


def test_timelapse_analyse_all_timepoints_with_viewer(make_napari_viewer):
    viewer = make_napari_viewer()

    image = np.asarray([
        [[[1, 2], [3, 4]]],
        [[[5, 6], [7, 8]]]
    ])

    labels = np.asarray([
        [[[1, 2], [3, 4]]],
        [[[1, 2], [3, 4]]]
    ])

    viewer.add_image(image)
    labels_layer = viewer.add_labels(labels)

    regionprops_table_all_frames(image, labels, size=False, napari_viewer=viewer)

    df = pd.DataFrame(labels_layer.properties)

    assert df.shape[0] == 8

