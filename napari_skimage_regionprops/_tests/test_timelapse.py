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

    from napari_skimage_regionprops import visualize_measurement_on_labels, visualize_measurement_on_labels_with_map_array

    visualize_measurement_on_labels(labels_layer, column="mean_intensity", viewer=viewer)

    visualize_measurement_on_labels_with_map_array(labels_layer, column="mean_intensity", viewer=viewer)

def test_frame_variable_in_timelapse(make_napari_viewer):
    viewer = make_napari_viewer()

    import numpy as np
    import pandas as pd
    from skimage.data import binary_blobs
    from skimage.measure import regionprops_table, label
    from napari_skimage_regionprops._table import add_table

    # a random image with 3 timepoints
    image = np.random.rand(3, 128, 128)

    # and a segmentation for the 3 timepoints
    segmentation = np.stack([label(binary_blobs(128, volume_fraction=0.25)) for _ in range(3)])
    print(image.shape, segmentation.shape)

    # compute the features for each timepoint
    features = []
    for t, (seg, im) in enumerate(zip(segmentation, image)):
        feats = regionprops_table(seg, im, properties=("label", "mean_intensity"))
        # add the frame column
        feats["frame"] = np.full(len(feats["label"]), t)
        features.append(pd.DataFrame(feats))
    features = pd.concat(features, axis=0)

    viewer.add_image(image)
    label_layer = viewer.add_labels(segmentation)
    label_layer.features = features
    add_table(label_layer, viewer)
