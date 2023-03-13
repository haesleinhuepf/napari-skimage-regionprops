
def test_napari_regionprops_map_2channels_2D(make_napari_viewer):
    import numpy as np
    from napari_skimage_regionprops import regionprops_measure_relationship_to_other_channels
    
    viewer = make_napari_viewer()
    
    num_dw = len(viewer.window._dock_widgets)

    image_2D = np.arange(12).reshape(3,4)

    # Create multichannel image
    multichannel_image = np.stack([image_2D,
                                   image_2D], axis=-1)
    viewer.add_image(multichannel_image, channel_axis=-1)

    # Create two label images (a reference and a target)
    ref_labels = np.array([[0, 1, 1, 1],
                           [0, 1, 1, 1],
                           [0, 1, 1, 1]])
    target_labels = np.array([[3, 0, 0, 0],
                              [0, 2, 2, 0],
                              [2, 2, 2, 1]])

    viewer.add_labels(ref_labels, name='ref_labels')
    viewer.add_labels(target_labels, name='target_labels')

    widget = regionprops_measure_relationship_to_other_channels()
    regionprops_measure_relationship_to_other_channels_function = widget._function
    # Measure everything we can
    table = regionprops_measure_relationship_to_other_channels_function(
        ref_labels,
        multichannel_image[..., 0],
        [target_labels],
        [multichannel_image[..., 1]],
        intensity = True,
        things_inside_things = True,
        size = True,
        perimeter = True,
        shape = True,
        position = True,
        moments = True,
        counts = True,
        average = True,
        std = True,
        minimum = True,
        percentile_25 = True,
        median = True,
        percentile_75 = True,
        maximum = True,
        napari_viewer = viewer)

    # there is now a table in the viewer
    assert len(viewer.window._dock_widgets) == num_dw + 1
    
    assert table.shape == (2, 597)
    # Check counts
    assert np.array_equal(table['counts_target_labels'].values, np.array([1., 2.]))
    # Check area average
    assert np.array_equal(table['area_target_labels average'].values, np.array([1., 3.]))
    # Check mean_intensity median
    assert np.array_equal(table['mean_intensity_target_labels 50%'].values, np.array([0., 9.3]))

def test_napari_regionprops_map_3channels_2D(make_napari_viewer):
    import numpy as np
    from napari_skimage_regionprops import regionprops_measure_relationship_to_other_channels
    
    viewer = make_napari_viewer()
    
    num_dw = len(viewer.window._dock_widgets)

    image_2D = np.arange(12).reshape(3,4)
    # Create multichannel image
    multichannel_image = np.stack([image_2D,
                                image_2D,
                                image_2D], axis=-1)
    viewer.add_image(multichannel_image, channel_axis=-1)

    # Create 3 label images (a reference and 2 targets)
    ref_labels = np.array([[0, 1, 1, 1],
                           [0, 1, 1, 1],
                           [0, 1, 1, 1]])
    target_labels = np.array([[3, 0, 0, 0],
                              [0, 2, 2, 0],
                              [2, 2, 2, 1]])
    target_labels_B = np.array([[1, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 0]])

    viewer.add_labels(ref_labels, name='ref_labels')
    viewer.add_labels(target_labels, name='target_labels')
    viewer.add_labels(target_labels_B, name='target_labels_B')

    widget = regionprops_measure_relationship_to_other_channels()
    regionprops_measure_relationship_to_other_channels_function = widget._function

    # Measure everything we can
    table = regionprops_measure_relationship_to_other_channels_function(
        ref_labels,
        multichannel_image[..., 0],
        [target_labels, target_labels_B],
        [multichannel_image[..., 1], multichannel_image[..., 2]],
        intensity = True,
        things_inside_things = True,
        size = True,
        perimeter = True,
        shape = True,
        position = True,
        moments = True,
        counts = True,
        average = True,
        std = True,
        minimum = True,
        percentile_25 = True,
        median = True,
        percentile_75 = True,
        maximum = True,
        napari_viewer = viewer)

    # there is now a table in the viewer
    assert len(viewer.window._dock_widgets) == num_dw + 1
    
    assert table.shape == (2, 1193)
    # Check counts
    assert np.array_equal(table['counts_target_labels'].values, np.array([1., 2.]))
    assert np.array_equal(table['counts_target_labels_B'].values, np.array([1., 0.]))
    # Check equivalent_diameter 25% percentile
    assert np.allclose(table['equivalent_diameter_target_labels 25%'], np.array([1.128379, 1.477067]))
    assert np.allclose(table['equivalent_diameter_target_labels_B 25%'], np.array([1.954410, np.nan]), equal_nan=True)

    # Check max_intensity max
    assert np.allclose(table['max_intensity_target_labels max'], np.array([0., 11.]), equal_nan=True)
    assert np.allclose(table['max_intensity_target_labels_B max'], np.array([4., np.nan]), equal_nan=True)

def test_napari_regionprops_map_2channels_3D(make_napari_viewer):
    import numpy as np
    from napari_skimage_regionprops import regionprops_measure_relationship_to_other_channels
    
    viewer = make_napari_viewer()
    
    num_dw = len(viewer.window._dock_widgets)

    image_3D = np.arange(27).reshape(3,3,3)

    # Create multichannel image
    multichannel_image = np.stack([image_3D,
                                image_3D])
    viewer.add_image(multichannel_image, channel_axis=0, rgb=False)

    # Create 2 label images (a reference and a target)
    ref_labels_2D = np.array(
        [[0, 1, 1],
         [0, 1, 1],
         [0, 0, 1]])
    ref_labels_3D = np.stack(
        [ref_labels_2D,
        ref_labels_2D,
        ref_labels_2D]
    )
    target_labels_2D = np.array(
        [[2, 2, 2],
         [1, 2, 0],
         [1, 1, 0]])
    target_labels_3D = np.stack(
        [target_labels_2D,
        target_labels_2D,
        target_labels_2D]
    )

    viewer.add_labels(ref_labels_3D, name='ref_labels_3D')
    viewer.add_labels(target_labels_3D, name='target_labels_3D')

    widget = regionprops_measure_relationship_to_other_channels()
    regionprops_measure_relationship_to_other_channels_function = widget._function

    # Measure everything we can
    table = regionprops_measure_relationship_to_other_channels_function(
        ref_labels_3D,
        multichannel_image[0],
        [target_labels_3D],
        [multichannel_image[1]],
        intensity = True,
        things_inside_things = True,
        size = True,
        perimeter = True,
        shape = True,
        position = True,
        moments = True,
        counts = True,
        average = True,# 3 channels 2D
        std = True,
        minimum = True,
        percentile_25 = True,
        median = True,
        percentile_75 = True,
        maximum = True,
        napari_viewer = viewer)

    # there is now a table in the viewer
    assert len(viewer.window._dock_widgets) == num_dw + 1
    
    assert table.shape == (2, 1549)
    # Check counts
    assert np.array_equal(table['counts_target_labels_3D'].values, np.array([1., 1.]))
    # Check local_centroid-2 average
    assert np.allclose(table['local_centroid-2_target_labels_3D average'].values, np.array([0.333333, 1.]))
    # Check min_intensity min
    assert np.array_equal(table['min_intensity_target_labels_3D min'].values, np.array([3., 0.]))
