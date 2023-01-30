# from napari_skimage_regionprops import threshold, image_arithmetic

# add your tests here...
def test_regionprops(make_napari_viewer):

    viewer = make_napari_viewer()


    num_dw = len(viewer.window._dock_widgets)


    import numpy as np

    image = np.asarray([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 3, 0, 0],
        [0, 0, 3, 3, 3, 0, 4],
    ]).astype(float)

    image_layer = viewer.add_image(image)
    labels_layer = viewer.add_labels(image.astype(int))

    # analyze everything we can
    from napari_skimage_regionprops import regionprops
    regionprops(image_layer, labels_layer, True, True, True, True, True, True, viewer)

    # there is now a table in the viewer
    assert len(viewer.window._dock_widgets) == num_dw + 1

    from napari_skimage_regionprops import get_table
    table_widget = get_table(labels_layer, viewer)
    assert table_widget is not None

    # save content
    table_widget._save_clicked(filename="test.csv")

    # select a cell, click the table and read out selected label
    table_widget._view.setCurrentCell(1, 1)
    table_widget._clicked_table()
    assert labels_layer.selected_label == 2

    # select a label, click the layer, read out selected row
    labels_layer.selected_label = 3
    table_widget._after_labels_clicked()
    assert table_widget._view.currentRow() == 2

    # check table results
    area_measurements = table_widget.get_content()['area']
    print(area_measurements)
    assert np.array_equal([9, 6, 6, 1], area_measurements)

    # generate a parametric image
    from napari_skimage_regionprops import visualize_measurement_on_labels
    result = visualize_measurement_on_labels(labels_layer, "area", viewer)
    assert result is not None

    reference = np.asarray([
        [0, 0, 0, 0, 0, 0, 0],
        [9, 9, 9, 0, 0, 6, 6],
        [9, 9, 9, 0, 0, 6, 6],
        [9, 9, 9, 0, 0, 6, 6],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 6, 6, 6, 0, 0],
        [0, 0, 6, 6, 6, 0, 1],
    ])
    print("result: ", result)
    print("reference: ", np.asarray(reference))
    assert np.array_equal(result, reference)

    # replace table
    from napari_skimage_regionprops import add_table
    add_table(labels_layer, viewer)

    # Append table
    fake_measurement = area_measurements * 2
    fake_area = {'Double area': fake_measurement}
    table_widget.append_content(fake_area)
    assert 'Double area' in table_widget.get_content().keys()

    # save table to disk
    import pandas as pd
    pd.DataFrame(labels_layer.properties).to_csv("test.csv")
    from napari_skimage_regionprops import load_csv
    load_csv("test.csv", labels_layer)
    load_csv("test.csv", labels_layer, viewer)


    # empty table
    table_widget.set_content(None)
    table_widget.update_content()

# add your tests here...
def test_append_table_by_merging(make_napari_viewer):

    viewer = make_napari_viewer()

    import numpy as np

    image = np.asarray([
        [0, 0],
        [0, 1],
    ])

    labels_layer = viewer.add_labels(image)

    table1 = {
        "A":[1,2,4],
        "B":[1,2,4]
    }
    table2 = {
        "B":[1,2,4],
        "C":[1,2,4]
    }
    labels_layer.properties = table1

    # Append table
    from napari_skimage_regionprops import add_table
    table_widget = add_table(labels_layer, viewer)
    table_widget.append_content(table2)
    assert 'A' in table_widget.get_content().keys()
    assert 'B' in table_widget.get_content().keys()
    assert 'C' in table_widget.get_content().keys()

def test_regionprops_without_moments(make_napari_viewer):

    viewer = make_napari_viewer()

    import numpy as np

    image = np.asarray([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 3, 4, 4],
        [0, 0, 3, 3, 3, 4, 4],
    ]).astype(float)

    # make 3D stack
    image = np.asarray([image, image, image])

    image_layer = viewer.add_image(image)
    labels_layer = viewer.add_labels(image.astype(int))

    # analyze everything we can
    from napari_skimage_regionprops import regionprops
    regionprops(image_layer, labels_layer, size=True, intensity=True, perimeter=True, shape=True, position=True, moments=False, napari_viewer=viewer)

def test_3d_2d(make_napari_viewer):

    viewer = make_napari_viewer()


    num_dw = len(viewer.window._dock_widgets)


    import numpy as np

    image = np.asarray([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 3, 0, 0],
        [0, 0, 3, 3, 3, 0, 4],
    ]).astype(float)

    labels = image
    # make a 3D dataset
    image = np.asarray([image, image, image])

    image_layer = viewer.add_image(image)
    labels_layer = viewer.add_labels(labels.astype(int))

    # analyze everything we can
    from napari_skimage_regionprops import regionprops
    regionprops(image_layer, labels_layer, size=True, intensity=True, perimeter=True, shape=True, position=True,
                moments=True, napari_viewer=viewer)

    # there is now a table in the viewer
    assert len(viewer.window._dock_widgets) == num_dw + 1

    from napari_skimage_regionprops import get_table
    table_widget = get_table(labels_layer, viewer)
    assert table_widget is not None

    area_measurements = table_widget.get_content()['area']
    print(area_measurements)
    assert np.array_equal([9, 6, 6, 1], area_measurements)


def test_3d(make_napari_viewer):

    viewer = make_napari_viewer()


    num_dw = len(viewer.window._dock_widgets)


    import numpy as np

    image = np.asarray([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 3, 4, 4],
        [0, 0, 3, 3, 3, 4, 4],
    ]).astype(float)

    # make a 3D dataset
    image = np.asarray([image, image, image])

    image_layer = viewer.add_image(image)
    labels_layer = viewer.add_labels(image.astype(int))

    # analyze everything we can
    from napari_skimage_regionprops import regionprops
    regionprops(image_layer, labels_layer, size=True, intensity=True, perimeter=True, shape=True, position=True,
                moments=True, napari_viewer=viewer)

    # there is now a table in the viewer
    assert len(viewer.window._dock_widgets) == num_dw + 1

    from napari_skimage_regionprops import get_table
    table_widget = get_table(labels_layer, viewer)
    assert table_widget is not None

    area_measurements = table_widget.get_content()['area']
    print(area_measurements)
    assert np.array_equal([27, 18, 18, 12], area_measurements)


def test_3d_dask(make_napari_viewer):

    viewer = make_napari_viewer()
    import dask.array as da

    num_dw = len(viewer.window._dock_widgets)


    import numpy as np

    image = np.asarray([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 3, 4, 4],
        [0, 0, 3, 3, 3, 4, 4],
    ]).astype(float)

    # make a 3D dataset
    image = da.asarray(np.asarray([image, image, image]))
    labels_image = da.asarray(image.astype(int))

    image_layer = viewer.add_image(image)
    labels_layer = viewer.add_labels(labels_image)

    # analyze everything we can
    from napari_skimage_regionprops import regionprops
    regionprops(image_layer, labels_layer, size=True, intensity=True, perimeter=True, shape=True, position=True,
                moments=True, napari_viewer=viewer)

    # there is now a table in the viewer
    assert len(viewer.window._dock_widgets) == num_dw + 1

    from napari_skimage_regionprops import get_table
    table_widget = get_table(labels_layer, viewer)
    assert table_widget is not None

    area_measurements = table_widget.get_content()['area']
    print(area_measurements)
    assert np.array_equal([27, 18, 18, 12], area_measurements)



def test_4d_3d(make_napari_viewer):

    viewer = make_napari_viewer()


    num_dw = len(viewer.window._dock_widgets)


    import numpy as np

    image = np.asarray([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 3, 4, 4],
        [0, 0, 3, 3, 3, 4, 4],
    ])

    # make a 3D dataset
    image = np.asarray([image, image, image])
    labels = image

    # make 4D timelapse
    image = np.asarray([image, image, image])

    image_layer = viewer.add_image(image)
    labels_layer = viewer.add_labels(labels)

    # analyze everything we can
    from napari_skimage_regionprops import regionprops
    regionprops(image_layer, labels_layer, size=True, intensity=True, perimeter=True, shape=True, position=True,
                moments=True, napari_viewer=viewer)

    # there is now a table in the viewer
    assert len(viewer.window._dock_widgets) == num_dw + 1

    from napari_skimage_regionprops import get_table
    table_widget = get_table(labels_layer, viewer)
    assert table_widget is not None

    area_measurements = table_widget.get_content()['area']
    print(area_measurements)
    assert np.array_equal([27, 18, 18, 12], area_measurements)

    # test duplicating the current frame
    from napari_skimage_regionprops._utilities import duplicate_current_frame
    new_layer = duplicate_current_frame(image_layer, viewer)

    assert len(new_layer.data.shape) == 3
    assert np.array_equal(new_layer.data.shape, [3, 7, 7])


    # test duplicating the current frame
    from napari_skimage_regionprops._utilities import duplicate_current_frame
    new_layer = duplicate_current_frame(labels_layer, viewer)
    print("LL", labels_layer.data.shape)
    print("NL", new_layer.data.shape)
    assert len(new_layer.data.shape) == 2
    assert np.array_equal(new_layer.data.shape, [7, 7])



def test_napari_api():
    from napari_skimage_regionprops import napari_experimental_provide_function
    napari_experimental_provide_function()

def test_napari_api2():
    from napari_skimage_regionprops._utilities import napari_experimental_provide_function
    napari_experimental_provide_function()

def test_shape_descriptors():
    import numpy as np
    labels = np.asarray([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 2, 2, 2, 3, 3],
        [1, 1, 0, 2, 2, 2, 3, 3],
        [0, 0, 0, 2, 2, 2, 3, 3],
        [0, 0, 0, 0, 0, 0, 3, 3],
        [0, 4, 4, 4, 4, 4, 0, 0],
        [0, 4, 4, 4, 4, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [5, 5, 5, 5, 5, 0, 6, 6],
        [5, 5, 5, 5, 5, 0, 6, 6],
        [5, 5, 5, 5, 5, 0, 6, 6],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])

    from napari_skimage_regionprops import regionprops_table
    table = regionprops_table(labels, labels, False, False, False, True)

    print(table.keys())
    assert "area" not in table.keys()
    assert "perimeter" not in table.keys()

    print("aspect_ratio", table['aspect_ratio'])
    assert np.allclose(table['aspect_ratio'], [1., 1., 2.23606798, 2.82842712, 1.73205081, 1.63299316])

    print("circularity", table['circularity'])
    assert np.allclose(table['circularity'], [3.14159265, 1.76714587, 1.57079633, 1.25663706, 1.30899694, 2.0943951 ])

    print("roundness", table['roundness'])
    # Values > 1 should actually not appear, but do so in case of very small objects apparently
    assert np.allclose(table['roundness'], [1.27323954, 1.07429587, 0.50929582, 0.39788736, 0.59683104, 0.71619724])

def test_napari_regionprops_map_2channels_2D(make_napari_viewer):
    import numpy as np
    from napari_skimage_regionprops import napari_regionprops_map_channels_table
    
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

    viewer.add_labels(ref_labels)
    viewer.add_labels(target_labels)

    widget = napari_regionprops_map_channels_table()
    regionprops_map_channels_table = widget._function
    # Measure everything we can
    table = regionprops_map_channels_table(
        ref_labels,
        multichannel_image[..., 0],
        [target_labels],
        [multichannel_image[..., 1]],
        intensity = True,
        multichannel = True,
        size = True,
        perimeter = True,
        shape = True,
        position = True,
        moments = True,
        counts = True,
        mean = True,
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
    # Check area mean
    assert np.array_equal(table['area_target_labels mean'].values, np.array([1., 3.]))
    # Check mean_intensity median
    assert np.array_equal(table['mean_intensity_target_labels 50%'].values, np.array([0., 9.3]))

def test_napari_regionprops_map_3channels_2D(make_napari_viewer):
    import numpy as np
    from napari_skimage_regionprops import napari_regionprops_map_channels_table
    
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
                    

    viewer.add_labels(ref_labels)
    viewer.add_labels(target_labels)
    viewer.add_labels(target_labels_B)

    widget = napari_regionprops_map_channels_table()
    regionprops_map_channels_table = widget._function

    # Measure everything we can
    table = regionprops_map_channels_table(
        ref_labels,
        multichannel_image[..., 0],
        [target_labels, target_labels_B],
        [multichannel_image[..., 1], multichannel_image[..., 2]],
        intensity = True,
        multichannel = True,
        size = True,
        perimeter = True,
        shape = True,
        position = True,
        moments = True,
        counts = True,
        mean = True,
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
    from napari_skimage_regionprops import napari_regionprops_map_channels_table
    
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

                

    viewer.add_labels(ref_labels_3D)
    viewer.add_labels(target_labels_3D)

    widget = napari_regionprops_map_channels_table()
    regionprops_map_channels_table = widget._function

    # Measure everything we can
    table = regionprops_map_channels_table(
        ref_labels_3D,
        multichannel_image[0],
        [target_labels_3D],
        [multichannel_image[1]],
        intensity = True,
        multichannel = True,
        size = True,
        perimeter = True,
        shape = True,
        position = True,
        moments = True,
        counts = True,
        mean = True,# 3 channels 2D
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
    # Check local_centroid-2 mean
    assert np.allclose(table['local_centroid-2_target_labels_3D mean'].values, np.array([0.333333, 1.]))
    # Check min_intensity min
    assert np.array_equal(table['min_intensity_target_labels_3D min'].values, np.array([3., 0.]))
