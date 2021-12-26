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
    ])

    image_layer = viewer.add_image(image)
    labels_layer = viewer.add_labels(image)

    # analyze everything we can
    from napari_skimage_regionprops import regionprops
    regionprops(image_layer, labels_layer, viewer, True, True, True, True, True, True)

    # there is now a table in the viewer
    assert len(viewer.window._dock_widgets) == num_dw + 1

    from napari_skimage_regionprops import get_table
    table_widget = get_table(labels_layer, viewer)
    assert table_widget is not None

    area_measurements = table_widget.get_content()['area']
    print(area_measurements)
    assert np.array_equal([9, 6, 6, 1], area_measurements)

    # generate a parametric image
    from napari_skimage_regionprops import visualize_measurement_on_labels
    visualize_measurement_on_labels(labels_layer, "area")

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
    ])

    labels = image
    # make a 3D dataset
    image = np.asarray([image, image, image])

    image_layer = viewer.add_image(image)
    labels_layer = viewer.add_labels(labels)

    # analyze everything we can
    from napari_skimage_regionprops import regionprops
    regionprops(image_layer, labels_layer, viewer, True, True, True, True, True, True)

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
    ])

    # make a 3D dataset
    image = np.asarray([image, image, image])

    image_layer = viewer.add_image(image)
    labels_layer = viewer.add_labels(image)

    # analyze everything we can
    from napari_skimage_regionprops import regionprops
    regionprops(image_layer, labels_layer, viewer, True, True, True, True, True, True)

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
    regionprops(image_layer, labels_layer, viewer, True, True, True, True, True, True)

    # there is now a table in the viewer
    assert len(viewer.window._dock_widgets) == num_dw + 1

    from napari_skimage_regionprops import get_table
    table_widget = get_table(labels_layer, viewer)
    assert table_widget is not None

    area_measurements = table_widget.get_content()['area']
    print(area_measurements)
    assert np.array_equal([27, 18, 18, 12], area_measurements)

