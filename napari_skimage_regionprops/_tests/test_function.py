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
    regionprops(image_layer, labels_layer, viewer, True, True, True, True, True, True)

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
    regionprops(image_layer, labels_layer, viewer, True, True, True, True, True, False)

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
    ]).astype(float)

    # make a 3D dataset
    image = np.asarray([image, image, image])

    image_layer = viewer.add_image(image)
    labels_layer = viewer.add_labels(image.astype(int))

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
    table = regionprops_table(labels, labels, None, False, False, False, True)

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
