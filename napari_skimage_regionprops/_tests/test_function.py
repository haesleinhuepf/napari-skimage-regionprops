# from napari_skimage_regionprops import threshold, image_arithmetic
def generate_sphere_mesh(radius, segments):
    """
    Generate vertices and faces for a sphere mesh.

    Parameters:
    radius (float): Radius of the sphere.
    segments (int): Number of segments used to generate the sphere, higher means finer mesh.

    Returns:
    tuple: vertices (N, 3 array), faces (M, 3 array)
    """
    import numpy as np
    theta = np.linspace(0, np.pi, segments)
    phi = np.linspace(0, 2 * np.pi, segments)
    theta, phi = np.meshgrid(theta, phi)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # Generate faces (indices of vertices that make up each triangle)
    faces = []
    for i in range(len(theta) - 1):
        for j in range(len(phi) - 1):
            v0 = i * len(phi) + j
            v1 = v0 + 1
            v2 = v0 + len(phi)
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    faces = np.array(faces)

    return vertices, faces

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
    from napari_skimage_regionprops import visualize_measurement_on_labels, map_measurements_on_labels
    result = visualize_measurement_on_labels(labels_layer, "area", viewer)
    assert result is not None
    result = map_measurements_on_labels(labels_layer, "area", viewer)
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

    points_layer = viewer.add_points(np.array([[2, 1], [2, 5], [5, 3], [6, 6]]))
    vertices, faces = generate_sphere_mesh(1, 50)
    surface_layer = viewer.add_surface((vertices, faces))

    # test loading csv to points
    load_csv("test.csv", points_layer)
    load_csv("test.csv", points_layer, viewer)

    # test loading csv to surface
    load_csv("test.csv", surface_layer)
    load_csv("test.csv", surface_layer, viewer)

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

def test_measure_points():
    import numpy as np
    import napari_skimage_regionprops as nsr

    image = np.random.random((100, 200)) * 10
    points = np.random.random((100, 2)) * 99

    nsr.measure_points(points, image)

def test_2d_labels_in_3d_image():
    import numpy as np
    labels = np.zeros((6,6,6), dtype=np.uint8)
    labels[1:4, 1:4, 1:4] = 1 # 3D label
    labels[4, 1:4, 1:4] = 2 # 2D label

    from napari_skimage_regionprops import regionprops_table
    table = regionprops_table(labels, labels, size=True, intensity=True, perimeter=True, shape=True, position=True,
                moments=True)

    # check one measurement
    assert np.array_equal(table['area'], [27, 9])

    # check that measurements that depend on convex_area are absent in this case
    assert "convex_area" not in table.keys()
    assert "feret_max_diameter" not in table.keys()
    assert "solidity" not in table.keys()

def test_map_measurements_WITHOUT_BG_on_sequential_labels_WITHOUT_BG(make_napari_viewer):
    import numpy as np
    from pandas import DataFrame
    from napari_skimage_regionprops._parametric_images import map_measurements_on_labels
    
    measurements = [6, 9, 12, 15]
    labels = np.array(
        [[1, 2],
         [3, 4]]
         )
    output = np.array(
        [[6, 9],
         [12, 15]]
    )

    table = DataFrame({
        'label': np.unique(labels[labels != 0]),
        'measurements': measurements})

    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(labels, features=table)
    result = map_measurements_on_labels(labels_layer, column='measurements')
    assert np.array_equal(output, result)

def test_map_measurements_WITHOUT_BG_on_NON_sequential_labels_WITHOUT_BG(make_napari_viewer):
    import numpy as np
    from pandas import DataFrame
    from napari_skimage_regionprops._parametric_images import map_measurements_on_labels
    
    measurements = [6, 9, 12, 15]
    labels = np.array(
        [[2, 4],
         [6, 8]]
         )
    output = np.array(
        [[6, 9],
         [12, 15]]
    )

    table = DataFrame({
        'label': np.unique(labels[labels != 0]),
        'measurements': measurements})

    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(labels, features=table)
    result = map_measurements_on_labels(labels_layer, column='measurements')
    assert np.array_equal(output, result)


def test_map_measurements_WITHOUT_BG_on_NON_sequential_labels_WITHOUT_BG_unsorted(make_napari_viewer):
    import numpy as np
    from pandas import DataFrame
    from napari_skimage_regionprops._parametric_images import map_measurements_on_labels

    measurements = [6, 9, 12, 15]
    labels = np.array(
        [[2, 4],
         [6, 8]]
    )
    output = np.array(
        [[9, 6],
         [12, 15]]
    )

    table = DataFrame({
        'label': [4, 2, 6, 8],
        'measurements': measurements})

    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(labels, features=table)
    result = map_measurements_on_labels(labels_layer, column='measurements')
    assert np.array_equal(output, result)


def test_map_measurements_WITHOUT_BG_on_sequential_labels_WITH_BG(make_napari_viewer):
    import numpy as np
    from pandas import DataFrame
    from napari_skimage_regionprops._parametric_images import map_measurements_on_labels
    
    measurements = [6, 9, 12]
    labels = np.array(
        [[0, 1],
         [2, 3]]
         )
    output = np.array(
        [[0, 6],
         [9, 12]]
    )

    table = DataFrame({
        'label': np.unique(labels[labels != 0]),
        'measurements': measurements})

    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(labels, features=table)
    result = map_measurements_on_labels(labels_layer, column='measurements')
    assert np.array_equal(output, result)

def test_map_measurements_WITHOUT_BG_on_NON_sequential_labels_WITH_BG(make_napari_viewer):
    import numpy as np
    from pandas import DataFrame
    from napari_skimage_regionprops._parametric_images import map_measurements_on_labels
    
    measurements = [6, 9, 12]
    labels = np.array(
        [[0, 2],
         [4, 6]]
         )
    output = np.array(
        [[0, 6],
         [9, 12]]
    )

    table = DataFrame({
        'label': np.unique(labels[labels != 0]),
        'measurements': measurements})

    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(labels, features=table)
    result = map_measurements_on_labels(labels_layer, column='measurements')
    assert np.array_equal(output, result)

def test_map_measurements_WITH_BG_on_sequential_labels_WITH_BG(make_napari_viewer):
    import numpy as np
    from pandas import DataFrame
    from napari_skimage_regionprops._parametric_images import map_measurements_on_labels
    
    measurements = [3, 6, 9, 12]
    labels = np.array(
        [[0, 1],
         [2, 3]]
         )
    output = np.array(
        [[3, 6],
         [9, 12]]
    )

    table = DataFrame({
        'label': np.unique(labels),
        'measurements': measurements})

    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(labels, features=table)
    result = map_measurements_on_labels(labels_layer, column='measurements')
    assert np.array_equal(output, result)

def test_map_measurements_WITH_BG_on_NON_sequential_labels_WITH_BG(make_napari_viewer):
    import numpy as np
    from pandas import DataFrame
    from napari_skimage_regionprops._parametric_images import map_measurements_on_labels
    
    measurements = [3, 6, 9, 12]
    labels = np.array(
        [[0, 2],
         [4, 6]]
         )
    output = np.array(
        [[3, 6],
         [9, 12]]
    )

    table = DataFrame({
        'label': np.unique(labels),
        'measurements': measurements})

    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(labels, features=table)
    result = map_measurements_on_labels(labels_layer, column='measurements')
    assert np.array_equal(output, result)
