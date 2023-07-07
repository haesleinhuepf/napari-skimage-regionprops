import numpy as np


def test_label_featuremaps(make_napari_viewer):
    from napari_skimage_regionprops import regionprops
    from .._table import create_feature_map

    viewer = make_napari_viewer()
    
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

    # analyze a few things
    regionprops(image_layer, labels_layer, True, False, False, False, False, False, viewer)
    feature_map = create_feature_map(viewer.layers[-1], 'area')

    assert feature_map.data.max() == 9


def test_vector_featuremaps(make_napari_viewer):
    from napari_skimage_regionprops import regionprops
    from .._table import create_feature_map
    import pandas as pd

    viewer = make_napari_viewer()

    np.random.seed(0)
    vectors = np.random.random((10, 2, 3))
    feature1 = np.random.random((10))
    feature2 = np.random.random((10))
    features = pd.DataFrame({'feature1': feature1, 'feature2': feature2})
    layer = viewer.add_vectors(vectors, features=features)

    feature_map = create_feature_map(layer, 'feature1')
    viewer.add_layer(feature_map)


def test_points_featuremaps(make_napari_viewer):
    from napari_skimage_regionprops import regionprops
    from .._table import create_feature_map
    import pandas as pd

    viewer = make_napari_viewer()

    np.random.seed(0)
    points = np.random.random((10, 2))
    feature1 = np.random.random((10))
    feature2 = np.random.random((10))
    features = pd.DataFrame({'feature1': feature1, 'feature2': feature2})
    layer = viewer.add_points(points, features=features)

    feature_map = create_feature_map(layer, 'feature1')
    viewer.add_layer(feature_map)


def test_surface_featuremaps(make_napari_viewer):
    from napari_skimage_regionprops import regionprops
    from .._table import create_feature_map
    import pandas as pd

    viewer = make_napari_viewer()

    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(size=(10, 3), high=9, low=0)
    surface = (vertices, faces)
    feature1 = np.random.random((10))
    feature2 = np.random.random((10))
    features = pd.DataFrame({'feature1': feature1, 'feature2': feature2})
    layer = viewer.add_surface(surface, metadata={'features': features})

    feature_map = create_feature_map(layer, 'feature1')
    assert np.array_equal(feature_map.data[2], feature1)
