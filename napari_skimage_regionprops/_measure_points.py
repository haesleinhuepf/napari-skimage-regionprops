import numpy as np
from napari_tools_menu import register_function

@register_function(menu = "Measurement tables > Measure intensity at point coordinates (nsr)")
def measure_points(points: "napari.types.PointsData", intensity_image: "napari.types.ImageData",
                   viewer: "napari.Viewer" = None):
    """
    After rounding a list of point coordinates, the intensity at specified (rounded) points will be measured and stored
    in a table. In this table, point with `label=1` corresponds to the first point.
    """
    num_dimensions = len(points[0])
    if num_dimensions > 3:
        raise RuntimeError("Points with more than 3 dimensions are not supported (yet).")

    original_points = points
    points = np.asarray(points)
    coordinates = (points + 0.5).astype(int)
    intensities = intensity_image.take(np.ravel_multi_index(coordinates.T, intensity_image.shape))

    result_dict = {
        'label': np.asarray(list(range(len(points)))) + 1,
        'intensity': intensities
    }
    axis_names = ['z', 'y', 'x']
    for d in range(num_dimensions):
        dimension_index = -1 - d
        point_coordinate = points[:, dimension_index]
        axis_name = axis_names[dimension_index]
        result_dict[axis_name] = point_coordinate

    for d in range(num_dimensions):
        dimension_index = -1 - d
        int_coordinate = coordinates[:, dimension_index]
        axis_name = axis_names[dimension_index] + "_int"
        result_dict[axis_name] = int_coordinate

    if viewer is not None:
        # store the layer for saving results later
        from napari_workflows._workflow import _get_layer_from_data
        points_layer = _get_layer_from_data(viewer, original_points)

        # Store results in the properties dictionary:
        points_layer.properties = result_dict

        # turn table into a widget
        from ._table import add_table
        add_table(points_layer, viewer)
    else:
        import pandas as pd
        return pd.DataFrame(result_dict)

