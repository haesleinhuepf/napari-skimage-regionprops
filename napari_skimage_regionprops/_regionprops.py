import warnings

import numpy as np
import pandas
try:
    import napari
except Exception as e:
    warnings.warn(str(e))

from napari_tools_menu import register_function
import math
from ._all_frames import analyze_all_frames

def regionprops(image_layer: "napari.layers.Layer", labels_layer: "napari.layers.Labels", size : bool = True, intensity : bool = True, perimeter : bool = False, shape : bool = False, position : bool = False, moments : bool = False, napari_viewer : "napari.Viewer" = None):
    warnings.warn("napari_skimage_regionprops.regionprops is deprecated. Use regionprops_table instead.")
    image_data = None
    if image_layer is not None:
        image_data = image_layer.data

    regionprops_table(image_data, labels_layer.data, size, intensity, perimeter, shape, position, moments, napari_viewer)

@register_function(menu="Measurement tables > Regionprops (scikit-image, nsr)")
def regionprops_table(image: "napari.types.ImageData", labels: "napari.types.LabelsData", size : bool = True, intensity : bool = True, perimeter : bool = False, shape : bool = False, position : bool = False, moments : bool = False, napari_viewer : "napari.Viewer" = None) -> "pandas.DataFrame":
    """
    Adds a table widget to a given napari viewer with quantitative analysis results derived from an image-label pair.
    """
    current_dim_value = 0
    if napari_viewer is not None:
        current_dim_value = napari_viewer.dims.current_step[0]

        # store the layer for saving results later
        from napari_workflows._workflow import _get_layer_from_data
        labels_layer = _get_layer_from_data(napari_viewer, labels)

        # deal with 4D data
        if len(image.shape) == 4:
            image = image[current_dim_value]
        if len(labels.shape) == 4:
            labels = labels[current_dim_value]

    # deal with dimensionality of data
    if len(image.shape) > len(labels.shape):
        dim = 0
        subset = ""
        while len(image.shape) > len(labels.shape):
            dim = dim + 1
            image = image[current_dim_value]
            subset = subset + ", " + str(current_dim_value)
        warnings.warn("Not the full image was analysed, just the subset [" + subset[2:] + "] according to selected timepoint / slice.")



    properties = ['label']
    extra_properties = []

    if size:
        properties = properties + ['area', 'bbox_area', 'equivalent_diameter']
        # Workaround to avoid existing scikit-image bug
        # See https://github.com/scikit-image/scikit-image/issues/6432
        if (len(labels.shape) == 3) and (_check_2D_labels_in_3D_images(labels)):
            warnings.warn("2D labels are present in 3D label image, 'convex_area' not calculated")
        else:
            properties = properties + ['convex_area']
    if intensity:
        properties = properties + ['max_intensity', 'mean_intensity', 'min_intensity']

        # arguments must be in the specified order, matching regionprops
        def standard_deviation_intensity(region, intensities):
            return np.std(intensities[region])

        extra_properties.append(standard_deviation_intensity)

    if perimeter:
        if len(labels.shape) == 2:
            properties = properties + ['perimeter', 'perimeter_crofton']
        else:
            warnings.warn("Perimeter measurements are not supported in 3D")

    if shape:
        properties = properties + ['extent', 'local_centroid']
        # Workaround to avoid existing scikit-image bug
        # See https://github.com/scikit-image/scikit-image/issues/6432
        if (len(labels.shape) == 3) and (_check_2D_labels_in_3D_images(labels)):
            warnings.warn("2D labels are present in 3D label image, 'feret_diameter_max' and 'solidity' not calculated")
        else:
            properties = properties + ['solidity', 'feret_diameter_max']
        if len(labels.shape) == 2:
            properties = properties + ['major_axis_length', 'minor_axis_length', 'orientation', 'eccentricity']
            # we need these two to compute some shape descriptors
            if not size:
                properties = properties + ['area']
            if not perimeter:
                properties = properties + ['perimeter']
        else:
            properties = properties + ['moments_central']
        # euler_number,

    if position:
        properties = properties + ['centroid', 'bbox', 'weighted_centroid']

    if moments:
        properties = properties + ['moments', 'moments_normalized']
        if 'moments_central' not in properties:
            properties = properties + ['moments_central']
        if len(labels.shape) == 2:
            properties = properties + ['moments_hu']

    # todo:
    # weighted_local_centroid
    # weighted_moments
    # weighted_moments_central
    # weighted_moments_hu
    # weighted_moments_normalized
    # quantitative analysis using scikit-image's regionprops
    from skimage.measure import regionprops_table as sk_regionprops_table
    table = sk_regionprops_table(np.asarray(labels).astype(int), intensity_image=np.asarray(image),
                              properties=properties, extra_properties=extra_properties)

    if shape:
        if len(labels.shape) == 2:
            # See https://imagej.nih.gov/ij/docs/menus/analyze.html
            table['aspect_ratio'] = table['major_axis_length'] / table['minor_axis_length']
            table['roundness'] = 4 * table['area'] / np.pi / pow(table['major_axis_length'], 2)
            table['circularity'] = 4 * np.pi * table['area'] / pow(table['perimeter'], 2)

        # 3D image
        if len(labels.shape) == 3:
            axis_lengths_0 = []
            axis_lengths_1 = []
            axis_lengths_2 = []
            for i in range(len(table['moments_central-0-0-0'])):
                table_temp = { # ugh
                    'moments_central-0-0-0': table['moments_central-0-0-0'][i],
                    'moments_central-2-0-0': table['moments_central-2-0-0'][i],
                    'moments_central-0-2-0': table['moments_central-0-2-0'][i],
                    'moments_central-0-0-2': table['moments_central-0-0-2'][i],
                    'moments_central-1-1-0': table['moments_central-1-1-0'][i],
                    'moments_central-1-0-1': table['moments_central-1-0-1'][i],
                    'moments_central-0-1-1': table['moments_central-0-1-1'][i]
                }
                axis_lengths = ellipsoid_axis_lengths(table_temp)
                axis_lengths_0.append(axis_lengths[0]) # ugh
                axis_lengths_1.append(axis_lengths[1])
                axis_lengths_2.append(axis_lengths[2])

            table["minor_axis_length"] = axis_lengths_2
            table["intermediate_axis_length"] = axis_lengths_1
            table["major_axis_length"] = axis_lengths_0

            if not moments:
                # remove moment from table as we didn't ask for them
                table = {k: v for k, v in table.items() if not 'moments_central' in k}

        if not size:
            table = {k: v for k, v in table.items() if k != 'area'}
        if not perimeter:
            table = {k: v for k, v in table.items() if k != 'perimeter'}

    if napari_viewer is not None:
        # Store results in the properties dictionary:
        labels_layer.properties = table

        # turn table into a widget
        from ._table import add_table
        add_table(labels_layer, napari_viewer)
    else:
        import pandas
        return pandas.DataFrame(table)

def ellipsoid_axis_lengths(table):
    """Compute ellipsoid major, intermediate and minor axis length.

    Adapted from https://forum.image.sc/t/scikit-image-regionprops-minor-axis-length-in-3d-gives-first-minor-radius-regardless-of-whether-it-is-actually-the-shortest/59273/2

    Parameters
    ----------
    table from regionprops containing moments_central

    Returns
    -------
    axis_lengths: tuple of float
        The ellipsoid axis lengths in descending order.
    """


    m0 = table['moments_central-0-0-0']
    sxx = table['moments_central-2-0-0'] / m0
    syy = table['moments_central-0-2-0'] / m0
    szz = table['moments_central-0-0-2'] / m0
    sxy = table['moments_central-1-1-0'] / m0
    sxz = table['moments_central-1-0-1'] / m0
    syz = table['moments_central-0-1-1'] / m0
    S = np.asarray([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    # determine eigenvalues in descending order
    eigvals = np.sort(np.linalg.eigvalsh(S))[::-1]
    return tuple([math.sqrt(20.0 * np.clip(e, a_min=0, a_max=None)) for e in eigvals])

regionprops_table_all_frames = analyze_all_frames(regionprops_table)
register_function(regionprops_table_all_frames, menu="Measurement tables > Regionprops of all frames (nsr)")

def _check_2D_labels_in_3D_images(label_image):
    '''
    Checks if there are 2D labels in a 3D label image
    '''
    from skimage.measure import regionprops as sk_regionprops

    measurements = sk_regionprops(label_image)
    for props in measurements:
        label_shape = np.array(props.bbox[3:]) - np.array(props.bbox[:3])
        if np.any(label_shape == 1):
            return True
    return False

try:
    # morphometrics API
    from morphometrics.measure import register_measurement_set

    # this is just a wrapper with the right image parameter names
    def regionprops_for_morphometrics(intensity_image : "napari.types.ImageData", label_image: "napari.types.LabelsData", size : bool = True, intensity : bool = True, perimeter : bool = False, shape : bool = False, position : bool = False, moments : bool = False) -> "pd.DataFrame":
        return regionprops_table(intensity_image, label_image,
                                 size=size,
                                 intensity=intensity,
                                 shape=shape,
                                 perimeter=perimeter,
                                 position=position,
                                 moments=moments)

    register_measurement_set(
        regionprops_for_morphometrics,
        name="regionprops (nsr)",
        choices=["intensity", "size", "shape", "perimeter", "position", "moments"],
        uses_intensity_image=True,
    )

    from napari_tools_menu import register_dock_widget
    from morphometrics._gui._qt.measurement_widgets import QtMeasurementWidget
    register_dock_widget(
        QtMeasurementWidget,
        "Measurement tables > Region properties (morphometrics)"
    )
except:
    pass




