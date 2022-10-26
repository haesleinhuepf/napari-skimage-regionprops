import warnings

import numpy as np
import pandas
from napari import Viewer
from napari_tools_menu import register_function
import napari
from typing import List
import math
from ._all_frames import analyze_all_frames
import sys

def regionprops(image_layer : napari.layers.Layer, labels_layer: napari.layers.Labels, size : bool = True, intensity : bool = True, perimeter : bool = False, shape : bool = False, position : bool = False, moments : bool = False, napari_viewer : Viewer = None):
    warnings.warn("napari_skimage_regionprops.regionprops is deprecated. Use regionprops_table instead.")
    image_data = None
    if image_layer is not None:
        image_data = image_layer.data

    regionprops_table(image_data, labels_layer.data, size, intensity, perimeter, shape, position, moments, napari_viewer)

@register_function(menu="Measurement > Regionprops (scikit-image, nsr)")
def regionprops_table(image : napari.types.ImageData, labels: napari.types.LabelsData, size : bool = True, intensity : bool = True, perimeter : bool = False, shape : bool = False, position : bool = False, moments : bool = False, napari_viewer : Viewer = None) -> "pandas.DataFrame":
    """
    Adds a table widget to a given napari viewer with quantitative analysis results derived from an image-label pair.
    """
    # Check if image was provided or just labels
    if image is not None:
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
        properties = properties + ['area', 'bbox_area', 'convex_area', 'equivalent_diameter']

    if intensity:
        if image is None:
            warnings.warn("No intensity image was provided, skipping intensity measurements.")
        else:
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
        properties = properties + ['solidity', 'extent', 'feret_diameter_max', 'local_centroid']
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
        if image is None:
            properties = properties + ['centroid', 'bbox']
        else:
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
    if image is not None:
        image = np.asarray(image)
    table = sk_regionprops_table(np.asarray(labels).astype(int), intensity_image=image,
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
    return tuple([math.sqrt(20.0 * e) for e in eigvals])

regionprops_table_all_frames = analyze_all_frames(regionprops_table)
register_function(regionprops_table_all_frames, menu="Measurement > Regionprops of all frames (nsr)")


def make_element_wise_dict(list_of_keys, list_of_values):
    """Return an element-wise dictionary of two lists.

    From a list of keys and a list of values, return a dictionnary linking
    those lists. Both lists must have the same length.

    Parameters
    ----------
    list_of_keys : List
        list to be used as dictionnary keys.
    list_of_values : List
        list to be used as dictionnary values.

    Returns
    -------
    Dictionnary
    """
    return dict(
                list(
                    map(
                        lambda list_of_keys, list_of_values:
                        [list_of_keys, list_of_values],
                        *[list_of_keys, list_of_values]
                        )
                    )
                )


@register_function(
    menu="Measurement > Regionprops map multichannel (scikit-image, nsr)")
def napari_regionprops_map_channels_table(
        label_images: List[napari.types.LabelsData],
        intensity_images: List[napari.types.ImageData],
        reference_label_image: napari.types.LabelsData,
        intersection_area_over_object_area: float = 0.5,
        return_summary_statistics: bool = True,
        size: bool = True,
        intensity: bool = True,
        perimeter: bool = False,
        shape: bool = False,
        position: bool = False,
        moments: bool = False,
        napari_viewer: Viewer = None) -> "pandas.DataFrame":
    """
    Add a table widget to a napari viewer with mapped summary statistics.

    Adds a table widget to a given napari viewer with summary statistics that
    relates objects of one reference label image to objects in other label
    images. Matched intensity images should be given for intensity
    measurements. If no intensity images are given, calculates only properties
    not related to intensities. If a single label image is given, it executes
    regular 'regionprops_table' function.
    """

    if napari_viewer is not None:
        # store list of labels layers for saving results later
        labels_layer_list = [None]*len(label_images)
        for layer in napari_viewer.layers:
            if type(layer) is napari.layers.Labels:
                # Store in the same order as labels_list
                for i, labels in enumerate(label_images):
                    if np.array_equal(layer.data, labels):
                        labels_layer_list[i] = layer
                    if np.array_equal(reference_label_image, labels):
                        ref_channel = i
    # If single label image is provided, indicate to do single channel regionprops
    if len(label_images) == 1:
        ref_channel = None
        label_images = label_images[0]
    # If no intensity image provided, indicate no intensity measurements
    if len(intensity_images) == 0:
        intensity_images = None
    else:
        intensity_images = np.asarray(intensity_images)

    table_list = regionprops_map_channels_table(
        labels_array=np.asarray(label_images),
        intensity_image=intensity_images,
        ref_channel=ref_channel,
        intersection_area_over_object_area=intersection_area_over_object_area,
        summary=return_summary_statistics,
        size=size,
        intensity=intensity,
        perimeter=perimeter,
        shape=shape,
        position=position,
        moments=moments)

    if napari_viewer is not None:
        # turn table into a widget
        from ._table import add_table
        # Store results in the properties dictionary:
        for labels_layer, table in zip(labels_layer_list, table_list):
            # Flatten summary statistics table
            table.columns = [' '.join(col).strip()
                             for col in table.columns.values]
            labels_layer.properties = table
            add_table(labels_layer, napari_viewer)
    else:
        return table_list


def regionprops_map_channels_table(labels_array, intensity_image=None,
                                   ref_channel=None,
                                   intersection_area_over_object_area=0.5,
                                   summary=True, **kwargs):
    """
    Measure properties from 2 (or more) channels and return summary statistics.

    For each channel in a multi-channel image and a multi-channel label array,
    it measures properties of a reference channel (default: 0) and a probe
    channel.
    For each object in the reference channel, it returns descriptive statistics
    of overlapping objects in the probe channel(s).
    Objects are considered to overlap when object in probe channel has more
    than 'intersection_area_over_object_area' of its area inside object in 
    ref channel (0 = No overlap, 1 = Full overlap, default = 0.5).
    If single labeled image is given, regular 'regionprops_table' function is
    executed.

    Parameters
    ----------
    labels_array : (C, M, N[,P]) ndarray
        Multichannel labels array. Channel must be the first dimention.
    intensity_image : (C, M, N[,P]) ndarray, optional
        Multichannel image. Channel must be the first dimention.
    ref_channel : int, optional
        Reference channel number. Default is 0.
    intersection_area_over_object_area : float, optional
        Ratio of area that a probe channel object must overlap with a reference
        channel object. It can be understood as area of intersection divided by
        area of probe object. Ranges from 0 to 1. Default is 0.5.
    summary: bool, optional
        Determine whether to return summary statistics or direct relatioships.
        Default is True.

    Returns
    -------
    table_list : List
        List of tables where each table relates one probe channel objects to
        the reference channel objects. If `summary` is True, for each property,
        returns count, mean, standard deviation, median, min, max, 25th
        quartile and 75th quartile. If `summary` is False, direct relationships
        are returned (which may lead to columns with repeated labels).
    """
    # quantitative analysis using scikit-image's regionprops
    from skimage.measure import regionprops_table as sk_regionprops_table
    import numpy as np
    import pandas as pd
    # To DO: check if input shape is correct
    table_list = []
    # Single channel
    if ref_channel is None:
        table_list += [regionprops_table(image = intensity_image,
                                        labels = labels_array,
                                        **kwargs)]
        return table_list
    # Channel axis is expected to be 0
    n_channels = labels_array.shape[0]
    
    large_numbers = False

    def highest_overlap(regionmask, intensity_image,
                        overlap_threshold=intersection_area_over_object_area):
        """
        Gets the label number with highest overlap with label in another image.
        
        This function masks a labeled image called 'intensity_image' with 
        'regionmask' and calculates the frequency of pixel values in that
        region. Disconsidering zeros (background), it returns the most frequent
        value if it overcomes the 'overlap_threshold'. Otherwise, it returns 0.
        In case of draws, it returns the first occurence. This function follows
        the standards of skimage.regionprops 'extra_properties'.
         
        Parameters
        ----------
        regionmask : (M, N[,P]) ndarray
            Label image (probe channel). Labels to be used as a mask.
        intensity_image : (M, N[,P]) ndarray
            Label image (reference channel). Labels to be measured using probe
            channel as a mask.
        
        Returns
        -------
        value : int
            Most frequent label number under regionmask, except 0, that
            overcomes threshold. Otherwise, it returns 0.
        """
        if overlap_threshold == 0:
            return 0
        values, counts = np.unique(np.sort(intensity_image[regionmask]),
                                   return_counts=True)
        # Probabilities of belonging to a certain label or bg
        probs = counts/np.sum(counts)
        # If there is underlying bg, take it out
        if values[0] == 0:
            values = np.delete(values, 0)
            probs = np.delete(probs, 0)

        # if any label overlap probability is bigger than overlap_threshold
        if (probs >= overlap_threshold).any():
            # find label with highest overlap
            # if equal frequency, return first occurence
            index_max_overlap_prob = np.argmax(probs)
            value = values[index_max_overlap_prob]
        else:  # otherwise, highest allowed overlap is considered to be
            # with background, i.e., object does not "belong" to any
            #  other label and gets 0
            value = 0
        return value

    # Measure properties of reference channel
    if intensity_image is not None:
        # If single intensity image was given, then use it
        if intensity_image.shape[0]==1:
            image = intensity_image[0]
        # Otherwise use the corresponding intensity image
        else:
            image = intensity_image[ref_channel]
    else:
        image = None
    ref_channel_props = regionprops_table(image=image,
                                          labels=labels_array[ref_channel],
                                          **kwargs)
    for i in range(n_channels):
        if i != ref_channel:
            # Create table (label_links) that links labels from probe channel
            # to reference channel
            label_links = pd.DataFrame(
                sk_regionprops_table(label_image=labels_array[i],
                                     intensity_image=labels_array[ref_channel],
                                     properties=['label', ],
                                     extra_properties=[highest_overlap]
                                     )
            ).astype(int)
            # rename column
            label_links.rename(columns={'label': 'label-ch' + str(i),
                                        'highest_overlap':
                                            'label-ch' + str(ref_channel)},
                               inplace=True)

            # Include extra properties of reference channel
            properties_with_extras = [props for props in ref_channel_props
                                      if props != 'label']
            # Append properties of reference channel to table
            for props in properties_with_extras:
                large_numbers = False
                # If large numbers, store format and convert to string to allow
                # pandas replacement below, restore data type afterwards
                if abs(ref_channel_props[props]).values.max() > np.iinfo(np.int32).max:
                    large_numbers = True
                    props_dtype = ref_channel_props[props].dtype
                    ref_channel_props[props] = \
                        ref_channel_props[props].astype(str)

                props_mapping = make_element_wise_dict(
                    ref_channel_props['label'].tolist(),
                    ref_channel_props[props].tolist())

                label_links[props + '-ch' + str(ref_channel)] = \
                    label_links['label-ch' + str(ref_channel)]

                label_links = label_links.replace(
                    {props + '-ch' + str(ref_channel): props_mapping}
                    )

                if large_numbers:
                    label_links[props + '-ch' + str(ref_channel)] = \
                        label_links[props + '-ch' + str(ref_channel)]\
                        .astype(props_dtype)

            col_names = label_links.columns.to_list()
            # Append properties of probe channel to table
            if intensity_image is not None:
                # If single intensity image was given, then use it
                if intensity_image.shape[0]==1:
                    image = intensity_image[0]
                # Otherwise use the corresponding intensity image
                else:
                    image = intensity_image[i]
            probe_channel_props = pd.DataFrame(
                regionprops_table(image=image,
                                  labels=labels_array[i],
                                  **kwargs)
            )
            # rename column
            probe_channel_props.rename(
                columns=dict([(props, props + '-ch' + str(i))
                              for props in properties_with_extras]),
                inplace=True)
            probe_channel_props.drop(columns='label', inplace=True)
            table = pd.concat([label_links, probe_channel_props], axis=1)

            # Insert new column names (from probe channel)
            probe_column_names = probe_channel_props.columns.to_list()
            col_names = col_names[1:] + [col_names[0]] + probe_column_names
            # Re-order columns
            table = table[col_names]

            if summary:
                grouped = table.groupby('label-ch' + str(ref_channel))
                table = grouped[probe_column_names].describe().reset_index()
            table_list += [table]

    return table_list


regionprops_map_channels_table_all_frames = analyze_all_frames(
    napari_regionprops_map_channels_table)
register_function(
    regionprops_map_channels_table_all_frames,
    menu="Measurement > Regionprops map multichannel of all frames (nsr)")
