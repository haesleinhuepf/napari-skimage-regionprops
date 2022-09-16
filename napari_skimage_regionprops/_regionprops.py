import warnings

import numpy as np
import pandas
from napari import Viewer
from napari_tools_menu import register_function
import napari
from typing import List
import math
from ._all_frames import analyze_all_frames

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
    return tuple([math.sqrt(20.0 * e) for e in eigvals])

regionprops_table_all_frames = analyze_all_frames(regionprops_table)
register_function(regionprops_table_all_frames, menu="Measurement > Regionprops of all frames (nsr)")

def make_element_wise_dict(a,b):
    '''Returns an element-wise dictionary of keys from a and values from b'''
    return dict(list(map(lambda a, b: [a,b], *[a, b])))

@register_function(menu="Measurement > Regionprops map multichannel (scikit-image, nsr)")
def regionprops_map_channels_table(image_list : List[napari.types.ImageData], 
                             labels_list: List[napari.types.LabelsData], 
                             ref_channel: int = 0,
                             overlap: float = 0.5, 
                             summary: bool = True,
                             size : bool = True, 
                             intensity : bool = True, 
                             perimeter : bool = False, 
                             shape : bool = False, 
                             position : bool = False,
                             moments : bool = False, 
                             napari_viewer : Viewer = None) -> "pandas.DataFrame":
    
    current_dim_value = 0
    if napari_viewer is not None:
        current_dim_value = napari_viewer.dims.current_step[0]

        # store list of labels layers for saving results later
        labels_layer_list = []
        for layer in napari_viewer.layers:
            print(layer, type(layer), layer.data)
            if type(layer) is 'napari.layers.labels.labels.Labels':
                labels_layer_list += [layer]
        print(labels_layer_list)
        # to do: deal with 4D data
        # for image, labels in image_list, labels_list:
        #     # deal with 4D data
        #     if len(image.shape) == 4:
        #         image = image[current_dim_value]
        #     if len(labels.shape) == 4:
        #         labels = labels[current_dim_value]
    
    table_list = regionprops_map_channels(labels_array = np.asarray(labels_list),
                                          intensity_image = np.asarray(image_list),
                                          ref_channel = ref_channel,
                                          overlap = overlap, summary = summary,
                                          size = size,
                                          intensity = intensity,
                                          perimeter = perimeter, shape = shape,
                                          position = position,
                                          moments = moments)
    
    if napari_viewer is not None:
        # turn table into a widget
        from ._table import add_table
        # Store results in the properties dictionary:
        for labels_layer, table in zip(labels_layer_list, table_list):
            # Flatten summary statistics table
            table.columns = [' '.join(col).strip() for col in table.columns.values]
            labels_layer.properties = table
            add_table(labels_layer, napari_viewer)
    else:
        return table_list
    
    


def regionprops_map_channels(labels_array, intensity_image, ref_channel = 0, overlap = 0.5, summary = True, **kwargs):
    '''Measure properties from 2 channels in and return summary statistics.
    
    For each channel in a multi-channel image and a multi-channel label array, 
    it measures properties of a reference channel (default: 0) and a probe channel. 
    For each object in the reference channel, it returns descriptive statistics of overlapping objects in the probe channel(s). 
    Objects are considered to overlap when object in probe channel has more than 'overlap' of its area inside object in ref channel
    (0 = No overlap, 1 = Full overlap, default = 0.5).
    '''
    # quantitative analysis using scikit-image's regionprops
    from skimage.measure import regionprops_table as sk_regionprops_table
    import numpy as np
    import pandas as pd
    # To DO: check if input shape is correct
    n_channels = intensity_image.shape[0]
    table_list = []
    
    def highest_overlap(regionmask, intensity_image, overlap_threshold = overlap):

        if overlap_threshold == 0:
            return 0
        values, counts = np.unique(np.sort(intensity_image[regionmask]), return_counts = True)
        # Probabilities of belonging to a certain label or bg
        probs = counts/np.sum(counts)
        # If there is underlying bg, take it out
        if values[0] == 0:
            values = np.delete(values, 0)
            probs = np.delete(probs, 0)
        
        # if any label overlap probability is bigger than overlap_threshold
        if (probs >= overlap_threshold).any():
            # find label with highest overlap
            index_max_overlap_prob = np.argmax(probs) # if equal frequency, return first occurence
            value = values[index_max_overlap_prob]
        else: # otherwise, highest allowed overlap is considered to be with background, 
            # i.e., object does not "belong" to any other label and gets 0
            value = 0
        return value
    
    # Measure properties of reference channel
    ref_channel_props =  regionprops_table(image = intensity_image[ref_channel],
                                           labels = labels_array[ref_channel], 
                                           **kwargs)
    for i in range(n_channels):
        if i != ref_channel:
            # Create table (label_links) that links labels from probe channel to reference channel
            label_links= pd.DataFrame(
                sk_regionprops_table(label_image = labels_array[i], 
                                          intensity_image = labels_array[ref_channel], 
                                          properties = ['label',],
                                          extra_properties = [highest_overlap]
                                         )
            ).astype(int)
            # rename column
            label_links.rename(columns={'label':'label-ch' + str(i), 'highest_overlap':'label-ch' + str(ref_channel)}, inplace=True)
             
            # Include extra properties of reference channel
            properties_with_extras = [props for props in ref_channel_props if props != 'label']
            # Append properties of reference channel to table
            for props in properties_with_extras:
                props_mapping = make_element_wise_dict(ref_channel_props['label'].tolist(), ref_channel_props[props].tolist())
                # label_links.insert(1, props + '-of-obj-at-ref-ch' + str(ref_channel), label_links['label-of-obj-at-ref-ch' + str(ref_channel)])
                label_links[props + '-ch' + str(ref_channel)] = label_links['label-ch' + str(ref_channel)]
                label_links = label_links.replace({props + '-ch' + str(ref_channel) : props_mapping})

            col_names = label_links.columns.to_list()
            # Append properties of probe channel to table
            probe_channel_props = pd.DataFrame(
                regionprops_table(image = intensity_image[i,],
                                  labels = labels_array[i], 
                                  **kwargs)
            )
            # rename column
            probe_channel_props.rename(columns=dict([(props , props + '-ch' + str(i)) for props in properties_with_extras]), inplace=True)
            probe_channel_props.drop(columns = 'label', inplace=True)
            table = pd.concat([label_links, probe_channel_props], axis=1)
            
            # Insert new column names (from probe channel)
            probe_column_names = probe_channel_props.columns.to_list()
            # col_names[1:1] = probe_column_names # this insert columns in the beggining after the first
            col_names = col_names[1:] + [col_names[0]] + probe_column_names
            # Re-order columns
            table = table[col_names]
            
            if summary:
                grouped = table.groupby('label-ch' + str(ref_channel))
                table = grouped[probe_column_names].describe().reset_index()
            
            table_list += [table]
        
    return table_list