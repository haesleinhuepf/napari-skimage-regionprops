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
        reference_label_image: napari.types.LabelsData,
        label_images_to_measure: List[napari.types.LabelsData],
        intensity_images: List[napari.types.ImageData],
        intersection_area_over_object_area: float = 0.5,
        return_summary_statistics: bool = True,
        size: bool = True,
        intensity: bool = True,
        perimeter: bool = False,
        shape: bool = False,
        position: bool = False,
        moments: bool = False,
        napari_viewer: Viewer = None) -> List["pandas.DataFrame"]:
    """
    Add a table widget to a napari viewer with mapped summary statistics.

    Adds a table widget to a given napari viewer with summary statistics that
    relates objects of one reference label image to objects in other label
    images. Matched intensity images should be given for intensity
    measurements. If no intensity images are given, calculates only properties
    not related to intensities. If a single label image is given, it executes
    regular 'regionprops_table' function.
    """

    suffixes = []
    
    if napari_viewer is not None:
        for layer in napari_viewer.layers:
            if type(layer) is napari.layers.Labels:
                if np.array_equal(layer.data, reference_label_image):
                    reference_labels_layer = layer
                    reference_suffix = '_' + layer.name
                else:
                    for labels in label_images_to_measure:
                        if np.array_equal(layer.data, labels):
                            suffixes.append('_' + layer.name)
        suffixes.insert(0, reference_suffix)
    print('suffixes = ', suffixes)
    
    ## Single image measurements
    if len(label_images_to_measure) == 0:
        ### Without intensity measurements
        if len(intensity_images) == 0:
            table = measure_labels(
                reference_labels=reference_label_image, 
                size=size, perimeter=perimeter,
                shape=shape, position=position,
                moments=moments)
        ### Or with intensity measurements
        else:
            reference_intensity_image = intensity_images[0]
            table = measure_labels_with_intensity(
                reference_labels=reference_label_image,
                intensity_image=reference_intensity_image,
                size=size, perimeter=perimeter,
                shape=shape, position=position,
                moments=moments)
    # More than one label image measurements
    else:
        ### Without intensity measurements
        if len(intensity_images) == 0:
            table = measure_labels_in_labels(reference_labels=reference_label_image,
                                          labels_to_measure=label_images_to_measure,
                                          size=size, perimeter=perimeter,
                                          shape=shape, position=position,
                                          moments=moments,
                                          intersection_area_over_object_area=intersection_area_over_object_area,
                                          suffixes=suffixes)
        else:
            reference_intensity_image = intensity_images[0]
            if len(intensity_images)==1:
                intensity_image_of_labels_to_measure = None
            else:
                intensity_image_of_labels_to_measure = intensity_images[1:]
            table = measure_labels_in_labels_with_intensity(
                reference_labels=reference_label_image,
                labels_to_measure=label_images_to_measure,
                intensity_image_of_reference=reference_intensity_image,
                intensity_image_of_labels_to_measure=intensity_image_of_labels_to_measure,
                size=size, perimeter=perimeter,
                shape=shape, position=position,
                moments=moments,
                intersection_area_over_object_area=intersection_area_over_object_area,
                suffixes=suffixes)
    
    if napari_viewer is not None:
        # turn table into a widget
        from ._table import add_table
        # Clear labels layer properties (avoid appending when re-running)
        reference_labels_layer.properties = {}
        # Append table to properties of reference layer
        reference_labels_layer.properties = table
        # Display table (which also adds it to features)
        add_table(reference_labels_layer, napari_viewer)
    else:
        return table

def link_two_label_images(reference_labels : napari.types.LabelsData,
                labels_to_measure : napari.types.LabelsData, 
                intersection_area_over_object_area: float = 0.5
                ) -> "pandas.DataFrame":
    import numpy as np
    import pandas as pd
    from skimage.measure import regionprops_table as sk_regionprops_table
    
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
    
    # Create table (label_links) that links labels from scanning channel
    # to reference channel
    table_linking_labels = pd.DataFrame(
        sk_regionprops_table(label_image=labels_to_measure,
                             intensity_image=reference_labels,
                             properties=['label', ],
                             extra_properties=[highest_overlap]
                             )
    ).astype(int)
    # rename column
    table_linking_labels.rename(columns={'highest_overlap': 'label_reference'},
                       inplace=True)
    table_linking_labels = table_linking_labels[['label_reference', 'label']]
    
    # Add eventual missing reference labels to table (they belong to background)
    bg_labels_list = []
    for i in np.unique(reference_labels)[1:].tolist():
        if i not in table_linking_labels['label_reference'].values:
            bg_labels_list.append([i, 0])
    bg_labels = pd.DataFrame(bg_labels_list,
                             columns=['label_reference', 'label'])
    table_linking_labels = pd.concat([table_linking_labels, bg_labels], axis=0) \
        .sort_values(by=['label_reference', 'label']).reset_index(drop=True)

    return table_linking_labels

def merge_measurements_to_reference(
    table_reference_labels_properties : "pandas.DataFrame",
    table_linking_labels : List["pandas.DataFrame"],
    table_labels_to_measure_properties : List["pandas.DataFrame"],
    suffixes=None) -> "pandas.DataFrame":
    import pandas as pd
    ## Shape input to right format
    ### Create lists of tables to iterate later
    if not isinstance(table_linking_labels, list):
        list_table_linking_labels = [table_linking_labels]
    else:
        list_table_linking_labels = table_linking_labels
    if not isinstance(table_labels_to_measure_properties, list):
        list_table_labels_to_measure_properties = [table_labels_to_measure_properties]
    else:
        list_table_labels_to_measure_properties = table_labels_to_measure_properties
    ### Build custom suffixes or check if provided suffixes match data size
    n_measurement_tables = len(list_table_labels_to_measure_properties)
    if suffixes is None:
        n_leading_zeros = n_measurement_tables // 10
        suffixes = ['_reference'] + ['_' + str(i+1).zfill(1+n_leading_zeros) for i in range(n_measurement_tables)]
    else:
        if len(suffixes) != len(table_labels_to_measure_properties) + 1:
            print('Error: List of suffixes must have the same length as the number of tables containing measurements')
            return
    
    ## Rename column names with appropriate suffixes
    ### Raname reference table columns
    table_reference_labels_properties.columns = [
            props + suffixes[0]
            for props in table_reference_labels_properties.columns]
    ### Rename columns of tables with linking labels 
    for i, table_linking_labels in enumerate(list_table_linking_labels):
        table_linking_labels.rename(
                columns={'label_reference': 'label' + suffixes[0],
                         'label': 'label' + suffixes[i+1]},
                inplace=True)
    ### Rename columns of tables with properties from other channels
    for i, table_labels_to_measure_properties in enumerate(list_table_labels_to_measure_properties):
        table_labels_to_measure_properties.columns = [
            props + suffixes[i+1]
            for props in table_labels_to_measure_properties.columns]
    
    ## output_table starts with reference labels and their properties
    output_table = table_reference_labels_properties
    ## Consecutively merge linking_labels tables and properties from other channels tables to the reference table
    for i, table_linking_labels, table_labels_to_measure_properties in zip(range(n_measurement_tables), list_table_linking_labels, list_table_labels_to_measure_properties):

        # Merge other labels to output table based on label_reference
        output_table = pd.merge(output_table,
                                table_linking_labels,
                                how='outer', on='label' + suffixes[0])
        # Fill NaN labels with zeros (if label were not linked, they belong to background)
        output_table['label' + suffixes[i+1]] = output_table['label' + suffixes[i+1]].fillna(0)
        # Merge other properties to output table based on new labels column
        output_table = pd.merge(output_table,
                                table_labels_to_measure_properties,
                                how='outer', on='label' + suffixes[i+1])
    return output_table

def make_summary_table(table: "pandas.DataFrame",
                      suffixes) -> "pandas.DataFrame":
    # If not provided, guess suffixes from column names (last string after '_')
    import re
    import pandas as pd
    if suffixes is None:
        try:
            suffixes = []
            for name in table.columns:
                new_entry = re.findall(r'_[^_]+$', name)[0]
                if new_entry not in suffixes:
                    suffixes.append(new_entry)
        except:
            print('Could not infer suffixes from column names. Pleas provide a list of suffixes identifying different channels')
            return
    
    grouped = table.groupby('label' + suffixes[0])
    probe_columns = [prop for prop in table.columns
                     if not prop.endswith(suffixes[0])]
    probe_measurement_columns = [name for name in probe_columns
                     if not name.startswith('label')]
    table = grouped[probe_measurement_columns].describe().reset_index()
    return table

def measure_labels(reference_labels : napari.types.LabelsData, 
                   size : bool = True, perimeter : bool = False,
                   shape : bool = False, position : bool = False,
                   moments : bool = False,
                   napari_viewer : Viewer = None) -> "pandas.DataFrame":
    table = regionprops_table(image = np.zeros_like(reference_labels), 
                              labels = reference_labels, size = size,
                              intensity = False, perimeter = perimeter,
                              shape = shape, position = position,
                              moments = moments, napari_viewer = napari_viewer)
    return table

def measure_labels_with_intensity(reference_labels : napari.types.LabelsData,
                                intensity_image : napari.types.ImageData,
                                size : bool = True, perimeter : bool = False,
                                shape : bool = False, position : bool = False,
                                moments : bool = False,
                                napari_viewer : Viewer = None) -> "pandas.DataFrame":
    table = regionprops_table(image = intensity_image, 
                              labels = reference_labels, size = size,
                              intensity = True, perimeter = perimeter,
                              shape = shape, position = position,
                              moments = moments, napari_viewer = napari_viewer)
    return table

def measure_labels_in_labels(reference_labels : napari.types.LabelsData,
                              labels_to_measure : List[napari.types.LabelsData],
                              size : bool = True, perimeter : bool = False,
                              shape : bool = False, position : bool = False,
                              moments : bool = False,
                              intersection_area_over_object_area: float = 0.5,
                              suffixes : List[str] = None,
                              napari_viewer : Viewer = None) -> "pandas.DataFrame":
    ## Get reference properties
    reference_labels_properties = measure_labels(
        reference_labels = reference_labels, size = size,
        perimeter = perimeter, shape = shape, position = position,
        moments = moments, napari_viewer = napari_viewer
        )
    list_table_linking_labels = []
    list_table_labels_to_measure_properties = []
    # Make labels_to_measure iterable
    if not isinstance(labels_to_measure, list):
        labels_to_measure = [labels_to_measure]
    # Link each labels_to_measure image to reference and get their properties
    for label_image in labels_to_measure:
        table_linking_labels = link_two_label_images(
            reference_labels=reference_labels,
            labels_to_measure=label_image,
            intersection_area_over_object_area=intersection_area_over_object_area
            )
        list_table_linking_labels.append(table_linking_labels)
            
        labels_to_measure_properties = measure_labels(
            reference_labels = label_image, size = size,
            perimeter = perimeter, shape = shape, position = position,
            moments = moments, napari_viewer = napari_viewer
            )
        list_table_labels_to_measure_properties.append(labels_to_measure_properties)
    
    # Merge tables
    table = merge_measurements_to_reference(
        table_reference_labels_properties=reference_labels_properties,
        table_linking_labels=list_table_linking_labels,
        table_labels_to_measure_properties=list_table_labels_to_measure_properties,
        suffixes=suffixes)
    return table

def measure_labels_in_labels_with_intensity(
        reference_labels : napari.types.LabelsData,
        labels_to_measure : List[napari.types.LabelsData],
        intensity_image_of_reference : napari.types.ImageData,
        intensity_image_of_labels_to_measure : List[napari.types.ImageData] = None,
        size : bool = True, perimeter : bool = False,
        shape : bool = False, position : bool = False,
        moments : bool = False,
        intersection_area_over_object_area: float = 0.5,
        suffixes : List[str] = None,
        napari_viewer : Viewer = None) -> "pandas.DataFrame":
    ## Get reference properties
    reference_labels_properties = measure_labels_with_intensity(
        reference_labels = reference_labels,
        intensity_image = intensity_image_of_reference,
        size = size,
        perimeter = perimeter, shape = shape, position = position,
        moments = moments, napari_viewer = napari_viewer
        )
    list_table_linking_labels = []
    list_table_labels_to_measure_properties = []
    # Make labels_to_measure iterable
    if not isinstance(labels_to_measure, list):
        labels_to_measure = [labels_to_measure]
    ## If no intensity_image_of_labels_to_measure provided, use reference intentity
    if intensity_image_of_labels_to_measure is None:
        intensity_image_of_labels_to_measure = [intensity_image_of_reference]*len(labels_to_measure)
    ## If intensity_image_of_labels_to_measure provided, check if sizes match
    else:
        # Make intensity_image_of_labels_to_measure iterable
        if not isinstance(intensity_image_of_labels_to_measure, list):
            intensity_image_of_labels_to_measure = [intensity_image_of_labels_to_measure]
        if len(intensity_image_of_labels_to_measure) != len(labels_to_measure):
            print('Error! Length of intensity_image_of_labels_to_measure and labels_to_measure must match.')
            return
    
    # Link each labels_to_measure image to reference and get their properties
    for label_image, intensity_image in zip(labels_to_measure, intensity_image_of_labels_to_measure):
        table_linking_labels = link_two_label_images(
            reference_labels=reference_labels,
            labels_to_measure=label_image,
            intersection_area_over_object_area=intersection_area_over_object_area
            )
        list_table_linking_labels.append(table_linking_labels)
            
        labels_to_measure_properties = measure_labels_with_intensity(
            reference_labels = label_image, 
            intensity_image = intensity_image,
            size = size,
            perimeter = perimeter, shape = shape, position = position,
            moments = moments, napari_viewer = napari_viewer
            )
        list_table_labels_to_measure_properties.append(labels_to_measure_properties)
    
    # Merge tables
    table = merge_measurements_to_reference(
        table_reference_labels_properties=reference_labels_properties,
        table_linking_labels=list_table_linking_labels,
        table_labels_to_measure_properties=list_table_labels_to_measure_properties,
        suffixes=suffixes)
    return table

regionprops_map_channels_table_all_frames = analyze_all_frames(
    napari_regionprops_map_channels_table)
register_function(
    regionprops_map_channels_table_all_frames,
    menu="Measurement > Regionprops map multichannel of all frames (nsr)")
