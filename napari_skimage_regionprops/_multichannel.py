import napari
import pandas
import numpy as np
from napari import Viewer
from typing import List
from magicgui import magic_factory
from napari_tools_menu import register_dock_widget
from ._regionprops import regionprops_table
from ._process_tables import merge_measurements_to_reference
from ._process_tables import make_summary_table


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
    return {k: v for k, v in zip(list_of_keys, list_of_values)}


def connect_events(widget):
    def toggle_intensity_widgets(event):
        widget.intensity_image_reference.visible = event
        # If multichannel is True when anebling intensity, then enable 
        # intensity image inputs
        if (event == True) & (widget.multichannel.value == True):
            widget.intensity_images_to_measure.visible = True
        else:
            widget.intensity_images_to_measure.visible = False
            
    def toggle_multichannel_widgets(event):
        widget.label_images_to_measure.visible = event
        widget.intersection_over_reference_area.visible = event
        widget.select_summary_statistics.visible = event
        if (event == True) & (widget.intensity.value == True):
            widget.intensity_images_to_measure.visible = True
        else:
            widget.intensity_images_to_measure.visible = False   
  
    def toggle_summary_statistics_widgets(event):
        widget.counts.visible = event
        widget.mean.visible = event
        widget.std.visible = event
        widget.minimum.visible = event
        widget.percentile_25.visible = event
        widget.median.visible = event
        widget.percentile_75.visible = event
        widget.maximum.visible = event
    
    widget.intensity.changed.connect(toggle_intensity_widgets)
    widget.multichannel.changed.connect(toggle_multichannel_widgets)
    widget.select_summary_statistics.changed.connect(toggle_summary_statistics_widgets)
    
    # Intial visibility states
    widget.intensity_image_reference.visible = False
    widget.intensity_images_to_measure.visible = False
    widget.label_images_to_measure.visible = False
    widget.select_summary_statistics.visible = False
    widget.intersection_over_reference_area.visible = False
    widget.counts.visible = False
    widget.mean.visible = False
    widget.std.visible = False
    widget.minimum.visible = False
    widget.percentile_25.visible = False
    widget.median.visible = False
    widget.percentile_75.visible = False
    widget.maximum.visible = False

widgets_layout_settings = {
    'label_image_reference': {
        'label': 'Label Image Reference'
    },
    'intensity_image_reference': {
        'label': 'Intensity Image Reference'
    },
    'label_images_to_measure': {
        'label': 'Label Image(s) to Measure'
    },
    'intensity_images_to_measure': {
        'label': 'Intensity Image(s) to Measure'
    },
    'intersection_over_reference_area': {
        'widget_type': 'FloatSlider',
        'min': 0,
        'max': 1,
        'step': 0.1,
        'tooltip': 'A ratio that determines if an object in one channel belongs to another in a reference channel.\nIt goes from 0 (target object always belongs to background) to 1 (target object belongs to reference object only if completely inside reference)'
    },
    'std': {
        'label': 'standard deviation (std)'
    },
    'percentile_25': {
        'label': '25% percentile'
    },
    'percentile_75': {
        'label': '75% percentile'
    }
}

@register_dock_widget(
    menu="Measurement > Regionprops map multichannel (scikit-image, nsr)")
# Need magic factory to make hidding and showing functionality available
@magic_factory(widget_init=connect_events,
               layout = 'vertical',
               label_image_reference=widgets_layout_settings['label_image_reference'],
               intensity_image_reference=widgets_layout_settings['intensity_image_reference'],
               label_images_to_measure=widgets_layout_settings['label_images_to_measure'],
               intensity_images_to_measure=widgets_layout_settings['intensity_images_to_measure'],
               intersection_over_reference_area=widgets_layout_settings['intersection_over_reference_area'],
               std=widgets_layout_settings['std'],
               percentile_25=widgets_layout_settings['percentile_25'],
               percentile_75=widgets_layout_settings['percentile_75'])
def napari_regionprops_map_channels_table(
        label_image_reference: napari.types.LabelsData,
        intensity_image_reference: napari.types.ImageData,
        label_images_to_measure: List[napari.types.LabelsData],
        intensity_images_to_measure: List[napari.types.ImageData],
        intensity: bool = False,
        multichannel: bool = False,
        size: bool = True,
        perimeter: bool = False,
        shape: bool = False,
        position: bool = False,
        moments: bool = False,
        intersection_over_reference_area: float = 0.5,
        select_summary_statistics: bool = False,
        counts: bool = True,
        mean: bool = False,
        std: bool = False,
        minimum: bool = False,
        percentile_25: bool = False,
        median: bool = False,
        percentile_75: bool = False,
        maximum: bool = False,
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
    statistics_list = []
    if counts:
        statistics_list += ['count']
    if mean:
        statistics_list += ['mean']
    if std:
        statistics_list += ['std']
    if minimum:
        statistics_list += ['min']
    if percentile_25:
        statistics_list += ['25%']
    if median:
        statistics_list += ['50%']
    if percentile_75:
        statistics_list += ['75%']
    if maximum:
        statistics_list += ['max']
    suffixes = []
    
    if napari_viewer is not None:
        for layer in napari_viewer.layers:
            if type(layer) is napari.layers.Labels:
                if np.array_equal(layer.data, label_image_reference):
                    reference_labels_layer = layer
                    reference_suffix = '_' + layer.name
                else:
                    for labels in label_images_to_measure:
                        if np.array_equal(layer.data, labels):
                            suffixes.append('_' + layer.name)
        suffixes.insert(0, reference_suffix)
    else:
        if not isinstance(label_images_to_measure, list):
            label_images_to_measure = [label_images_to_measure]
        if not isinstance(intensity_images_to_measure, list):
            intensity_images_to_measure = [intensity_images_to_measure]


    
    ## Single image measurements
    if multichannel == False:
        ### Without intensity measurements
        if intensity == False:
            table = measure_labels(
                label_image_reference=label_image_reference, 
                size=size, perimeter=perimeter,
                shape=shape, position=position,
                moments=moments)
        ### Or with intensity measurements
        else:
            table = measure_labels_with_intensity(
                label_image_reference=label_image_reference,
                intensity_image_reference=intensity_image_reference,
                size=size, perimeter=perimeter,
                shape=shape, position=position,
                moments=moments)
    ## More than one label image measurements
    else:
        if len(suffixes) == 0:               
            n_leading_zeros = len(label_images_to_measure) // 10
            suffixes = ['_reference'] + ['_' + str(i+1).zfill(1+n_leading_zeros)
                                    for i in range(len(label_images_to_measure))]
        ### Without intensity measurements
        if intensity == False:
            table = measure_labels_in_labels(label_image_reference=label_image_reference,
                                          labels_to_measure=label_images_to_measure,
                                          size=size, perimeter=perimeter,
                                          shape=shape, position=position,
                                          moments=moments,
                                          intersection_over_reference_area=intersection_over_reference_area,
                                          suffixes=suffixes)
        ### Or with intensity measurements
        else:
            table = measure_labels_in_labels_with_intensity(
                label_image_reference=label_image_reference,
                labels_to_measure=label_images_to_measure,
                intensity_image_reference=intensity_image_reference,
                intensity_image_of_labels_to_measure=intensity_images_to_measure,
                size=size, perimeter=perimeter,
                shape=shape, position=position,
                moments=moments,
                intersection_over_reference_area=intersection_over_reference_area,
                suffixes=suffixes)
        ### Compute summary statistics instead of individual relationships
        ### This avoids repeated number in the 'label' column
        table = make_summary_table(table, suffixes = suffixes,
                                    statistics_list = statistics_list)
    # Rename first column to just 'label' (to be compatible with other plugins)
    table.rename(columns={"label" + suffixes[0]: "label"}, inplace=True)
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

def link_two_label_images(label_image_reference : napari.types.LabelsData,
                labels_to_measure : napari.types.LabelsData, 
                intersection_over_reference_area: float = 0.5
                ) -> "pandas.DataFrame":
    import numpy as np
    import pandas as pd
    from skimage.measure import regionprops_table as sk_regionprops_table
    
    def highest_overlap(regionmask, label_image_reference,
                        overlap_threshold=intersection_over_reference_area):
        """
        Gets the label number with highest overlap with label in another image.
        
        This function masks a labeled image called 'label_image_reference' with 
        'regionmask' and calculates the frequency of pixel values in that
        region. Disconsidering zeros (background), it returns the most frequent
        value if it overcomes the 'overlap_threshold'. Otherwise, it returns 0.
        In case of draws, it returns the first occurence. This function follows
        the standards of skimage.regionprops 'extra_properties'.
         
        Parameters
        ----------
        regionmask : (M, N[,P]) ndarray
            Label image (probe channel). Labels to be used as a mask.
        label_image_reference : (M, N[,P]) ndarray
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
        values, counts = np.unique(np.sort(label_image_reference[regionmask]),
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
                             intensity_image=label_image_reference,
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
    for i in np.unique(label_image_reference)[1:].tolist():
        if i not in table_linking_labels['label_reference'].values:
            bg_labels_list.append([i, 0])
    bg_labels = pd.DataFrame(bg_labels_list,
                             columns=['label_reference', 'label'])
    table_linking_labels = pd.concat([table_linking_labels, bg_labels], axis=0) \
        .sort_values(by=['label_reference', 'label']).reset_index(drop=True)

    return table_linking_labels

def measure_labels(label_image_reference : napari.types.LabelsData, 
                   size : bool = True, perimeter : bool = False,
                   shape : bool = False, position : bool = False,
                   moments : bool = False,
                   napari_viewer : Viewer = None) -> "pandas.DataFrame":
    table = regionprops_table(image = np.zeros_like(label_image_reference), 
                              labels = label_image_reference, size = size,
                              intensity = False, perimeter = perimeter,
                              shape = shape, position = position,
                              moments = moments, napari_viewer = napari_viewer)
    return table

def measure_labels_with_intensity(label_image_reference : napari.types.LabelsData,
                                intensity_image_reference : napari.types.ImageData,
                                size : bool = True, perimeter : bool = False,
                                shape : bool = False, position : bool = False,
                                moments : bool = False,
                                napari_viewer : Viewer = None) -> "pandas.DataFrame":
    table = regionprops_table(image = intensity_image_reference, 
                              labels = label_image_reference, size = size,
                              intensity = True, perimeter = perimeter,
                              shape = shape, position = position,
                              moments = moments, napari_viewer = napari_viewer)
    return table

def measure_labels_in_labels(label_image_reference : napari.types.LabelsData,
                              labels_to_measure : List[napari.types.LabelsData],
                              size : bool = True, perimeter : bool = False,
                              shape : bool = False, position : bool = False,
                              moments : bool = False,
                              intersection_over_reference_area: float = 0.5,
                              suffixes : List[str] = None,
                              napari_viewer : Viewer = None) -> "pandas.DataFrame":
    ## Get reference properties
    reference_labels_properties = measure_labels(
        label_image_reference = label_image_reference, size = size,
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
            label_image_reference=label_image_reference,
            labels_to_measure=label_image,
            intersection_over_reference_area=intersection_over_reference_area
            )
        list_table_linking_labels.append(table_linking_labels)
            
        labels_to_measure_properties = measure_labels(
            label_image_reference = label_image, size = size,
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
        label_image_reference : napari.types.LabelsData,
        labels_to_measure : List[napari.types.LabelsData],
        intensity_image_reference : napari.types.ImageData,
        intensity_image_of_labels_to_measure : List[napari.types.ImageData] = None,
        size : bool = True, perimeter : bool = False,
        shape : bool = False, position : bool = False,
        moments : bool = False,
        intersection_over_reference_area: float = 0.5,
        suffixes : List[str] = None,
        napari_viewer : Viewer = None) -> "pandas.DataFrame":
    ## Get reference properties
    reference_labels_properties = measure_labels_with_intensity(
        label_image_reference = label_image_reference,
        intensity_image_reference = intensity_image_reference,
        size = size,
        perimeter = perimeter, shape = shape, position = position,
        moments = moments, napari_viewer = napari_viewer
        )
    list_table_linking_labels = []
    list_table_labels_to_measure_properties = []
    # Make labels_to_measure iterable
    if not isinstance(labels_to_measure, list):
        labels_to_measure = [labels_to_measure]
    ## If no intensity_image_of_labels_to_measure provided, use reference intensity
    if intensity_image_of_labels_to_measure is None:
        intensity_image_of_labels_to_measure = [intensity_image_reference]*len(labels_to_measure)
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
            label_image_reference=label_image_reference,
            labels_to_measure=label_image,
            intersection_over_reference_area=intersection_over_reference_area
            )
        list_table_linking_labels.append(table_linking_labels)
            
        labels_to_measure_properties = measure_labels_with_intensity(
            label_image_reference = label_image, 
            intensity_image_reference = intensity_image,
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