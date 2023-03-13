from typing import List
from magicgui import magic_factory
from napari_tools_menu import register_dock_widget


def make_element_wise_dict(list_of_keys, list_of_values):
    """Return an element-wise dictionary of two lists.

    From a list of keys and a list of values, return a dictionary linking
    those lists. Both lists must have the same length.

    Parameters
    ----------
    list_of_keys : List
        list to be used as dictionary keys.
    list_of_values : List
        list to be used as dictionary values.

    Returns
    -------
    Dictionary
    """
    return {k: v for k, v in zip(list_of_keys, list_of_values)}


def _connect_events(widget):
    def toggle_intensity_widgets(event):
        widget.intensity_image_reference.visible = event
        # If relate_to_other_channels is True when enabling intensity, then enable
        # intensity image inputs
        if (event == True) & (widget.relate_to_other_channels.value == True):
            widget.intensity_images_other_channels.visible = True
        else:
            widget.intensity_images_other_channels.visible = False

    def toggle_relate_to_other_channels_widgets(event):
        widget.label_images_other_channels.visible = event
        widget.intersection_over_other_channel_obj_area.visible = event
        widget.configure_summary_statistics.visible = event
        if (event == True) & (widget.intensity.value == True):
            widget.intensity_images_other_channels.visible = True
        else:
            widget.intensity_images_other_channels.visible = False
        if event == True:
            if widget.configure_summary_statistics.value == True:
                toggle_summary_statistics_widgets(True)
        else:
            widget.configure_summary_statistics.value = False
            toggle_summary_statistics_widgets(False)

    def toggle_summary_statistics_widgets(event):
        widget.counts.visible = event
        widget.average.visible = event
        widget.std.visible = event
        widget.minimum.visible = event
        widget.percentile_25.visible = event
        widget.median.visible = event
        widget.percentile_75.visible = event
        widget.maximum.visible = event

    widget.intensity.changed.connect(toggle_intensity_widgets)
    widget.relate_to_other_channels.changed.connect(toggle_relate_to_other_channels_widgets)
    widget.configure_summary_statistics.changed.connect(
        toggle_summary_statistics_widgets
        )

    # Intial visibility states
    widget.intensity_image_reference.visible = False
    widget.intensity_images_other_channels.visible = False
    widget.label_images_other_channels.visible = False
    widget.configure_summary_statistics.visible = False
    widget.intersection_over_other_channel_obj_area.visible = False
    widget.counts.visible = False
    widget.average.visible = False
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
    'label_images_other_channels': {
        'label': 'Label Image(s) from other Channel(s)'
    },
    'intensity_images_other_channels': {
        'label': 'Intensity Image(s) from other Channel(s)'
    },
    'relate_to_other_channels': {
        'label': 'relate to other channel(s)'
    },
    'intersection_over_other_channel_obj_area': {
        'widget_type': 'FloatSlider',
        'min': 0,
        'max': 1,
        'step': 0.1,
        'tooltip': ('A ratio that determines if an object in one channel '
                    'belongs to another in a reference channel.\nIt goes from '
                    '0 (target object always belongs to background) to 1 '
                    '(target object belongs to reference object only if '
                    'completely inside reference)')
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
    menu="Measurement tables > Object Features/Properties (scikit-image, nsr)")
# Need magic factory to make hidding and showing functionality available
@magic_factory(widget_init=_connect_events,
               layout='vertical',
               label_image_reference=widgets_layout_settings[
                   'label_image_reference'],
               intensity_image_reference=widgets_layout_settings[
                   'intensity_image_reference'],
               label_images_other_channels=widgets_layout_settings[
                   'label_images_other_channels'],
               intensity_images_other_channels=widgets_layout_settings[
                   'intensity_images_other_channels'],
               intersection_over_other_channel_obj_area=widgets_layout_settings[
                   'intersection_over_other_channel_obj_area'],
               relate_to_other_channels=widgets_layout_settings[
                   'relate_to_other_channels'],
               std=widgets_layout_settings['std'],
               percentile_25=widgets_layout_settings['percentile_25'],
               percentile_75=widgets_layout_settings['percentile_75'])
def regionprops_measure_relationship_to_other_channels(
        label_image_reference: "napari.types.LabelsData",
        intensity_image_reference: "napari.types.ImageData",
        label_images_other_channels: List["napari.types.LabelsData"],
        intensity_images_other_channels: List["napari.types.ImageData"],
        intensity: bool = False,
        relate_to_other_channels: bool = False,
        size: bool = True,
        perimeter: bool = False,
        shape: bool = False,
        position: bool = False,
        moments: bool = False,
        intersection_over_other_channel_obj_area: float = 0.5,
        configure_summary_statistics: bool = False,
        counts: bool = False,
        average: bool = True,
        std: bool = False,
        minimum: bool = False,
        percentile_25: bool = False,
        median: bool = False,
        percentile_75: bool = False,
        maximum: bool = False,
        napari_viewer: "napari.Viewer" = None) -> "pandas.DataFrame":
    """
    Add a table widget to a napari viewer with mapped summary statistics.

    Adds a table widget to a given napari viewer with summary statistics that
    relates objects of one reference label image to objects in other label
    images. Matched intensity images should be given for intensity
    measurements. If a single label image is given, it executes
    regular 'regionprops_table' function.

    Parameters
    ----------
    label_image_reference : array_like
        a label image used as a reference.
    intensity_image_reference : array_like
        an intensity image, matching the `label_image_reference` shape.
    label_images_other_channels : List[array_like]
        a list of label images, whose measured features will be related to
        labels from `label_image_reference`.
    intensity_images_other_channels : List[array_like]
        a list of label images, whose measured intensity features will be
        related to labels from `label_image_reference`.
    intensity : bool
        a flag indicating that intensity features should be measured. If
        `True`, `intensity_image_reference` must be given.
    relate_to_other_channels : bool
        a flag indicating that relate_to_other_channels analysis should be performed,
        relating labels from `label_image_reference` to
        `label_images_other_channels`. If `True`, `label_images_other_channels`
        must be provided.
    size : bool
        a flag indicating to measure size features. By default True.
    perimeter :  bool
        a flag indicating to measure perimeter features. By default False.
    shape : bool
        a flag indicating to measure shape features. By default False.
    position : bool
        a flag indicating to measure position features. By default False.
    moments : bool
        a flag indicating to measure moments features. By default False.
    intersection_over_other_channel_obj_area : float
        a ratio of areas that indicates whether an object is considered
        'inside' another. It is the intersection area divided by the target
        object area. It goes from 0 to 1. 0 indicates object always belongs to
        background. 1 indicates object must be completely inside another in the
        reference labels. By default 0.5.
    configure_summary_statistics : bool
        a flag to make summary statistics visible in the GUI.
    counts :  bool
        a flag indicating to calculate how many objects are 'inside' each
        reference labeled object. By default False.
    average : bool
        a flag indicating to calculate the average of the objects features
        'inside' each reference labeled object. By default True.
    std : bool
        a flag indicating to calculate the standard deviation of the objects
        features 'inside' each reference labeled object. By default False.
    minimum : bool
        a flag indicating to calculate the minimum of the objects features
        'inside' each reference labeled object. By default False.
    percentile_25 : bool
        a flag indicating to calculate the 25% percentile of the objects
        features 'inside' each reference labeled object. By default False.
    median : bool
        a flag indicating to calculate the median of the objects features
        'inside' each reference labeled object. By default False.
    percentile_75 : bool
        a flag indicating to calculate the 75% percentile of the objects
        features 'inside' each reference labeled object. By default False.
    maximum : bool
        a flag indicating to calculate the maximum of the objects features
        'inside' each reference labeled object. By default False.
    napari_viewer : napari.Viewer
        a handle to an instance of the napari Viewer. By default None.

    Returns
    -------
    pandas.DataFrame
        A table where the first column contains the labels from the reference
        label image and the other columns contain summary statistics of objects
        in other channels.
    """
    from napari.utils import notifications
    from ._process_tables import make_summary_table
    import napari
    import numpy as np

    statistics_list = []
    if counts:
        statistics_list += ['count']
    if average:
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

    # Stores reference labels layer and suffixes from layer names
    if napari_viewer is not None:
        for layer in napari_viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                if np.array_equal(layer.data, label_image_reference):
                    reference_labels_layer = layer
                    reference_suffix = '_' + layer.name
        suffixes.insert(0, reference_suffix)
        for i, labels in enumerate(label_images_other_channels):
            for layer in napari_viewer.layers:
                if isinstance(layer, napari.layers.Labels) and np.array_equal(layer.data, labels):
                    new_suffix = '_' + layer.name
                    if new_suffix not in suffixes:
                        suffixes.append(new_suffix)
                    else:
                        # labels_to_measure layer is a duplicated reference
                        n_leading_zeros = len(label_images_other_channels) // 10
                        suffixes.append(new_suffix + str(i+1).zfill(1+n_leading_zeros))
    # Ensures proper iterable inputs
    else:
        if not isinstance(label_images_other_channels, list):
            label_images_other_channels = [label_images_other_channels]
        if not isinstance(intensity_images_other_channels, list):
            intensity_images_other_channels = [intensity_images_other_channels]
        # in case function is used without viewer and suffixes are missing
        if len(suffixes) == 0:
            n_leading_zeros = len(label_images_other_channels) // 10
            suffixes = ['_reference'] + ['_' + str(i+1).zfill(
                1+n_leading_zeros)
                for i in range(len(label_images_other_channels))]

    # Single image measurements
    if relate_to_other_channels == False:
        # Without intensity measurements
        if intensity == False:
            table = measure_labels(
                label_image_reference=label_image_reference,
                size=size, perimeter=perimeter,
                shape=shape, position=position,
                moments=moments)
        # Or with intensity measurements
        else:
            table = measure_labels_with_intensity(
                label_image_reference=label_image_reference,
                intensity_image_reference=intensity_image_reference,
                size=size, perimeter=perimeter,
                shape=shape, position=position,
                moments=moments)

    # More than one label image measurements
    else:
        # If no summary statistics or no features, call regionprops to
        # return table with only labels
        if len(statistics_list) == 0 or not any([intensity, size, perimeter, shape, position, moments]):
            table = measure_labels(
                label_image_reference=label_image_reference,
                size=size, perimeter=perimeter,
                shape=shape, position=position,
                moments=moments)
        else:
            # Check if user provided 'label_images_other_channels'
            if len(label_images_other_channels) == 0:
                notifications.show_warning(('Error! Add a labels layer to '
                                            '\'Label Image(s) from other '
                                            'Channel(s)\' when \'relate '
                                            'to other channel(s)\' is enabled!')
                                            )
                return
            
            # Without intensity measurements
            if intensity == False:
                table = measure_labels_in_labels(
                    label_image_reference=label_image_reference,
                    labels_to_measure=label_images_other_channels,
                    size=size,
                    perimeter=perimeter,
                    shape=shape,
                    position=position,
                    moments=moments,
                    intersection_over_other_channel_obj_area=intersection_over_other_channel_obj_area,
                    suffixes=suffixes)

            # Or with intensity measurements
            else:
                # Check if user provided 'intensity_images_other_channels'
                if len(intensity_images_other_channels) == 0:
                    notifications.show_warning(('Error! Add an image layer to '
                                            '\'Intensity Image(s) from other '
                                            'Channel(s)\' when both \'relate '
                                            'to other channel(s)\' and \'intensity\''
                                             ' are enabled!')
                                            )
                    return
                table = measure_labels_in_labels_with_intensity(
                label_image_reference=label_image_reference,
                labels_to_measure=label_images_other_channels,
                intensity_image_reference=intensity_image_reference,
                intensity_image_of_labels_to_measure=intensity_images_other_channels,
                size=size, perimeter=perimeter,
                shape=shape, position=position,
                moments=moments,
                intersection_over_other_channel_obj_area=intersection_over_other_channel_obj_area,
                suffixes=suffixes)
            # Compute summary statistics instead of individual relationships
            # This guarantees no repeated numbers in the 'label' column
            table = make_summary_table(table,
                                    suffixes=suffixes,
                                    statistics_list=statistics_list)
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
        add_table(reference_labels_layer, napari_viewer, tabify = True)

    return table


def link_two_label_images(label_image_reference: "napari.types.LabelsData",
                          labels_to_measure: "napari.types.LabelsData",
                          intersection_over_other_channel_obj_area: float = 0.5
                          ) -> "pandas.DataFrame":
    """
    Associate each label from a reference to a target label image.

    It takes two label images, being the first a reference and the
    second a target, and returns a table with two columns that
    associates each label in the target image to a label (or background)
    in the reference image.

    Parameters
    ----------
    label_image_reference : napari.types.LabelsData
        a label image to be used as reference labels.
    labels_to_measure : napari.types.LabelsData
        a label image to be used as target labels.
    intersection_over_other_channel_obj_area : float, optional
        an area ratio threshold value that determines if a label in
        the target image belongs to another in the reference image.
        The area ratio is calculated by the intersection area divided
        by the target label area. If this area ratio is bigger
        or equal to `intersection_over_other_channel_obj_area`, the target object
        gets associated to the reference object, otherwise, it gets
        associated to the background. This parameter goes from 0 to 1,
        by default 0.5.

    Returns
    -------
    pandas.DataFrame
        a table with two columns: 'label_reference' and 'label_target'.
        Each row expresses that target label is associated with the
        corresponding reference label.
    """
    import numpy as np
    import pandas as pd
    from skimage.measure import regionprops_table as sk_regionprops_table

    def highest_overlap(regionmask, label_image_reference,
                        overlap_threshold=intersection_over_other_channel_obj_area):
        """
        Get the label number with highest overlap with label in another image.

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

    # Create table that links labels from scanning channel to reference channel
    table_linking_labels = pd.DataFrame(
        sk_regionprops_table(label_image=np.asarray(labels_to_measure).astype(int),
                             intensity_image=np.asarray(label_image_reference).astype(int),
                             properties=['label', ],
                             extra_properties=[highest_overlap]
                             )
    ).astype(int)
    # rename column
    table_linking_labels.rename(columns={
        'highest_overlap': 'label_reference',
        'label': 'label_target'
        },
        inplace=True)
    # Re-order columns
    table_linking_labels = table_linking_labels[
        ['label_reference', 'label_target']
        ]
    # Add eventual missing reference labels to table, they belong to background
    bg_labels_list = []
    for i in np.unique(label_image_reference)[1:].tolist():
        if i not in table_linking_labels['label_reference'].values:
            bg_labels_list.append([i, 0])
    bg_labels = pd.DataFrame(bg_labels_list,
                             columns=['label_reference', 'label_target'])
    table_linking_labels = pd.concat(
        [table_linking_labels, bg_labels], axis=0) \
        .sort_values(by=['label_reference', 'label_target']) \
        .reset_index(drop=True)

    return table_linking_labels


def measure_labels(label_image_reference: "napari.types.LabelsData",
                   size: bool = True,
                   perimeter: bool = False,
                   shape: bool = False,
                   position: bool = False,
                   moments: bool = False,
                   napari_viewer: "napari.Viewer" = None) -> "pandas.DataFrame":
    """
    Measure a label image features.

    Measure features using skimage.regionprops.

    Parameters
    ----------
    label_image_reference : napari.types.LabelsData, array_like
        a label image to measure features.
    size : bool, optional
        measure size related features.
        By default True.
    perimeter : bool, optional
        measure perimeter related features.
        By default False.
    shape : bool, optional
        measure shape related features.
        By default False.
    position : bool, optional
        measure position related features.
        By default False.
    moments : bool, optional
        measure moments related features.
        By default False.
    napari_viewer : Viewer, optional
        a handle to an instance of the napari Viewer, by default None.

    Returns
    -------
    pandas.DataFrame
        A table containing labels and corresponding measured features.
    """
    from ._regionprops import regionprops_table
    import numpy as np

    table = regionprops_table(image=np.zeros_like(label_image_reference),
                              labels=label_image_reference,
                              size=size,
                              intensity=False,
                              perimeter=perimeter,
                              shape=shape,
                              position=position,
                              moments=moments,
                              napari_viewer=napari_viewer)
    return table


def measure_labels_with_intensity(
        label_image_reference: "napari.types.LabelsData",
        intensity_image_reference: "napari.types.ImageData",
        size: bool = True,
        perimeter: bool = False,
        shape: bool = False,
        position: bool = False,
        moments: bool = False,
        napari_viewer: "napari.Viewer" = None) -> "pandas.DataFrame":
    """
    Measure a label image features, including intensity features.

    Measure features using skimage.regionprops.

    Parameters
    ----------
    label_image_reference : napari.types.LabelsData, array_like
        a label image to measure features.
    intensity_image_reference : napari.types.ImageData, array_like
        an intensity image to measure features.
    size : bool, optional
        measure size related features.
        By default True.
    perimeter : bool, optional
        measure perimeter related features.
        By default False.
    shape : bool, optional
        measure shape related features.
        By default False.
    position : bool, optional
        measure position related features.
        By default False.
    moments : bool, optional
        measure moments related features.
        By default False.
    napari_viewer : Viewer, optional
        a handle to an instance of the napari Viewer, by default None.

    Returns
    -------
    pandas.DataFrame
        A table containing labels and corresponding measured features.
    """
    from ._regionprops import regionprops_table

    table = regionprops_table(image=intensity_image_reference,
                              labels=label_image_reference,
                              size=size,
                              intensity=True,
                              perimeter=perimeter,
                              shape=shape, position=position,
                              moments=moments,
                              napari_viewer=napari_viewer)
    return table


def measure_labels_in_labels(label_image_reference: "napari.types.LabelsData",
                             labels_to_measure: List["napari.types.LabelsData"],
                             size: bool = True,
                             perimeter: bool = False,
                             shape: bool = False,
                             position: bool = False,
                             moments: bool = False,
                             intersection_over_other_channel_obj_area: float = 0.5,
                             suffixes: List[str] = None,
                             napari_viewer: "napari.Viewer" = None
                             ) -> List["pandas.DataFrame"]:
    """
    Measure label images features and associates them.

    Measure features from a reference label image and
    one or more target label images. It returns a table
    where each column is a feature and each row associates
    a label (and its features) from `label_image_reference`
    to a target label (and its features) from `labels_to_measure`.

    Parameters
    ----------
    label_image_reference : napari.types.LabelsData
        a reference label image to measure features.
    labels_to_measure : List[napari.types.LabelsData]
        a list of target label images to measure features.
    size : bool, optional
        measure size related features.
        By default True.
    perimeter : bool, optional
        measure perimeter related features.
        By default False.
    shape : bool, optional
        measure shape related features.
        By default False.
    position : bool, optional
        measure position related features.
        By default False.
    moments : bool, optional
        measure moments related features.
        By default False.
    intersection_over_other_channel_obj_area : float, optional
        an area ratio threshold value that determines if a label in
        the target image belongs to another in the reference image.
        The area ratio is calculated by the intersection area divided
        by the target label area. If this area ratio is bigger
        or equal to `intersection_over_other_channel_obj_area`, the target object
        gets associated to the reference object, otherwise, it gets
        associated to the background. This parameter goes from 0 to 1,
        by default 0.5.
    suffixes : List[str], optional
        a list of suffixes to be appended to table columns.
        If None (default), it appends '_reference' for reference features
        and increasing numbers for target features.
    napari_viewer : Viewer, optional
        a handle to an instance of the napari Viewer, by default None.

    Returns
    -------
    List[pandas.DataFrame]
        A list of tables containing labels and corresponding measured features.
        Each row associates reference label features to target label features.
    """
    from ._process_tables import merge_measurements_to_reference

    # Get reference properties
    reference_labels_properties = measure_labels(
        label_image_reference=label_image_reference,
        size=size,
        perimeter=perimeter,
        shape=shape,
        position=position,
        moments=moments,
        napari_viewer=napari_viewer
        )
    list_table_linking_labels = []
    list_table_other_channel_labels_properties = []
    # Make labels_to_measure iterable
    if not isinstance(labels_to_measure, list):
        labels_to_measure = [labels_to_measure]
    # Link each labels_to_measure image to reference and get their properties
    for label_image in labels_to_measure:
        table_linking_labels = link_two_label_images(
            label_image_reference=label_image_reference,
            labels_to_measure=label_image,
            intersection_over_other_channel_obj_area=intersection_over_other_channel_obj_area
            )
        list_table_linking_labels.append(table_linking_labels)

        labels_to_measure_properties = measure_labels(
            label_image_reference=label_image,
            size=size,
            perimeter=perimeter,
            shape=shape,
            position=position,
            moments=moments,
            napari_viewer=napari_viewer
            )
        list_table_other_channel_labels_properties.append(
            labels_to_measure_properties)

    # Merge each table with target label properties to table with reference
    # label properties
    table_list = merge_measurements_to_reference(
        table_reference_labels_properties=reference_labels_properties,
        table_linking_labels=list_table_linking_labels,
        table_other_channel_labels_properties=list_table_other_channel_labels_properties,
        suffixes=suffixes)
    return table_list


def measure_labels_in_labels_with_intensity(
        label_image_reference: "napari.types.LabelsData",
        labels_to_measure: List["napari.types.LabelsData"],
        intensity_image_reference: "napari.types.ImageData",
        intensity_image_of_labels_to_measure: List["napari.types.ImageData"] = None,
        size: bool = True,
        perimeter: bool = False,
        shape: bool = False,
        position: bool = False,
        moments: bool = False,
        intersection_over_other_channel_obj_area: float = 0.5,
        suffixes: List[str] = None,
        napari_viewer: "napari.Viewer" = None) -> List["pandas.DataFrame"]:
    """
    Measure label images features, including intensity, and associates them.

    Measure features from a reference label image and
    one or more target label images. It also measure intensity features
    from a reference intensity image and from one or more target intensity
    images. It returns a table
    where each column is a feature and each row associates
    a label (and its features) from `label_image_reference`
    to a target label (and its features) from `labels_to_measure`.

    Parameters
    ----------
    label_image_reference : napari.types.LabelsData
        a reference label image to measure features.
    labels_to_measure : List[napari.types.LabelsData], array_like
        a list of target label images to measure features.
    intensity_image_reference : napari.types.ImageData, array_like
        a reference intensity image to measure intensity features.
    intensity_image_of_labels_to_measure : List[napari.types.ImageData],
    array_like, optional
        a list of target intensity images to measure intensity features.
        If None (default), intensity features are extracted from 
        `intensity_image_reference`.
    size : bool, optional
        measure size related features.
        By default True.
    perimeter : bool, optional
        measure perimeter related features.
        By default False.
    shape : bool, optional
        measure shape related features.
        By default False.
    position : bool, optional
        measure position related features.
        By default False.
    moments : bool, optional
        measure moments related features.
        By default False.
    intersection_over_other_channel_obj_area : float, optional
        an area ratio threshold value that determines if a label in
        the target image belongs to another in the reference image.
        The area ratio is calculated by the intersection area divided
        by the target label area. If this area ratio is bigger
        or equal to `intersection_over_other_channel_obj_area`, the target object
        gets associated to the reference object, otherwise, it gets
        associated to the background. This parameter goes from 0 to 1,
        by default 0.5.
    suffixes : List[str], optional
        a list of suffixes to be appended to table columns.
        If None (default), it appends '_reference' for reference features
        and increasing numbers for target features.
    napari_viewer : Viewer, optional
        a handle to an instance of the napari Viewer, by default None.

    Returns
    -------
    List[pandas.DataFrame]
        A list of tables containing labels and corresponding measured features.
        Each row associates reference label features to target label features.
    """
    from ._process_tables import merge_measurements_to_reference

    # Get reference properties
    reference_labels_properties = measure_labels_with_intensity(
        label_image_reference=label_image_reference,
        intensity_image_reference=intensity_image_reference,
        size=size,
        perimeter=perimeter,
        shape=shape,
        position=position,
        moments=moments,
        napari_viewer=napari_viewer
        )
    list_table_linking_labels = []
    list_table_other_channel_labels_properties = []
    # Make labels_to_measure iterable
    if not isinstance(labels_to_measure, list):
        labels_to_measure = [labels_to_measure]
    # If no intensity_image_of_labels_to_measure provided, use reference
    # intensity
    if intensity_image_of_labels_to_measure is None:
        intensity_image_of_labels_to_measure = [intensity_image_reference] * \
            len(labels_to_measure)
    # If intensity_image_of_labels_to_measure provided, check if sizes match
    else:
        # Make intensity_image_of_labels_to_measure iterable
        if not isinstance(intensity_image_of_labels_to_measure, list):
            intensity_image_of_labels_to_measure = \
                [intensity_image_of_labels_to_measure]
        if len(intensity_image_of_labels_to_measure) != len(labels_to_measure):
            print(('Error! Length of intensity_image_of_labels_to_measure and'
                   'labels_to_measure must match.'))
            return

    # Link each labels_to_measure image to reference and get their properties
    for label_image, intensity_image in zip(
            labels_to_measure, intensity_image_of_labels_to_measure):
        table_linking_labels = link_two_label_images(
            label_image_reference=label_image_reference,
            labels_to_measure=label_image,
            intersection_over_other_channel_obj_area=intersection_over_other_channel_obj_area
            )
        list_table_linking_labels.append(table_linking_labels)

        labels_to_measure_properties = measure_labels_with_intensity(
            label_image_reference=label_image,
            intensity_image_reference=intensity_image,
            size=size,
            perimeter=perimeter,
            shape=shape,
            position=position,
            moments=moments,
            napari_viewer=napari_viewer
            )
        list_table_other_channel_labels_properties.append(
            labels_to_measure_properties)

    # Merge each table with target label properties to table with reference
    # label properties
    table_list = merge_measurements_to_reference(
        table_reference_labels_properties=reference_labels_properties,
        table_linking_labels=list_table_linking_labels,
        table_other_channel_labels_properties=list_table_other_channel_labels_properties,
        suffixes=suffixes)
    return table_list
