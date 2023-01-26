import napari
import pandas
from typing import List

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
    # Ensure label columns type to be integer
    for column in output_table.columns:
        if column.startswith('label'):
            output_table[column] = output_table[column].astype(int)
    return output_table

def make_summary_table(table: "pandas.DataFrame",
                      suffixes=None,
                      statistics_list = ['count',]) -> "pandas.DataFrame":
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
    summary_table = grouped[probe_measurement_columns].describe().reset_index()
    
    # Mark counts to be added later in a single column (otherwise counts are
    # added for every property, which is redundant)
    if 'count' in statistics_list:
        counts = True
        statistics_list.remove('count')
    else:
        counts = False
        
    # Filter by selected statistics
    selected_columns = [('label' + suffixes[0], '')]
    for stat in statistics_list:
        for column in summary_table.columns:

            column_stat = column[-1]
            if stat == column_stat:
                selected_columns.append(column)
    summary_table = summary_table.loc[:,selected_columns]
    
    # Add counts
    if counts:
        for suf in suffixes[1:]:
            counts_column = table.groupby('label' + suffixes[0]).count()['label' + suf].fillna(0)
            if summary_table.shape[-1]==1:
                summary_table['counts' + suf] = counts_column
            else:
                for i, column in enumerate(summary_table.columns):
                    if (column[0].endswith(suf)):
                        summary_table.insert(i, 'counts' + suf, counts_column)
                        break
    # Flatten summary statistics table
    summary_table.columns = [' '.join(col).strip()
                     for col in summary_table.columns.values]
    return summary_table