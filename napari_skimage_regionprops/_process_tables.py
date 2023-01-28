import napari
import pandas
from typing import List

# TO DO: make 'merge_measurements_to_reference' return a list of tables
# because merging is giving wrong statistics like counts

# measure_labels_in_labels will have to return a list
# notebook showing functions must be updated

def merge_measurements_to_reference(
    table_reference_labels_properties : "pandas.DataFrame",
    table_linking_labels : List["pandas.DataFrame"],
    table_labels_to_measure_properties : List["pandas.DataFrame"],
    suffixes=None) -> "pandas.DataFrame":
    """
    Merge 

    Parameters
    ----------
    table_reference_labels_properties : pandas.DataFrame
        _description_
    table_linking_labels : List[&quot;pandas.DataFrame&quot;]
        _description_
    table_labels_to_measure_properties : List[&quot;pandas.DataFrame&quot;]
        _description_
    suffixes : _type_, optional
        _description_, by default None

    Returns
    -------
    pandas.DataFrame
        _description_
    """    
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
    print('\n\ntable_reference_labels_properties = ', table_reference_labels_properties)

    ## output_table starts with reference labels and their properties
    output_table = table_reference_labels_properties
    ## Consecutively merge linking_labels tables and properties from other channels tables to the reference table
    for i, table_linking_labels, table_labels_to_measure_properties in zip(range(n_measurement_tables), list_table_linking_labels, list_table_labels_to_measure_properties):
        ## !!!Problem!!!: If I do like this, with more than 2 tables, I am also duplicating the second labels
        print('\n\table_linking_labels = ', table_linking_labels)
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

def make_summary_table(table: List["pandas.DataFrame"],
                      suffixes=None,
                      statistics_list = ['count',]) -> "pandas.DataFrame":
    # If not provided, guess suffixes from column names (last string after '_')
    import re
    import pandas as pd
    if suffixes is None:
        suffixes = []
        pattern = 'label*(_\w+)$' # get everything after '_' that starts with 'label'
        for tab in table:
            for name in tab.columns:
                matches = re.match(pattern, name)
                if matches is not None:
                    new_entry = matches.group(1)
                    if new_entry not in suffixes:
                        suffixes.append(new_entry)
            if len(suffixes) == 0:
                print('Could not infer suffixes from column names. Please provide a list of suffixes identifying different channels')

    if 'count' in statistics_list:
        counts = True
        statistics_list.remove('count')
    else:
        counts = False
    
    summary_table_list = []
    for tab, suf in zip(table, suffixes[1:]):

        grouped = tab.groupby('label' + suffixes[0])
        probe_columns = [prop for prop in tab.columns
                            if not prop.endswith(suffixes[0])]
        probe_measurement_columns = [name for name in probe_columns
                            if not name.startswith('label')]
        summary_tab = grouped[probe_measurement_columns].describe().reset_index()

        # Filter by selected statistics
        selected_columns = [('label' + suffixes[0], '')]
        for stat in statistics_list:
            for column in summary_tab.columns:

                column_stat = column[-1]
                if stat == column_stat:
                    selected_columns.append(column)
        summary_tab = summary_tab.loc[:,selected_columns]
        
        if counts:
            # counts [label + suf] elements grouped by label_reference
            counts_column = tab.groupby('label' + suffixes[0]).count()['label' + suf].fillna(0)
            # if only 'counts' was asked, append to table
            if len(statistics_list)==0:
                summary_tab['counts' + suf] = counts_column
            # otherwise, insert 'counts' at column just before each suffix features
            else:
                for i, column in enumerate(summary_tab.columns):
                    if (column[0].endswith(suf)):
                        summary_tab.insert(i, 'counts' + suf, counts_column)
                        break
        
        summary_table_list.append(summary_tab)

    # Join summary tables
    summary_table = summary_table_list[0]
    for summary_tab in summary_table_list[1:]:
        summary_table = pd.concat([summary_table, summary_tab.iloc[:,1:]], axis=1)
    # Flatten summary statistics table
    summary_table.columns = [' '.join(col).strip()
                        for col in summary_table.columns.values]
    return summary_table

def make_summary_table_old(table: "pandas.DataFrame",
                      suffixes=None,
                      statistics_list = ['count',]) -> "pandas.DataFrame":
    # If not provided, guess suffixes from column names (last string after '_')
    import re
    import pandas as pd
    if suffixes is None:
        suffixes = []
        pattern = 'label*(_\w+)$' # get everything after '_' that starts with 'label'
        for name in table.columns:
            matches = re.match(pattern, name)
            if matches is not None:
                new_entry = matches.group(1)
                if new_entry not in suffixes:
                    suffixes.append(new_entry)
        if len(suffixes) == 0:
            print('Could not infer suffixes from column names. Please provide a list of suffixes identifying different channels')
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
    print('\nsummary_table = ', summary_table)
    # Add counts
    if counts:
        for suf in suffixes[1:]:
            # counts [label + suf] elements grouped by label_reference
            print('\n\ntable = ', table, '\n\n\n')
            print('counts = ', table.groupby('label' + suffixes[0]).count())
            counts_column = table.groupby('label' + suffixes[0]).count()['label' + suf].fillna(0)
            print(suf, statistics_list)
            # if only 'counts' was asked, append to table
            if len(statistics_list)==0:
                summary_table['counts' + suf] = counts_column
            # otherwise, insert 'counts' at column just before each suffix features
            else:
                for i, column in enumerate(summary_table.columns):
                    if (column[0].endswith(suf)):
                        summary_table.insert(i, 'counts' + suf, counts_column)
                        break
    print('\nsummary_table2 = ', summary_table)
    # Flatten summary statistics table
    summary_table.columns = [' '.join(col).strip()
                     for col in summary_table.columns.values]
    print('\nsummary_table3 = ', summary_table)
    return summary_table