import pandas
from typing import List


def merge_measurements_to_reference(
        table_reference_labels_properties: "pandas.DataFrame",
        table_linking_labels: List["pandas.DataFrame"],
        table_labels_to_measure_properties: List["pandas.DataFrame"],
        suffixes=None) -> List["pandas.DataFrame"]:
    """
    Merge measurements from target to reference table through a linking table.

    Parameters
    ----------
    table_reference_labels_properties : pandas.DataFrame
        a table to be used as a reference with a column 'label' and other
        columns with features.
    table_linking_labels : List["pandas.DataFrame"]
        a list of tables. Each table should contain 2 columns, a
        label_reference' and a 'label_target'. Each table row associates a
        target label to a reference label.
    table_labels_to_measure_properties : List["pandas.DataFrame"]
        a list of tables to be used as targets with a column 'label' and other
        columns with features.
    suffixes : List[str], optional
        list of strings containing suffixes to be added to the output table
        columns. If None (default), '_reference' and increasing numbers are
        used as suffixes.

    Returns
    -------
    List[pandas.DataFrame]
        a list of relationship tables, which associate each target label (with
        its properties) to a reference label (with its properties).
    """
    import pandas as pd
    # Shape input to right format
    # Create lists of tables to iterate later
    if not isinstance(table_linking_labels, list):
        list_table_linking_labels = [table_linking_labels]
    else:
        list_table_linking_labels = table_linking_labels
    if not isinstance(table_labels_to_measure_properties, list):
        list_table_labels_to_measure_properties = [
            table_labels_to_measure_properties]
    else:
        list_table_labels_to_measure_properties = \
            table_labels_to_measure_properties
    # Build custom suffixes or check if provided suffixes match data size
    n_measurement_tables = len(list_table_labels_to_measure_properties)
    if suffixes is None:
        n_leading_zeros = n_measurement_tables // 10
        suffixes = ['_reference'] + ['_' + str(i+1).zfill(1+n_leading_zeros)
                                     for i in range(n_measurement_tables)]
    else:
        if len(suffixes) != len(table_labels_to_measure_properties) + 1:
            print(('Error: List of suffixes must have the same length as the'
                  'number of tables containing measurements'))
            return

    # Rename column names with appropriate suffixes
    # Raname reference table columns
    table_reference_labels_properties.columns = [
            props + suffixes[0]
            for props in table_reference_labels_properties.columns]
    # Rename columns of tables with linking labels 
    for i, table_linking_labels in enumerate(list_table_linking_labels):
        table_linking_labels.rename(
                columns={'label_reference': 'label' + suffixes[0],
                         'label_target': 'label' + suffixes[i+1]},
                inplace=True)
    # Rename columns of tables with properties from other channels
    for i, table_labels_to_measure_properties in enumerate(
            list_table_labels_to_measure_properties):
        table_labels_to_measure_properties.columns = [
            props + suffixes[i+1]
            for props in table_labels_to_measure_properties.columns]

    output_table_list = []
    # Consecutively merge linking_labels tables and properties from other 
    # channels tables to the reference table
    for i, table_linking_labels, table_labels_to_measure_properties in zip(
            range(n_measurement_tables),
            list_table_linking_labels,
            list_table_labels_to_measure_properties):
        # Merge other labels to label_reference
        output_table = pd.merge(table_reference_labels_properties,
                                table_linking_labels,
                                how='outer', on='label' + suffixes[0])
        # Fill NaN labels with zeros (if label were not linked, they belong to
        # background)
        output_table['label' + suffixes[i+1]] = output_table[
            'label' + suffixes[i+1]].fillna(0)
        # Merge other properties to output table based on new labels column
        output_table = pd.merge(output_table,
                                table_labels_to_measure_properties,
                                how='outer', on='label' + suffixes[i+1])
        # Ensure label columns type to be integer
        for column in output_table.columns:
            if column.startswith('label'):
                output_table[column] = output_table[column].astype(int)
        # Append output table to list (each table may have different shapes)
        output_table_list.append(output_table)
    return output_table_list


def make_summary_table(table: List["pandas.DataFrame"],
                       suffixes=None,
                       statistics_list=['count',]) -> "pandas.DataFrame":
    """
    Calculate summary statistics of a list of relationship tables.

    For each relationship table, which relates target labels and its properties
    to reference labels (and its properties), calculate summary statistics
    defined by `statistics_list` and concatenates outputs to the rigth as new
    columns.

    Parameters
    ----------
    table : List[pandas.DataFrame]
        a relationship table or a list of them.
    suffixes : List[str], optional
        list of strings containing suffixes to be added to the output table
        columns. If None (default), it looks for strings after 'label_' in the
        tables and uses them as suffixes.
    statistics_list : List[str], optional
        list of strings determining summary statistics to be calculated.
        Possible entries are 'count', 'mean', 'std', 'min', '25%', '50%',
        '75%', 'max'. The percentages correspond to percentiles.

    Returns
    -------
    pandas.DataFrame
        a table containing summary statistics.
    """
    # If not provided, guess suffixes from column names (last string after '_')
    import re
    import pandas as pd
    if suffixes is None:
        suffixes = []
        # get everything after '_' that starts with 'label'
        pattern = 'label*(_\w+)$'
        for tab in table:
            for name in tab.columns:
                matches = re.match(pattern, name)
                if matches is not None:
                    new_entry = matches.group(1)
                    if new_entry not in suffixes:
                        suffixes.append(new_entry)
            if len(suffixes) == 0:
                print(('Could not infer suffixes from column names. Please '
                       'provide a list of suffixes identifying different '
                       'channels'))
    if isinstance(table, pandas.DataFrame):
        table = [table]

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
        summary_tab = grouped[probe_measurement_columns]\
            .describe().reset_index()

        # Filter by selected statistics
        selected_columns = [('label' + suffixes[0], '')]
        for stat in statistics_list:
            for column in summary_tab.columns:

                column_stat = column[-1]
                if stat == column_stat:
                    selected_columns.append(column)
        summary_tab = summary_tab.loc[:, selected_columns]
        print(tab)
        print(tab['label' + suf])
        if counts:
            # counts [label + suf] elements grouped by label_reference
            counts_column = tab.astype(bool).groupby('label' + suffixes[0]).sum()[
                'label' + suf].fillna(0).values
            # if only 'counts' was asked, append to table
            if len(statistics_list) == 0:
                summary_tab['counts' + suf] = counts_column
            # otherwise, insert 'counts' at column just before each suffix
            # features
            else:
                for i, column in enumerate(summary_tab.columns):
                    if (column[0].endswith(suf)):
                        summary_tab.insert(i, 'counts' + suf, counts_column)
                        break

        summary_table_list.append(summary_tab)

    # Join summary tables
    summary_table = summary_table_list[0]
    for summary_tab in summary_table_list[1:]:
        summary_table = pd.concat([
            summary_table,
            summary_tab.iloc[:, 1:]
            ], axis=1)
    # Flatten summary statistics table
    summary_table.columns = [' '.join(col).strip()
                             for col in summary_table.columns.values]
    return summary_table
