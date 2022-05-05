from ._table import add_table, get_table, TableWidget
from ._regionprops import regionprops, regionprops_table, regionprops_table_all_frames
from ._parametric_images import visualize_measurement_on_labels
from napari_plugin_engine import napari_hook_implementation
from ._load_csv import load_csv

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.4.5"


@napari_hook_implementation
def napari_experimental_provide_function():
    return [regionprops_table, visualize_measurement_on_labels, load_csv]

