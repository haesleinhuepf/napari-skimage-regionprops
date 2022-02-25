from ._table import add_table, get_table, TableWidget
from ._regionprops import regionprops
from ._parametric_images import visualize_measurement_on_labels
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.3.0"

from napari_plugin_engine import napari_hook_implementation


@napari_hook_implementation
def napari_experimental_provide_function():
    return [regionprops, visualize_measurement_on_labels]

