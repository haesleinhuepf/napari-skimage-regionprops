from ._table import add_table, get_table, TableWidget
from ._regionprops import regionprops

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.2.4"
