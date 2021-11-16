from ._table import add_table, get_table, TableWidget

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.11"
