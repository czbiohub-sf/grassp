import sys

from . import datasets, io
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl

# has to be done at the end, after everything has been imported
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "pp", "pl"]})

# from ._utils import annotate_doc_types
# annotate_doc_types(sys.modules[__name__], "scanpy")
# del sys, annotate_doc_types
