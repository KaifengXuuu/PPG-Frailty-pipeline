"""四种规范 representation 门面 / Four canonical representation facades.

中文：集中公开 raw/vector/matrix/fusion 构造。English: Export raw, vector, matrix and fusion builders.
"""

from .feature_matrix import validate_feature_matrix
from .feature_vector import validate_feature_vector
from .fusion import masked_file_mean
from .modes import RepresentationMode, assert_mode
from .raw import RawWindows, build_raw_windows

__all__ = [
    "RawWindows", "RepresentationMode", "assert_mode", "build_raw_windows",
    "masked_file_mean", "validate_feature_matrix", "validate_feature_vector",
]
