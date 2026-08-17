"""四种规范 representation 门面 / Four canonical representation facades.

中文：集中公开 raw/vector/matrix/fusion 构造。English: Export raw, vector, matrix and fusion builders.
"""

from .feature_matrix import validate_feature_matrix
from .feature_vector import validate_feature_vector
from .fusion import masked_file_mean
from .imu_transform import (
    FoldImuChannelTransform,
    IMU_CHANNEL_SCHEMA,
    IMU_TRANSFORM_SCHEMA_VERSION,
    apply_fold_imu_channel_transform,
    fit_fold_imu_channel_transform,
    transform_raw_windows_imu,
)
from .modes import RepresentationMode, assert_mode
from .motion import (
    MOTION_AUGMENTED_CHANNEL_SCHEMA,
    MOTION_AUGMENTED_SCHEMA_SHA256,
    MOTION_DERIVED_AUGMENTATION_PROFILE_ID,
    MOTION_NETWORK_CHANNEL_SCHEMA,
    MOTION_NETWORK_SCHEMA_SHA256,
    MOTION_REFERENCE_PROFILE_ID,
    MOTION_WINDOW_SAMPLES,
    MotionFoldImuTransform,
    MotionWindowTensors,
    apply_motion_fold_imu_transform,
    build_motion_window_tensors,
    fit_motion_fold_imu_transform,
    motion_network_schema_payload,
)
from .raw import RawWindows, build_raw_windows

__all__ = [
    "FoldImuChannelTransform", "IMU_CHANNEL_SCHEMA", "IMU_TRANSFORM_SCHEMA_VERSION",
    "RawWindows", "RepresentationMode", "apply_fold_imu_channel_transform",
    "MOTION_AUGMENTED_CHANNEL_SCHEMA", "MOTION_AUGMENTED_SCHEMA_SHA256",
    "MOTION_DERIVED_AUGMENTATION_PROFILE_ID",
    "MOTION_NETWORK_CHANNEL_SCHEMA", "MOTION_NETWORK_SCHEMA_SHA256",
    "MOTION_REFERENCE_PROFILE_ID",
    "MOTION_WINDOW_SAMPLES", "MotionFoldImuTransform", "MotionWindowTensors",
    "apply_motion_fold_imu_transform", "build_motion_window_tensors",
    "fit_motion_fold_imu_transform", "motion_network_schema_payload",
    "assert_mode", "build_raw_windows", "fit_fold_imu_channel_transform",
    "masked_file_mean", "transform_raw_windows_imu", "validate_feature_matrix",
    "validate_feature_vector",
]
