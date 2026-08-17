"""Participant/file/window 平衡采样 / Participant-file-window sampling facade.

中文：复用唯一层级采样权重。English: Re-export the sole hierarchy-balanced sampling weights.
"""

from ..training.trainer import participant_file_window_sampling_weights

__all__ = ["participant_file_window_sampling_weights"]
