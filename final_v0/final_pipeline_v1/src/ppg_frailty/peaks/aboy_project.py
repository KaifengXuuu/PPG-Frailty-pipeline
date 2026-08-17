"""项目 pulse detector 规范入口 / Canonical project pulse-detector entry.

中文：名称明确为 project detector，避免错误宣称为未逐项复现的原始 Aboy。
English: The name explicitly says project detector and does not claim exact Aboy parity.
"""

from ..signal.peaks import DETECTOR_VERSION, detect_pulses

DETECTOR_NAME = f"project_pulse_detector:{DETECTOR_VERSION}"

__all__ = ["DETECTOR_NAME", "detect_pulses"]
