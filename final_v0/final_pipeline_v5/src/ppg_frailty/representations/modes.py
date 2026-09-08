"""Representation 枚举检查 / Representation enum validation."""

from ..contracts import RepresentationMode

def assert_mode(value: RepresentationMode | str) -> RepresentationMode:
    """返回严格枚举或失败 / Return a strict enum or fail."""

    return value if isinstance(value, RepresentationMode) else RepresentationMode(value)


__all__ = ["RepresentationMode", "assert_mode"]
