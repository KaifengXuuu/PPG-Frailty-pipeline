"""Shared fail-closed exceptions for the active recording caches."""


class CacheMissError(FileNotFoundError):
    """The requested complete provenance identity is absent."""


class StaleCacheError(ValueError):
    """Cached metadata or payload does not match its provenance."""
