"""Dash inspection application for the V2 research pipeline.

The dashboard is deliberately a presentation adapter: it calls canonical
pipeline services and never owns signal-processing or model algorithms.
"""

from .app import create_app
from .job_manager import StudyJobManager
from .preview_service import PipelinePreviewService

__all__ = ["PipelinePreviewService", "StudyJobManager", "create_app"]
