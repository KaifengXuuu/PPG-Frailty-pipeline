"""Dash control panel for the V5 research pipeline.

The dashboard is deliberately a presentation adapter: it calls canonical
pipeline services and never owns signal-processing or model algorithms.
"""
from .app import create_app
from .control_service import V5ControlService
from .job_manager import DashboardJobManager, StudyJobManager
from .preview_service import PipelinePreviewService

__all__ = ['DashboardJobManager', 'PipelinePreviewService', 'StudyJobManager', 'V5ControlService', 'create_app']
