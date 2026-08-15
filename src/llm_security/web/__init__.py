"""Web application for upload, analysis, approval, and patched downloads."""

from .service import JobRecord, JobStatus, WebJobService, WebSettings

__all__ = ["JobRecord", "JobStatus", "WebJobService", "WebSettings"]
