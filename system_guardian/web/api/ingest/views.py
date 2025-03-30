"""Views for ingest API."""

from fastapi import APIRouter

from system_guardian.web.api.ingest.datadog import router as datadog_router

router = APIRouter()

# Include Datadog routes
router.include_router(datadog_router, prefix="/datadog", tags=["datadog"])
