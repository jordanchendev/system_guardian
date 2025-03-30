from importlib import metadata

from fastapi import FastAPI
from fastapi.responses import UJSONResponse
from loguru import logger

from system_guardian.logging_config import configure_logging, InterceptHandler
from system_guardian.web.api.router import api_router
from system_guardian.web.lifetime import register_shutdown_event, register_startup_event


def get_app() -> FastAPI:
    """
    Get FastAPI application.

    This is the main constructor of an application.

    :return: application.
    """
    # Configure logging first
    configure_logging()
    
    # Intercept standard library logging
    interceptor = InterceptHandler()
    interceptor.intercept_all_loggers()
    
    logger.info("Starting System Guardian application")
    
    app = FastAPI(
        title="system_guardian",
        version=metadata.version("system_guardian"),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        default_response_class=UJSONResponse,
    )

    # Adds startup and shutdown events.
    register_startup_event(app)
    register_shutdown_event(app)

    # Main router for the API.
    app.include_router(router=api_router, prefix="/api")
    
    logger.info("Application startup complete")

    return app
