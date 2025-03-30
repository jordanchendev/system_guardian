from typing import Awaitable, Callable
import asyncio
import logging

from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from openai import AsyncOpenAI

from system_guardian.db.meta import meta
from system_guardian.db.models import load_all_models
from system_guardian.services.rabbit.lifetime import init_rabbit, shutdown_rabbit
from system_guardian.services.consumers.event_consumer import EventConsumer
from system_guardian.services.ai.engine import AIEngine
from system_guardian.services.vector_db.qdrant_client import get_qdrant_client
from system_guardian.settings import settings
from system_guardian.logging_config import configure_sqlalchemy_logging
from system_guardian.services.vector_db.dependencies import (
    initialize_vector_collections,
)


def _setup_db(app: FastAPI) -> None:  # pragma: no cover
    """
    Creates connection to the database.

    This function creates SQLAlchemy engine instance,
    session_factory for creating sessions
    and stores them in the application's state property.

    :param app: fastAPI application.
    """
    configure_sqlalchemy_logging()
    for logger_name in [
        "sqlalchemy",
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "sqlalchemy.orm",
    ]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL + 10)
        logging.getLogger(logger_name).disabled = True
        logging.getLogger(logger_name).propagate = False

    engine = create_async_engine(str(settings.db_url), echo=False)
    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )
    app.state.db_engine = engine
    app.state.db_session_factory = session_factory


def _setup_ai_engine(app: FastAPI) -> None:  # pragma: no cover
    """
    Initialize AI engine.

    This function creates the AI engine and stores it
    in the application's state property.

    :param app: fastAPI application.
    """
    # Initialize OpenAI client
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Get Qdrant client
    qdrant_client = get_qdrant_client()

    # Initialize Qdrant collections
    asyncio.create_task(_initialize_qdrant_collections(qdrant_client))

    # Initialize AI engine
    ai_engine = AIEngine(
        vector_db_client=qdrant_client,
        llm_client=openai_client,
        embedding_model=settings.openai_embedding_model,
        llm_model=settings.openai_completion_model,
        enable_metrics=True,
    )

    # Store in app state
    app.state.ai_engine = ai_engine


async def _initialize_qdrant_collections(qdrant_client) -> None:  # pragma: no cover
    """
    Initialize Qdrant collections required by the application.

    :param qdrant_client: Qdrant client instance
    """
    logging.getLogger("system_guardian").info("Initializing Qdrant collections")

    # Standard collection names used in the application
    COLLECTIONS = {
        "system_knowledge": 1536,  # OpenAI embedding dimension
        "incidents": 1536,  # OpenAI embedding dimension
    }

    try:
        # Initialize all required collections
        for collection_name, vector_size in COLLECTIONS.items():
            try:
                await qdrant_client.ensure_collection_exists(
                    collection_name=collection_name,
                    vector_size=vector_size,
                    distance="Cosine",
                )
                logging.getLogger("system_guardian").info(
                    f"Collection {collection_name} initialized"
                )
            except Exception as e:
                logging.getLogger("system_guardian").error(
                    f"Failed to initialize collection {collection_name}: {e}"
                )

        logging.getLogger("system_guardian").info(
            "Vector collections initialization completed"
        )
    except Exception as e:
        logging.getLogger("system_guardian").error(
            f"Error initializing Qdrant collections: {e}"
        )


async def get_ai_engine():
    """
    Get AI engine dependency.

    :return: AI engine instance
    """
    from fastapi import Request

    def _get_ai_engine(request: Request):
        return request.app.state.ai_engine

    return _get_ai_engine


async def _create_tables() -> None:  # pragma: no cover
    """Create database tables based on model definitions if they don't exist."""
    from loguru import logger
    from sqlalchemy import text
    from system_guardian.db.base import Base
    from system_guardian.settings import settings
    from sqlalchemy.ext.asyncio import create_async_engine

    # Load all models
    load_all_models()

    # Create engine for table operations
    engine = create_async_engine(str(settings.db_url), echo=False)

    try:
        logger.info("Creating database tables if they don't exist...")
        # Create all tables defined in models
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Verify critical tables
        critical_tables = ["events", "incidents", "resolutions"]
        async with engine.connect() as conn:
            for table in critical_tables:
                result = await conn.execute(
                    text(
                        f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}')"
                    )
                )
                exists = result.scalar()
                logger.info(f"- {table} table: {'✓' if exists else '✗'}")

    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
    finally:
        await engine.dispose()


async def _start_event_consumer(app: FastAPI) -> None:  # pragma: no cover
    """
    Start the event consumer.

    :param app: fastAPI application.
    """
    # Create event consumer
    event_consumer = EventConsumer(app.state.db_session_factory)

    # Store it in app state for later reference
    app.state.event_consumer = event_consumer

    # Start it as a background task
    app.state.event_consumer_task = asyncio.create_task(event_consumer.start())


async def _stop_event_consumer(app: FastAPI) -> None:  # pragma: no cover
    """
    Stop the event consumer.

    :param app: fastAPI application.
    """
    if hasattr(app.state, "event_consumer"):
        # Signal consumer to stop
        await app.state.event_consumer.stop()

        # Wait for the task to complete
        if hasattr(app.state, "event_consumer_task"):
            try:
                await asyncio.wait_for(app.state.event_consumer_task, timeout=5.0)
            except asyncio.TimeoutError:
                # If it doesn't stop in time, cancel it
                app.state.event_consumer_task.cancel()


def register_startup_event(
    app: FastAPI,
) -> Callable[[], Awaitable[None]]:  # pragma: no cover
    """
    Actions to run on application startup.

    This function uses fastAPI app to store data
    in the state, such as db_engine.

    :param app: the fastAPI application.
    :return: function that actually performs actions.
    """

    @app.on_event("startup")
    async def _startup() -> None:  # noqa: WPS430
        app.middleware_stack = None
        _setup_db(app)
        _setup_ai_engine(app)
        await _create_tables()
        init_rabbit(app)
        await _start_event_consumer(app)
        app.middleware_stack = app.build_middleware_stack()
        pass  # noqa: WPS420

    return _startup


def register_shutdown_event(
    app: FastAPI,
) -> Callable[[], Awaitable[None]]:  # pragma: no cover
    """
    Actions to run on application's shutdown.

    :param app: fastAPI application.
    :return: function that actually performs actions.
    """

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # noqa: WPS430
        await _stop_event_consumer(app)
        await app.state.db_engine.dispose()
        await shutdown_rabbit(app)
        pass  # noqa: WPS420

    return _shutdown
