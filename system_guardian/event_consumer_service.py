"""
Event Consumer Service

This module provides a standalone service that consumes event messages
from RabbitMQ and stores them in the database.

Usage:
    python -m system_guardian.event_consumer_service
"""

import asyncio
import signal
import sys
import logging
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from openai import AsyncOpenAI
from sqlalchemy import text
from aio_pika import connect_robust
from aio_pika.pool import Pool

from system_guardian.db.models import load_all_models
from system_guardian.services.consumers.event_consumer import EventConsumer
from system_guardian.services.ai.engine import AIEngine
from system_guardian.services.vector_db.dependencies import get_qdrant_client
from system_guardian.settings import settings
from system_guardian.logging_config import configure_sqlalchemy_logging

# Ensure SQLAlchemy logging is disabled after importing any SQLAlchemy
configure_sqlalchemy_logging()

# Ensure SQLAlchemy logging is completely disabled
for logger_name in [
    "sqlalchemy",
    "sqlalchemy.engine",
    "sqlalchemy.pool",
    "sqlalchemy.orm",
]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 10)
    logging.getLogger(logger_name).disabled = True
    logging.getLogger(logger_name).propagate = False


# Initialize RabbitMQ connection pool for standalone service
async def create_rmq_channel_pool():
    """Create RabbitMQ channel pool"""
    logger.info("Creating RabbitMQ channel pool...")
    from aio_pika.pool import Pool

    async def get_connection():
        """Get RabbitMQ connection"""
        return await connect_robust(
            host=settings.rabbit_host,
            port=settings.rabbit_port,
            login=settings.rabbit_user,
            password=settings.rabbit_pass,
            virtualhost=settings.rabbit_vhost,
        )

    async def get_channel():
        """Get RabbitMQ channel"""
        async with connection_pool.acquire() as connection:
            return await connection.channel()

    connection_pool = Pool(get_connection, max_size=settings.rabbit_pool_size)
    channel_pool = Pool(get_channel, max_size=settings.rabbit_channel_pool_size)

    logger.info("RabbitMQ channel pool created")
    return channel_pool


async def main():
    """Run the event consumer service."""
    # Configure logging
    logger.info("Starting Event Consumer Service")
    logger.info(f"Database URL: {settings.db_url}")
    logger.info(f"RabbitMQ URL: {settings.rabbit_url}")

    # Ensure SQLAlchemy logging is disabled again
    configure_sqlalchemy_logging()

    # Load all database models
    logger.info("Loading all database models...")
    load_all_models()

    # Create database engine with more robust configuration
    logger.info("Creating database engine...")
    engine = create_async_engine(
        str(settings.db_url),
        echo=False,  # Force disable SQL echo, ignore settings configuration
        # Increase connection pool size and timeout settings
        pool_size=20,
        max_overflow=30,
        pool_timeout=60,
        pool_recycle=1800,  # Reconnect every 30 minutes
        pool_pre_ping=True,  # Ping connection before use to ensure it's valid
    )

    # Test database connection before proceeding
    logger.info("Testing database connection...")
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection test successful!")
    except Exception as db_error:
        logger.error(f"Database connection test failed: {str(db_error)}")
        logger.critical("Cannot connect to database, service cannot start")
        return

    # Create session factory with proper settings
    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
        class_=AsyncSession,
    )

    # Test session creation
    logger.info("Testing database session creation...")
    try:
        async with session_factory() as test_session:
            # Execute simple query to confirm session is working properly
            result = await test_session.execute(text("SELECT NOW()"))
            timestamp = result.scalar_one()
            logger.info(
                f"Database session test successful! Current database time: {timestamp}"
            )
    except Exception as session_error:
        logger.error(f"Database session test failed: {str(session_error)}")
        logger.critical("Cannot create database session, service cannot start")
        return

    # Initialize AI services
    logger.info("Initializing AI engine...")
    try:
        llm_client = AsyncOpenAI(api_key=settings.openai_api_key)
        # Get QDrant client synchronously, without using await
        vector_db_client = get_qdrant_client()

        # Create AI engine
        ai_engine = AIEngine(
            vector_db_client=vector_db_client,
            llm_client=llm_client,
            embedding_model=settings.openai_embedding_model,
            llm_model=settings.openai_completion_model,
            enable_metrics=True,
        )
        logger.info("AI engine initialization successful!")
    except Exception as ai_error:
        logger.error(f"AI engine initialization failed: {str(ai_error)}")
        logger.warning(
            "Continuing to start service, but AI functionality may be unavailable"
        )
        ai_engine = None

    # Create RabbitMQ channel pool
    rmq_channel_pool = await create_rmq_channel_pool()

    # Create event consumer with initialized message brokers
    logger.info("Creating event consumer...")
    consumer = EventConsumer(
        db_session_factory=session_factory,
        ai_engine=ai_engine,
        rmq_channel_pool=rmq_channel_pool,
    )

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    signals = (signal.SIGINT, signal.SIGTERM)
    for sig in signals:
        loop.add_signal_handler(
            sig, lambda: asyncio.create_task(shutdown(consumer, engine))
        )

    # Start consumer
    logger.info("Starting event consumer service!")
    try:
        await consumer.start()
    except asyncio.CancelledError:
        logger.info("Consumer cancelled, shutting down...")
    except Exception as e:
        logger.error(f"Error running event consumer: {str(e)}")
        logger.exception("Detailed exception information:")
    finally:
        await shutdown(consumer, engine)

    logger.info("Event consumer service terminated")


async def shutdown(consumer, engine):
    """Gracefully shut down the service."""
    logger.info("Shutting down event consumer service...")

    # Stop the consumer
    await consumer.stop()

    # Close database connections
    await engine.dispose()

    # Exit the program
    asyncio.get_running_loop().stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        sys.exit(0)
