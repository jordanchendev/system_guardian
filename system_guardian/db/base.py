from sqlalchemy.orm import DeclarativeBase

from system_guardian.db.meta import meta


class Base(DeclarativeBase):
    """Base for all models."""

    metadata = meta
