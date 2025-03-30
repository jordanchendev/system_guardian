from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Enum,
    Boolean,
    ForeignKey,
    Float,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from system_guardian.db.base import Base


class Incident(Base):
    """Incident model representing a system issue."""

    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    severity = Column(Enum("low", "medium", "high", "critical", name="severity_enum"))
    status = Column(
        Enum("open", "investigating", "resolving", "resolved", name="status_enum")
    )
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    source = Column(String)  # 'github', 'jira', 'slack', etc.

    # Reference to the event that triggered this incident
    trigger_event_id = Column(Integer, ForeignKey("events.id"), nullable=True)

    # Relationships
    # 關聯到此意外的所有事件
    events = relationship(
        "Event",
        back_populates="incident",
        foreign_keys="Event.related_incident_id",
        overlaps="trigger_event,trigger_for",
    )

    resolutions = relationship("Resolution", back_populates="incident")

    # 觸發此意外的事件
    trigger_event = relationship(
        "Event",
        back_populates="trigger_for",
        foreign_keys=[trigger_event_id],
        primaryjoin="Incident.trigger_event_id == Event.id",
        uselist=False,  # 一對一關係
        overlaps="events,incident",
        post_update=False,
    )


class Event(Base):
    """Events related to an incident from various sources."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True)
    related_incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=True)
    source = Column(String)  # 'github', 'jira', 'slack', etc.
    event_type = Column(String)  # 'commit', 'issue', 'message', etc.
    content = Column(JSONB)  # Store the full event payload
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    # 事件相關的意外 (多對一)
    incident = relationship(
        "Incident",
        back_populates="events",
        foreign_keys=[related_incident_id],
        primaryjoin="Event.related_incident_id == Incident.id",
    )

    # 由此事件觸發的意外 (一對一，可選)
    trigger_for = relationship(
        "Incident",
        back_populates="trigger_event",
        foreign_keys="Incident.trigger_event_id",
        primaryjoin="Event.id == Incident.trigger_event_id",
        uselist=False,
    )


class Resolution(Base):
    """AI-generated or manual resolution for an incident."""

    __tablename__ = "resolutions"

    id = Column(Integer, primary_key=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"))
    suggestion = Column(Text)
    confidence = Column(Float)  # AI confidence score
    is_applied = Column(Boolean, default=False)
    generated_at = Column(DateTime, default=datetime.utcnow)
    feedback_score = Column(Integer, nullable=True)  # User feedback

    # Relationships
    incident = relationship("Incident", back_populates="resolutions")
