"""Slack service for sending notifications."""

from system_guardian.services.slack.client import SlackClient
from system_guardian.services.slack.templates import SlackMessageTemplate

__all__ = ["SlackClient", "SlackMessageTemplate"]
