"""Slack client for sending notifications."""

import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp
from pydantic import ValidationError

from system_guardian.services.slack.templates import SlackMessageTemplate
from system_guardian.settings import settings

logger = logging.getLogger(__name__)


class SlackClient:
    """Slack client for sending notifications."""

    def __init__(
        self,
        token: Optional[str] = None,
        default_channel: Optional[str] = None,
        username: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Initialize Slack client.

        :param token: Slack bot token
        :param default_channel: Default channel to send messages to
        :param username: Username to use for messages
        :param icon_emoji: Emoji to use as icon for messages
        :param timeout: Timeout for requests in seconds
        """
        self.token = token or settings.slack_bot_token
        self.default_channel = default_channel or settings.slack_channel_id
        self.username = username or settings.slack_username
        self.icon_emoji = icon_emoji or settings.slack_icon_emoji
        self.timeout = timeout or settings.slack_timeout

        if not self.token:
            logger.warning("Slack bot token not configured, messages will not be sent")

        self._api_url = "https://slack.com/api"
        self._enabled = settings.slack_enabled

    async def send_message(
        self,
        text: str,
        channel: Optional[str] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Send a message to Slack.

        :param text: Message text
        :param channel: Channel to send message to, defaults to default_channel
        :param blocks: Slack blocks for advanced formatting
        :param attachments: Slack attachments
        :return: Slack API response
        """
        if not self._enabled:
            logger.info("Slack notifications disabled, message not sent: %s", text[:50])
            return {"ok": False, "error": "slack_disabled"}

        if not self.token:
            logger.error("Cannot send Slack message: bot token not configured")
            return {"ok": False, "error": "token_not_configured"}

        target_channel = channel or self.default_channel
        if not target_channel:
            logger.error("Cannot send Slack message: channel not specified")
            return {"ok": False, "error": "channel_not_specified"}

        payload = {
            "channel": target_channel,
            "text": text,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
        }

        if blocks:
            payload["blocks"] = blocks

        if attachments:
            payload["attachments"] = attachments

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.token}",
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self._api_url}/chat.postMessage",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                ) as response:
                    result = await response.json()
                    if not result.get("ok", False):
                        logger.error(
                            "Failed to send Slack message: %s",
                            result.get("error", "unknown_error"),
                        )
                    else:
                        logger.debug("Slack message sent successfully")
                    return result
            except aiohttp.ClientError as e:
                logger.exception("Error sending Slack message: %s", str(e))
                return {"ok": False, "error": str(e)}

    async def send_template(
        self,
        template: Union[SlackMessageTemplate, Dict[str, Any]],
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a message using a template.

        :param template: SlackMessageTemplate instance or dict with template data
        :param channel: Channel to send message to, defaults to default_channel
        :return: Slack API response
        """
        if isinstance(template, dict):
            try:
                template = SlackMessageTemplate.model_validate(template)
            except ValidationError as e:
                logger.error("Invalid template data: %s", str(e))
                return {"ok": False, "error": f"invalid_template: {str(e)}"}

        return await self.send_message(
            text=template.fallback_text,
            channel=channel,
            blocks=template.blocks,
            attachments=template.attachments,
        )

    @property
    def is_configured(self) -> bool:
        """Check if Slack client is properly configured."""
        return bool(self.token and self._enabled)
