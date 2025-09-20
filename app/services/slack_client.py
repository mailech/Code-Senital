from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from app.config import settings
from app.logging_setup import logger


def notify(channel: str | None, text: str) -> None:
    if not settings.slack_token:
        logger.info("slack_dry_run", channel=channel or settings.slack_channel, text=text)
        return
    client = WebClient(token=settings.slack_token)
    try:
        client.chat_postMessage(channel=(channel or settings.slack_channel), text=text)
    except SlackApiError as e:
        logger.error("slack_error", error=str(e))
