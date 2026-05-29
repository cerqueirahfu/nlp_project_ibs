"""Thin wrapper around AWS Bedrock for the PricePulse AI features.

Used by the Streamlit dashboard for (1) on-demand insight summaries and
(2) the dataset chat page. All calls go through `complete()`.

Design goals:
- **No AWS keys in code.** On EC2 the credentials come from the instance's
  IAM role (see EC2_DEPLOYMENT_GUIDE.md step C). Locally they come from
  `aws configure` if present.
- **Degrade gracefully.** If boto3 is missing or no credentials/permission are
  available, `available()` returns False and the UI hides/disables the AI bits
  instead of crashing. `complete()` raises `BedrockError` with a readable
  message that the UI surfaces.

Config via environment variables (all optional, sensible defaults):
    BEDROCK_MODEL_ID   default: anthropic.claude-haiku-4-5-20251001-v1:0
    BEDROCK_REGION     default: us-east-1  (falls back to AWS_REGION)
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

DEFAULT_MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION = "us-east-1"
ANTHROPIC_VERSION = "bedrock-2023-05-31"


class BedrockError(RuntimeError):
    """Raised when a Bedrock call fails for a reason worth showing the user."""


def model_id() -> str:
    return os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)


def region() -> str:
    return os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION") or DEFAULT_REGION


@lru_cache(maxsize=1)
def _client():
    """Build a cached bedrock-runtime client, or raise BedrockError.

    boto3 resolves credentials from the standard chain: env vars, shared
    config (`aws configure`), or the EC2 instance IAM role.
    """
    try:
        import boto3  # imported lazily so the app runs without boto3 installed
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise BedrockError(
            "boto3 is not installed. Run `pip install boto3` to enable AI features."
        ) from exc
    return boto3.client("bedrock-runtime", region_name=region())


@lru_cache(maxsize=1)
def available() -> bool:
    """True if AI calls are likely to work (boto3 importable + creds resolvable).

    Cheap check — does not make a network call. A True result can still fail
    later with a permission/region error, which `complete()` reports cleanly.
    """
    try:
        import boto3
        from botocore.exceptions import BotoCoreError
    except ImportError:
        return False
    try:
        session = boto3.session.Session()
        return session.get_credentials() is not None
    except BotoCoreError:
        return False


def complete(
    messages: list[dict],
    *,
    system: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """Send a chat-style request to Claude on Bedrock and return the text.

    `messages` is the Anthropic format: [{"role": "user"|"assistant",
    "content": "..."}]. `max_tokens` is always capped to keep cost bounded.
    """
    body: dict = {
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        body["system"] = system

    try:
        from botocore.exceptions import ClientError, BotoCoreError
    except ImportError as exc:  # pragma: no cover
        raise BedrockError("boto3/botocore not installed.") from exc

    client = _client()
    try:
        resp = client.invoke_model(modelId=model_id(), body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        return payload["content"][0]["text"].strip()
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "AccessDeniedException":
            raise BedrockError(
                "Access denied. The IAM role/credentials lack `bedrock:InvokeModel` "
                "(see EC2_DEPLOYMENT_GUIDE.md step C)."
            ) from exc
        if code in ("ValidationException", "ResourceNotFoundException"):
            raise BedrockError(
                f"Model `{model_id()}` not available in region `{region()}`. "
                "Check the exact model ID in the Bedrock console / your region "
                "(EC2_DEPLOYMENT_GUIDE.md step B)."
            ) from exc
        raise BedrockError(f"Bedrock call failed ({code or 'unknown error'}).") from exc
    except BotoCoreError as exc:
        raise BedrockError(f"Could not reach Bedrock: {exc}") from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise BedrockError("Unexpected response shape from Bedrock.") from exc
