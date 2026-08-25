"""
chatbot_module.py

MODULE -- Chatbot Assistant (Ollama-backed)

Single Responsibility
----------------------
This module's ONLY job is: take a piece of text (typically a finalized
signed sentence from caption_generator.py, but also usable for manually
typed chat input) and get a conversational reply from a local LLM via
Ollama's REST API, while keeping a bounded conversation history.

It does NOT know about sign recognition, captions, or any GUI -- it
consumes plain text and produces plain text, exactly the same shape of
contract voice_assistant.py's TTS engine has (text in, something out),
which is what lets web_app.py wire "finalized caption" straight into
"chatbot input" without this module ever importing anything
recognition-related.

Design notes
------------
- STRATEGY PATTERN + NULL OBJECT, the same pattern used for every other
  pluggable backend in this project (TTSEngine/NullTTSEngine,
  SequenceClassifier/NullSequenceClassifier, AvatarRenderer/
  NullAvatarRenderer): `ChatbotEngine` is the abstract interface,
  `OllamaChatbotEngine` the real implementation, `NullChatbotEngine` a
  safe fallback used automatically if Ollama isn't installed/running --
  so the rest of the app (and web_app.py in particular) never crashes
  just because the local LLM server happens to be down; it gets a clear,
  friendly message back instead.
- STDLIB-ONLY HTTP: uses `urllib.request` (the same approach
  hand_detection.py already uses to download its model asset) rather
  than adding a `requests` dependency, since Ollama's REST API is plain
  JSON-over-HTTP and doesn't need anything heavier.
- CONVERSATION HISTORY: `ChatbotSession` keeps a bounded list of
  {role, content} turns (trimmed to `max_history_turns`) and prepends a
  configurable system prompt describing the sign-language context -- so
  the assistant has short-term memory of the conversation without
  unbounded growth, and replies appropriately to text that may read as
  simple/loose wording (an artifact of sign-to-text conversion).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChatMessage:
    """One turn in a chatbot conversation."""
    role: str  # "system" | "user" | "assistant"
    content: str
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000.0)


# ----------------------------------------------------------------------
# Pluggable chatbot engine (Strategy Pattern)
# ----------------------------------------------------------------------


class ChatbotEngine(ABC):
    """
    Abstract interface for any backend that can turn a message history
    into a reply. ChatbotSession depends only on this interface, never on
    Ollama specifically, so a different local or hosted LLM backend could
    be swapped in later without touching ChatbotSession at all.
    """

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Args:
            messages: The full conversation so far, oldest first, each
                item shaped {"role": "system"|"user"|"assistant",
                "content": str}.

        Returns:
            The assistant's reply text.

        Raises:
            RuntimeError: If the backend couldn't be reached or returned
                an unusable response.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap reachability check, used to decide whether to use this
        engine or fall back to NullChatbotEngine."""
        raise NotImplementedError


class OllamaChatbotEngine(ChatbotEngine):
    """
    ChatbotEngine implementation backed by a local Ollama server
    (https://ollama.com). Assumes Ollama is already installed, running
    (`ollama serve`, or the desktop app running in the background), and
    that the configured model has been pulled (e.g. `ollama pull
    llama3.2`).
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        """Pings Ollama's tag-listing endpoint (cheap, no model load) to
        check the server is up and reachable."""
        try:
            request = urllib.request.Request(f"{self._host}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=3.0) as response:
                return response.status == 200
        except Exception:
            return False

    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = json.dumps(
            {"model": self._model, "messages": messages, "stream": False}
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self._host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.exception("Ollama returned an HTTP error.")
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}. Is model '{self._model}' pulled? "
                f"(Run: ollama pull {self._model}). Response: {body[:200]}"
            ) from exc
        except urllib.error.URLError as exc:
            logger.exception("Failed to reach Ollama at %s", self._host)
            raise RuntimeError(
                f"Could not reach Ollama at {self._host}. Is it running? "
                f"(Start it with: ollama serve). Underlying error: {exc}"
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error calling Ollama.")
            raise RuntimeError(f"Unexpected error calling Ollama: {exc}") from exc

        content = raw.get("message", {}).get("content", "").strip()
        return content or "(Ollama returned an empty response.)"


class NullChatbotEngine(ChatbotEngine):
    """
    A safe, no-op chatbot engine used automatically when Ollama isn't
    reachable, so the rest of the app (and web_app.py's chat panel) keeps
    working and gives the person a clear, actionable message instead of
    crashing or hanging.
    """

    def __init__(self) -> None:
        logger.warning(
            "ChatbotSession is using NullChatbotEngine -- Ollama isn't "
            "reachable, so chat replies will be a placeholder message. "
            "Install Ollama (https://ollama.com), run `ollama serve`, and "
            "pull a model (e.g. `ollama pull llama3.2`) to enable real replies."
        )

    def is_available(self) -> bool:
        return True  # it's the safe fallback itself; always "available"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return (
            "(Chatbot unavailable: Ollama isn't running or isn't reachable. "
            "Start it with `ollama serve` and make sure a model is pulled, "
            "e.g. `ollama pull llama3.2`, then restart this app.)"
        )


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class ChatbotModuleConfig:
    """Tunable parameters for the chatbot session and its Ollama connection."""

    host: str = "http://localhost:11434"

    # Must match a model you've already pulled via `ollama pull <model>`.
    model: str = "llama3.2"

    system_prompt: str = (
        "You are a friendly, patient assistant chatting with someone who "
        "communicates via sign language. Their messages have been "
        "automatically converted from sign language to text, so wording "
        "may be simple, short, or grammatically loose -- interpret "
        "generously and don't comment on the phrasing. Keep replies "
        "concise and conversational."
    )

    # Number of user+assistant TURN PAIRS retained in history (beyond the
    # system prompt), trimmed oldest-first once exceeded.
    max_history_turns: int = 10

    timeout_seconds: float = 30.0


# ----------------------------------------------------------------------
# Chatbot session
# ----------------------------------------------------------------------


class ChatbotSession:
    """
    Holds a single ongoing conversation and mediates all calls to a
    ChatbotEngine.

    Usage:
        session = ChatbotSession()
        reply = session.send("hello how are you")
    """

    def __init__(
        self,
        engine: Optional[ChatbotEngine] = None,
        config: Optional[ChatbotModuleConfig] = None,
    ) -> None:
        """
        Args:
            engine: Optional ChatbotEngine. If omitted, attempts to
                connect to Ollama using `config`'s host/model and falls
                back to NullChatbotEngine if that fails for any reason
                (not installed, not running, model not pulled, etc.) --
                the chatbot being unavailable should never crash the app.
            config: Optional ChatbotModuleConfig; defaults used if omitted.
        """
        self._config = config or ChatbotModuleConfig()
        self._engine = engine or self._create_default_engine()
        self._history: List[ChatMessage] = []

    def _create_default_engine(self) -> ChatbotEngine:
        candidate = OllamaChatbotEngine(
            host=self._config.host,
            model=self._config.model,
            timeout_seconds=self._config.timeout_seconds,
        )
        if candidate.is_available():
            logger.info(
                "ChatbotSession connected to Ollama at %s (model=%s).",
                self._config.host,
                self._config.model,
            )
            return candidate

        logger.warning(
            "Ollama not reachable at %s; falling back to NullChatbotEngine.",
            self._config.host,
        )
        return NullChatbotEngine()

    def send(self, user_text: str) -> str:
        """
        Send a message and get a reply, updating conversation history.

        Args:
            user_text: The message to send (e.g. a finalized signed
                sentence, or manually typed chat text).

        Returns:
            The assistant's reply text. Never raises -- backend failures
            are caught and turned into a friendly fallback reply, since a
            chat failure should never take down whatever called this
            (e.g. web_app.py's request handler).
        """
        if not user_text or not user_text.strip():
            return ""

        self._history.append(ChatMessage(role="user", content=user_text.strip()))
        self._trim_history()

        messages = [{"role": "system", "content": self._config.system_prompt}]
        messages.extend({"role": m.role, "content": m.content} for m in self._history)

        try:
            reply_text = self._engine.chat(messages)
        except Exception:
            logger.exception("ChatbotSession failed to get a reply; using fallback message.")
            reply_text = "(Sorry, I couldn't reach the chatbot right now.)"

        self._history.append(ChatMessage(role="assistant", content=reply_text))
        self._trim_history()
        return reply_text

    def _trim_history(self) -> None:
        """Keep only the most recent max_history_turns*2 messages
        (user+assistant pairs), dropping the oldest first."""
        max_messages = self._config.max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def get_history(self) -> List[ChatMessage]:
        """Return the full retained conversation history, oldest first."""
        return list(self._history)

    def reset(self) -> None:
        """Clear conversation history, starting a fresh conversation."""
        self._history = []
        logger.info("ChatbotSession history cleared.")


if __name__ == "__main__":
    # Minimal manual smoke-test. Run via: `python chatbot_module.py`.
    # Works with or without Ollama actually running -- if it's not
    # reachable, this exercises the NullChatbotEngine fallback path
    # instead, and the script still completes successfully.
    logger.info("Running chatbot_module.py standalone demo.")

    session = ChatbotSession()
    reply = session.send("hello how are you")
    print("User: hello how are you")
    print("Assistant:", reply)

    reply2 = session.send("what is sign language")
    print("\nUser: what is sign language")
    print("Assistant:", reply2)

    print("\nHistory length:", len(session.get_history()))
    logger.info("Demo complete.")
