from collections.abc import Sequence
from enum import IntEnum
import logging
from typing import Any
import softmaxx.util as smaxx_util


class Trace:
    """
    logger wrapper that translates and
    promotes trace logs to INFO when enabled.
    """

    class Type(IntEnum):
        NONE = 0
        MAP = 1
        MAPS = 2
        RECORD = 3
        RECORDS = 4
        LIST = 5
        DOC_CHUNKS = 100

    __slots__ = ("_logger", "_enabled", "_translator")

    def __init__(
        self,
        logger: logging.Logger,
        enabled: bool,
        translator: "type[Translator] | None" = None,
    ):
        self._logger = logger
        self._enabled = bool(enabled)
        self._translator = translator or Translator

    def log(
        self,
        msg: str,
        trace_type: "Trace.Type" = Type.NONE,
        payload: Any = None,
        **kwargs: Any,
    ) -> None:
        if not self._enabled:
            return

        if trace_type == Trace.Type.NONE or payload is None:
            self._logger.info("[TRACE] %s", msg, stacklevel=2)
            return

        formatted = self._translator.translate(trace_type, payload, **kwargs)
        self._logger.info("[TRACE] %s: %s", msg, formatted, stacklevel=2)


class Translator:
    """Translates Trace payloads into JSON strings using softmaxx.util."""

    @staticmethod
    def translate(trace_type: Trace.Type, payload: Any, **kwargs: Any) -> str:
        keys: Sequence[str] = kwargs.get("keys", ())

        match trace_type:
            case Trace.Type.MAP:
                return smaxx_util.format_map(payload, keys=keys)

            case Trace.Type.MAPS:
                return smaxx_util.format_maps(payload, keys=keys)

            case Trace.Type.RECORD:
                return smaxx_util.format_record(payload, keys=keys)

            case Trace.Type.RECORDS:
                return smaxx_util.format_records(payload, keys=keys)

            case Trace.Type.LIST:
                if keys:
                    raise TypeError("keys filtering not supported for Trace.Type.LIST")
                return smaxx_util.format_list(payload)

            case _:
                return str(payload)
