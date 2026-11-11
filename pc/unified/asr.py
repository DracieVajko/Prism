"""Registry for the optional, non-Whisper ASR engines.

Nemotron and Sherpa-ONNX used to be dispatched by an if/elif ladder in the HTTP
layer, and each module repeated the same install/init/transcribe skeleton. This
module gives them one small contract: resolve a request model name once, then
call ``installed()`` / ``transcribe()`` without knowing which module is behind
it.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple


class AsrEngineModule(Protocol):
    """The part of an engine module this registry depends on."""

    MODEL_ID: str

    def enabled(self) -> bool: ...

    def is_installed(self, model_id: Optional[str] = None) -> bool: ...

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Tuple[str, str]: ...


@dataclass(frozen=True)
class AsrEngine:
    """A non-Whisper engine that can be selected by model id or environment."""

    name: str
    prefix: str
    install_hint: str
    module: AsrEngineModule

    @property
    def default_id(self) -> str:
        return self.module.MODEL_ID

    def installed(self, model_id: Optional[str] = None) -> bool:
        return self.module.is_installed(model_id or self.default_id)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        return self.module.transcribe(
            audio_path, language, model_id or self.default_id)


_ENGINES: Optional[Tuple[AsrEngine, ...]] = None


def engines() -> Tuple[AsrEngine, ...]:
    """Build the registry once; engine modules are only imported on first use."""
    global _ENGINES
    if _ENGINES is None:
        from unified import nemotron_asr, sherpa_asr

        _ENGINES = (
            AsrEngine(
                name="nemotron-asr",
                prefix="nemotron",
                install_hint="python pc/tools/download_nemotron_asr.py",
                module=nemotron_asr,
            ),
            AsrEngine(
                name="sherpa-onnx",
                prefix="sherpa",
                install_hint="python pc/tools/download_sherpa_asr.py",
                module=sherpa_asr,
            ),
        )
    return _ENGINES


def resolve_engine(raw_model: str) -> Tuple[Optional[AsrEngine], str]:
    """Resolve a request model name to ``(engine or None, model id)``.

    * ``nemotron-*`` / ``sherpa-*`` select that engine explicitly, so the
      caller must not silently fall back to Whisper when it is missing.
    * the generic ``whisper-1`` alias follows ``PRISM_ASR_ENGINE`` when one of
      the optional engines is enabled, otherwise it means Whisper ``turbo``.
    * any other name is returned unchanged for the Whisper allow-list check.
    """
    model = (raw_model or "").strip() or "whisper-1"
    for engine in engines():
        if model.startswith(engine.prefix):
            return engine, model
    if model == "whisper-1":
        for engine in engines():
            if engine.module.enabled():
                return engine, engine.default_id
        return None, "turbo"
    return None, model
