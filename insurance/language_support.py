"""
language_support.py — Multilingual Support (Feature 4)
========================================================
Wraps LLM response generation with a language instruction prefix.

Supported languages: Hindi, Tamil, Telugu, Bengali, Marathi,
                     Kannada, Malayalam, Gujarati, Punjabi, English

Usage:
    ls = LanguageSupport(llm_client)
    answer = ls.generate(prompt, language="hindi")
"""

from typing import Dict, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Supported language map ────────────────────────────────────────────────
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "english":   "English",
    "hindi":     "Hindi (हिन्दी)",
    "tamil":     "Tamil (தமிழ்)",
    "telugu":    "Telugu (తెలుగు)",
    "bengali":   "Bengali (বাংলা)",
    "marathi":   "Marathi (मराठी)",
    "kannada":   "Kannada (ಕನ್ನಡ)",
    "malayalam": "Malayalam (മലയാളം)",
    "gujarati":  "Gujarati (ગુજરાતી)",
    "punjabi":   "Punjabi (ਪੰਜਾਬੀ)",
    "urdu":      "Urdu (اردو)",
    "odia":      "Odia (ଓଡ଼ିଆ)",
}

# Language-specific instruction prefix
_LANG_PREFIX_TEMPLATE = (
    "IMPORTANT: You must respond ONLY in {language}. "
    "Use simple, clear language that is easy to understand. "
    "Do not use English unless the user's original query was in English.\n\n"
)


class LanguageSupport:
    """
    Multilingual wrapper for LLM calls.

    Prepends a language instruction to any prompt so the LLM
    responds in the user's preferred language.

    Args:
        llm_client: Groq or Ollama client instance
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Core multilingual generate
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt:     str,
        language:   str = "english",
        max_tokens: int = 1024,
        **kwargs,
    ) -> Dict:
        """
        Generate an LLM response in the specified language.

        Args:
            prompt    : The full prompt to send
            language  : Target language key (see SUPPORTED_LANGUAGES)
            max_tokens: Maximum tokens to generate

        Returns:
            LLM result dict with keys: answer, model_used, tokens_used
        """
        lang_key = language.lower().strip()

        if lang_key not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"Language '{language}' not in supported list. Defaulting to English."
            )
            lang_key = "english"

        lang_display = SUPPORTED_LANGUAGES[lang_key]

        if lang_key == "english":
            # No prefix needed for English
            final_prompt = prompt
        else:
            prefix = _LANG_PREFIX_TEMPLATE.format(language=lang_display)
            final_prompt = prefix + prompt

        logger.debug(f"Generating response in: {lang_display}")

        result = self.llm_client.generate(
            prompt     = final_prompt,
            max_tokens = max_tokens,
            **kwargs,
        )

        result["language"] = lang_display
        return result

    # ------------------------------------------------------------------
    # Translate an existing answer
    # ------------------------------------------------------------------
    def translate(
        self,
        text:       str,
        target_lang: str = "hindi",
        max_tokens: int  = 1024,
    ) -> str:
        """
        Translate an existing English answer into the target language.

        Useful when you want to generate in English first (for accuracy)
        then translate for display.

        Args:
            text        : Text to translate
            target_lang : Target language key
            max_tokens  : Maximum tokens

        Returns:
            Translated text string
        """
        lang_key     = target_lang.lower().strip()
        lang_display = SUPPORTED_LANGUAGES.get(lang_key, "Hindi (हिन्दी)")

        if lang_key == "english":
            return text   # nothing to do

        translate_prompt = (
            f"Translate the following insurance-related text accurately into "
            f"{lang_display}. Keep technical terms (like policy numbers, "
            f"medical terms, ICD codes) in their original form. "
            f"Use simple, clear language.\n\n"
            f"Text to translate:\n{text}\n\n"
            f"Translation in {lang_display}:"
        )

        result = self.llm_client.generate(
            prompt     = translate_prompt,
            max_tokens = max_tokens,
        )
        translated = result.get("answer", text).strip()
        logger.debug(f"Translated to {lang_display}: {translated[:80]}…")
        return translated

    # ------------------------------------------------------------------
    # Detect language of input text
    # ------------------------------------------------------------------
    def detect_language(self, text: str) -> str:
        """
        Ask the LLM to detect the language of the input text.
        Returns a language key from SUPPORTED_LANGUAGES, defaulting to 'english'.

        Args:
            text: User's input text

        Returns:
            Language key string (e.g. 'hindi', 'tamil', 'english')
        """
        detect_prompt = (
            f"Detect the language of the following text. "
            f"Reply with ONLY ONE of these exact keys:\n"
            f"{', '.join(SUPPORTED_LANGUAGES.keys())}\n\n"
            f"Text: {text[:200]}\n\nLanguage key:"
        )
        try:
            result = self.llm_client.generate(detect_prompt, max_tokens=10)
            detected = result.get("answer", "english").strip().lower()
            if detected in SUPPORTED_LANGUAGES:
                logger.debug(f"Detected language: {detected}")
                return detected
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")

        return "english"

    @staticmethod
    def list_languages() -> str:
        """Return a formatted list of all supported languages."""
        lines = ["Supported Languages:"]
        for key, display in SUPPORTED_LANGUAGES.items():
            lines.append(f"  • {key:<12} → {display}")
        return "\n".join(lines)
