# ============================================================
# llm/clients.py — Groq API + Ollama LLM Clients
# ============================================================
#
# Groq  : Ultra-fast cloud inference (text) — free tier
# Ollama: Local model inference (text + vision) — fully private
#
# Routing logic:
#   If query contains images → Ollama (LLaVA vision model)
#   If text-only query       → Groq  (llama3-70b, fastest)
#   If Groq unavailable      → Ollama text model (fallback)
# ============================================================

import time
import json
from typing import Any, Dict, List, Optional

from utils.logger import get_logger, Timer

logger = get_logger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq not installed. Run: pip install groq")

try:
    import ollama as ollama_lib
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama not installed. Run: pip install ollama")


# ─────────────────────────────────────────────
# Groq Client (Fast Text Inference)
# ─────────────────────────────────────────────
class GroqClient:
    """
    Fast cloud LLM inference via Groq API.

    Free tier limits (generous):
      llama3-70b-8192 : 6,000 req/day, 500k tokens/min
      mixtral-8x7b    : 14,400 req/day

    Best for: text-only RAG generation, HyDE, multi-query
    """

    def __init__(
        self,
        api_key:    str,
        model:      str = "llama3-70b-8192",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        if not GROQ_AVAILABLE:
            raise ImportError("Install groq: pip install groq")
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")

        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.client      = Groq(api_key=api_key)

        logger.info(f"⚡ GroqClient initialized | model={model}")

    def generate(
        self,
        prompt:     str,
        max_tokens: Optional[int]   = None,
        temperature: Optional[float] = None,
        system:     str             = "",
    ) -> str:
        """
        Generate text response.

        Args:
            prompt:     User prompt
            max_tokens: Override default
            temperature: Override default
            system:     System message

        Returns:
            Generated text string
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        with Timer("groq_inference") as t:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                max_tokens  = max_tokens  or self.max_tokens,
                temperature = temperature or self.temperature,
            )

        text = response.choices[0].message.content.strip()
        logger.debug(
            f"⚡ Groq: {len(text)} chars in {t.elapsed:.2f}s "
            f"({response.usage.total_tokens} tokens)"
        )
        return text

    def generate_with_context(
        self,
        query:          str,
        context:        str,
        system_prompt:  str = "",
        max_tokens:     int = 1024,
    ) -> str:
        """Generate RAG answer from retrieved context."""
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant. Answer questions based ONLY "
                "on the provided context. If the answer is not in the context, "
                "say 'I don't have enough information to answer that.'"
            )

        prompt = f"""Context:
{context}

Question: {query}

Answer based solely on the context above:"""

        return self.generate(
            prompt      = prompt,
            system      = system_prompt,
            max_tokens  = max_tokens,
            temperature = 0.1,
        )

    def chat_completion(
        self,
        messages:    List[Dict[str, str]],
        max_tokens:  int   = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Raw chat completion with full message history."""
        response = self.client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Ollama Client (Local Vision + Text)
# ─────────────────────────────────────────────
class OllamaClient:
    """
    Local LLM inference via Ollama.

    Use for:
      - Vision queries (images/charts from PDFs)
      - Private data that must stay local
      - Offline operation

    Recommended models:
      Text  : llama3:8b, mistral:7b, phi3:mini
      Vision: llava:13b, bakllava:7b, llava-llama3:8b
    """

    def __init__(
        self,
        text_model:   str = "mistral:latest",
        vision_model: str = "llava:13b",
        base_url:     str = "http://localhost:11434",
        max_tokens:   int = 1024,
        temperature:  float = 0.1,
    ):
        if not OLLAMA_AVAILABLE:
            raise ImportError("Install ollama: pip install ollama")

        self.text_model   = text_model
        self.vision_model = vision_model
        self.base_url     = base_url
        self.max_tokens   = max_tokens
        self.temperature  = temperature

        # Verify Ollama is running
        self._check_connection()
        logger.info(f"🦙 OllamaClient initialized")
        logger.info(f"   Text model  : {text_model}")
        logger.info(f"   Vision model: {vision_model}")
        logger.info(f"   Base URL    : {base_url}")

    def _check_connection(self) -> bool:
        """Verify Ollama server is reachable."""
        try:
            import httpx
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                logger.info(f"✅ Ollama server reachable at {self.base_url}")
                return True
        except Exception:
            logger.warning(
                f"⚠️  Ollama server not reachable at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        return False

    def generate(
        self,
        prompt:      str,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
        system:      str             = "",
        images:      Optional[List[str]] = None,  # base64 or file paths
    ) -> str:
        """
        Generate text (or vision) response.

        Args:
            prompt:      User prompt
            images:      List of image paths for vision queries
            system:      System message

        Returns:
            Generated text string
        """
        use_vision = bool(images) and self.vision_model
        model      = self.vision_model if use_vision else self.text_model

        try:
            with Timer("ollama_inference") as t:
                msg: Dict[str, Any] = {
                    "model":  model,
                    "prompt": prompt,
                    "options": {
                        "num_predict": max_tokens  or self.max_tokens,
                        "temperature": temperature or self.temperature,
                        "num_ctx"    : 1024,  # Re-added to fit in limited sys memory 
                        "low_vram"   : True,
                    },
                    "stream": False,
                }

                if system:
                    msg["system"] = system

                if use_vision and images:
                    import base64
                    b64_images = []
                    for img_path in images:
                        with open(img_path, "rb") as f:
                            b64_images.append(
                                base64.b64encode(f.read()).decode()
                            )
                    msg["images"] = b64_images

                response = ollama_lib.generate(**msg)

            text = response["response"].strip()
            logger.debug(
                f"🦙 Ollama ({model}): {len(text)} chars in {t.elapsed:.2f}s"
            )
            return text

        except Exception as e:
            logger.error(f"❌ Ollama generation failed: {e}")
            return f"Error: {e}"

    def generate_with_context(
        self,
        query:    str,
        context:  str,
        images:   Optional[List[str]] = None,
        max_tokens: int = 1024,
    ) -> str:
        """Generate RAG answer, optionally with image context."""
        if images:
            prompt = f"""You are analyzing a document with images/charts.

Context from document:
{context}

Question: {query}

Analyze the image(s) and context to answer:"""
        else:
            prompt = f"""Context:
{context}

Question: {query}

Answer based solely on the context above:"""

        return self.generate(
            prompt     = prompt,
            images     = images,
            max_tokens = max_tokens,
        )

    def list_models(self) -> List[str]:
        """List available local Ollama models."""
        try:
            result = ollama_lib.list()
            return [m["name"] for m in result.get("models", [])]
        except Exception as e:
            logger.error(f"Cannot list Ollama models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull/download a model from Ollama hub."""
        logger.info(f"⬇️  Pulling model: {model_name}")
        try:
            ollama_lib.pull(model_name)
            logger.info(f"✅ Model pulled: {model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Model pull failed: {e}")
            return False


# ─────────────────────────────────────────────
# Unified LLM Router
# ─────────────────────────────────────────────
class LLMRouter:
    """
    Routes queries to the best available LLM:
      - Image queries  → Ollama (LLaVA)
      - Text queries   → Groq (fastest) → Ollama (fallback)
    """

    def __init__(
        self,
        groq_client:   Optional[GroqClient]   = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.groq   = groq_client
        self.ollama = ollama_client

        if not self.groq and not self.ollama:
            raise ValueError("At least one LLM client (Groq or Ollama) is required")

        logger.info("🔀 LLMRouter initialized")
        logger.info(f"   Groq   : {'✅ available' if self.groq else '❌ not configured'}")
        logger.info(f"   Ollama : {'✅ available' if self.ollama else '❌ not configured'}")

    def generate(
        self,
        prompt:     str,
        context:    str  = "",
        images:     Optional[List[str]] = None,
        max_tokens: int  = 1024,
        temperature: float = 0.1,
        system:     str  = "",
    ) -> Dict[str, Any]:
        """
        Route to best LLM and generate response.

        Returns dict with: answer, model_used, latency_s
        """
        start = time.time()
        has_images = bool(images)

        # Route: images → Ollama, text → Groq → Ollama fallback
        if has_images and self.ollama:
            model_used = f"ollama/{self.ollama.vision_model}"
            answer = self.ollama.generate_with_context(
                query      = prompt,
                context    = context,
                images     = images,
                max_tokens = max_tokens,
            )

        elif self.groq:
            model_used = f"groq/{self.groq.model}"
            try:
                answer = self.groq.generate_with_context(
                    query      = prompt,
                    context    = context,
                    system_prompt = system,
                    max_tokens = max_tokens,
                )
            except Exception as e:
                logger.warning(f"Groq failed, falling back to Ollama: {e}")
                if self.ollama:
                    model_used = f"ollama/{self.ollama.text_model}"
                    answer = self.ollama.generate_with_context(
                        query      = prompt,
                        context    = context,
                        max_tokens = max_tokens,
                    )
                else:
                    raise

        elif self.ollama:
            model_used = f"ollama/{self.ollama.text_model}"
            answer = self.ollama.generate_with_context(
                query      = prompt,
                context    = context,
                max_tokens = max_tokens,
            )
        else:
            raise RuntimeError("No LLM available")

        return {
            "answer":     answer,
            "model_used": model_used,
            "latency_s":  round(time.time() - start, 3),
            "has_images": has_images,
        }
