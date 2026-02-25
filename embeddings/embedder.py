# ============================================================
# embeddings/embedder.py — HuggingFace Multimodal Embeddings
# ============================================================
#
# Text Embeddings  : BAAI/bge-base-en-v1.5 (via sentence-transformers)
# Image Embeddings : openai/clip-vit-base-patch32 (via open_clip / HF)
# Unified space    : Both stored in ChromaDB with type metadata
# ============================================================

import os
import io
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

from utils.logger import get_logger, Timer

logger = get_logger(__name__)

# ── Sentence Transformers ────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")

# ── CLIP (image embeddings) ──────────────────────────────────
try:
    import open_clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("open_clip not installed. Image embeddings disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ─────────────────────────────────────────────
# Text Embedder
# ─────────────────────────────────────────────
class HuggingFaceTextEmbedder:
    """
    Generates dense text embeddings using HuggingFace sentence-transformers.

    Default model: BAAI/bge-base-en-v1.5
      - 768 dimensions
      - Best-in-class for RAG retrieval
      - Supports long documents (512 tokens)

    Alternative: sentence-transformers/all-MiniLM-L6-v2
      - 384 dimensions
      - Faster, smaller
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device:     str = "cpu",
        batch_size: int = 32,
        normalize:  bool = True,
    ):
        self.model_name = model_name
        self.device     = device
        self.batch_size = batch_size
        self.normalize  = normalize
        self.model      = None

        if not ST_AVAILABLE:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        logger.info(f"🔤 Loading text embedding model: {model_name}")
        with Timer("model_load") as t:
            self.model = SentenceTransformer(model_name, device=device)

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"✅ Text embedder ready | dim={self.embedding_dim} | {t.elapsed:.1f}s")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            numpy array shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        # BGE models benefit from instruction prefix for queries
        if "bge" in self.model_name.lower():
            # Add retrieval instruction for BGE models
            texts = [
                f"Represent this sentence for searching relevant passages: {t}"
                if len(t) < 512 else t
                for t in texts
            ]

        logger.debug(f"🔤 Embedding {len(texts)} texts (batch_size={self.batch_size})")

        with Timer("embedding") as t:
            embeddings = self.model.encode(
                texts,
                batch_size          = self.batch_size,
                show_progress_bar   = len(texts) > 50,
                normalize_embeddings = self.normalize,
                convert_to_numpy    = True,
            )

        logger.debug(f"✅ Embedded {len(texts)} texts in {t.elapsed:.2f}s")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        For BGE models uses query-specific instruction prefix.
        """
        if "bge" in self.model_name.lower():
            query = f"Represent this question for searching relevant passages: {query}"

        with Timer("query_embedding") as t:
            embedding = self.model.encode(
                query,
                normalize_embeddings = self.normalize,
                convert_to_numpy     = True,
            )

        logger.debug(f"✅ Query embedded in {t.elapsed:.3f}s")
        return embedding

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text (not a query)."""
        return self.embed_texts([text])[0]


# ─────────────────────────────────────────────
# Image / CLIP Embedder
# ─────────────────────────────────────────────
class CLIPImageEmbedder:
    """
    Generates image embeddings using OpenAI CLIP via open_clip.

    These embeddings are in the SAME SPACE as CLIP text embeddings,
    enabling cross-modal retrieval (text query → find images).

    Model: openai/clip-vit-base-patch32
      - 512 dimensions
      - Supports text + image in unified space
    """

    def __init__(
        self,
        model_name:    str = "ViT-B-32",
        pretrained:    str = "openai",
        device:        str = "cpu",
    ):
        if not CLIP_AVAILABLE:
            raise ImportError("Install open_clip: pip install open-clip-torch")
        if not PIL_AVAILABLE:
            raise ImportError("Install Pillow: pip install Pillow")

        self.device = device
        logger.info(f"🖼️  Loading CLIP model: {model_name} ({pretrained})")

        with Timer("clip_load") as t:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.model = self.model.to(device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(model_name)

        self.embedding_dim = 512
        logger.info(f"✅ CLIP embedder ready | dim={self.embedding_dim} | {t.elapsed:.1f}s")

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """
        Embed a list of image file paths.

        Args:
            image_paths: List of image file paths

        Returns:
            numpy array shape (n_images, 512)
        """
        import torch

        embeddings = []
        for path in image_paths:
            try:
                image  = Image.open(path).convert("RGB")
                tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.model.encode_image(tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
                embeddings.append(emb.cpu().numpy()[0])
            except Exception as e:
                logger.warning(f"Image embed failed for {path}: {e}")
                embeddings.append(np.zeros(self.embedding_dim))

        return np.array(embeddings) if embeddings else np.array([])

    def embed_text_for_image_search(self, text: str) -> np.ndarray:
        """
        Embed text in CLIP space for cross-modal (text → image) search.
        """
        import torch

        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def embed_pil_image(self, pil_image) -> np.ndarray:
        """Embed a PIL Image object directly."""
        import torch

        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]


# ─────────────────────────────────────────────
# Unified Multimodal Embedder
# ─────────────────────────────────────────────
class MultimodalEmbedder:
    """
    Unified embedder that handles both text and images.
    Routes to appropriate model based on input type.

    For RAG:
        - Text chunks   → HuggingFace BGE embeddings (768d)
        - Image chunks  → CLIP embeddings (512d)
    """

    def __init__(
        self,
        text_model:  str  = "BAAI/bge-base-en-v1.5",
        clip_model:  str  = "ViT-B-32",
        device:      str  = "cpu",
        enable_clip: bool = True,
    ):
        # Text embedder (always required)
        self.text_embedder = HuggingFaceTextEmbedder(
            model_name = text_model,
            device     = device,
        )

        # CLIP embedder (optional, for images)
        self.clip_embedder = None
        if enable_clip and CLIP_AVAILABLE:
            try:
                self.clip_embedder = CLIPImageEmbedder(
                    model_name = clip_model,
                    device     = device,
                )
                logger.info("✅ Multimodal embedder ready (text + image)")
            except Exception as e:
                logger.warning(f"CLIP initialization failed: {e}. Image search disabled.")
        else:
            logger.info("✅ Text-only embedder ready")

    def embed_chunks(self, chunks) -> List[np.ndarray]:
        """Embed a list of DocumentChunk objects."""
        texts = [c.content for c in chunks]
        return self.text_embedder.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a user query for retrieval."""
        return self.text_embedder.embed_query(query)

    def embed_image_for_search(self, query: str) -> Optional[np.ndarray]:
        """Embed text for cross-modal image search (requires CLIP)."""
        if self.clip_embedder:
            return self.clip_embedder.embed_text_for_image_search(query)
        return None

    @property
    def text_dim(self) -> int:
        return self.text_embedder.embedding_dim

    @property
    def image_dim(self) -> int:
        return self.clip_embedder.embedding_dim if self.clip_embedder else 0
