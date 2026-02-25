# ============================================================
# ingest/pdf_loader.py — Enterprise PDF Document Ingestion
# ============================================================
#
# Handles:
#   • Text extraction with fallback methods
#   • Image & chart extraction from PDFs
#   • OCR for scanned/image-based PDFs
#   • Metadata enrichment
#   • Quality assessment & validation
# ============================================================

import os
import io
import re
import uuid
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# PDF libraries
try:
    import fitz                          # PyMuPDF — best for image extraction
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber                    # Good for tables
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from utils.logger import get_logger, Timer, validate_pdf_path, truncate_text

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────
@dataclass
class PageContent:
    """Represents content extracted from a single PDF page."""
    page_number:    int
    raw_text:       str
    cleaned_text:   str
    char_count:     int
    word_count:     int
    quality_score:  float
    has_images:     bool        = False
    has_tables:     bool        = False
    image_paths:    List[str]   = field(default_factory=list)
    table_data:     List[Any]   = field(default_factory=list)
    error:          Optional[str] = None


@dataclass
class DocumentMetadata:
    """Comprehensive metadata for a processed document."""
    document_id:        str     = field(default_factory=lambda: str(uuid.uuid4()))
    filename:           str     = ""
    filepath:           str     = ""
    total_pages:        int     = 0
    total_characters:   int     = 0
    total_words:        int     = 0
    total_images:       int     = 0
    processing_timestamp: str   = field(default_factory=lambda: datetime.now().isoformat())
    language:           str     = "en"
    quality_score:      float   = 0.0
    extraction_method:  str     = "unknown"
    file_size_bytes:    int     = 0
    has_scanned_content: bool   = False


@dataclass
class ProcessedDocument:
    """Complete processed document with all content and metadata."""
    content:        str                     # Full concatenated text
    metadata:       DocumentMetadata
    pages:          List[PageContent]       = field(default_factory=list)
    images:         List[Dict[str, Any]]    = field(default_factory=list)


# ─────────────────────────────────────────────
# Document Processor
# ─────────────────────────────────────────────
class EnterpriseDocumentProcessor:
    """
    Production-grade PDF processor with:
      - Multi-method text extraction (PyMuPDF → pdfplumber → PyPDF2)
      - Automatic fallback chain
      - Image & chart extraction
      - OCR for scanned content
      - Text quality scoring
      - Full metadata enrichment
    """

    def __init__(
        self,
        min_chars_per_page: int     = 50,
        extract_images: bool        = True,
        ocr_fallback: bool          = True,
        image_output_dir: str       = "./extracted_images",
    ):
        self.min_chars_per_page = min_chars_per_page
        self.extract_images     = extract_images
        self.ocr_fallback       = ocr_fallback
        self.image_output_dir   = Path(image_output_dir)

        if extract_images:
            self.image_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("📄 EnterpriseDocumentProcessor initialized")
        logger.info(f"   PyMuPDF    : {'✅' if PYMUPDF_AVAILABLE else '❌ not installed'}")
        logger.info(f"   pdfplumber : {'✅' if PDFPLUMBER_AVAILABLE else '❌ not installed'}")
        logger.info(f"   PyPDF2     : {'✅' if PYPDF2_AVAILABLE else '❌ not installed'}")
        logger.info(f"   OCR        : {'✅' if OCR_AVAILABLE else '❌ not installed'}")

    # ── Public API ──────────────────────────────────────────

    def load_pdf(self, pdf_path: str) -> ProcessedDocument:
        """
        Main entry: load and process a PDF file.

        Args:
            pdf_path: Absolute or relative path to PDF

        Returns:
            ProcessedDocument with all extracted content

        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If PDF cannot be processed
        """
        if not validate_pdf_path(pdf_path):
            raise FileNotFoundError(f"PDF not found or invalid: {pdf_path}")

        logger.info(f"📂 Loading PDF: {pdf_path}")

        with Timer("pdf_loading") as t:
            metadata = DocumentMetadata(
                filename        = Path(pdf_path).name,
                filepath        = str(Path(pdf_path).resolve()),
                file_size_bytes = os.path.getsize(pdf_path),
            )

            # Try extraction methods in priority order
            pages, method = self._extract_with_fallback(pdf_path, metadata)

            # Build full text
            full_text = "\n\n".join(
                p.cleaned_text for p in pages if p.cleaned_text.strip()
            )

            # Finalize metadata
            metadata.total_pages        = len(pages)
            metadata.total_characters   = len(full_text)
            metadata.total_words        = len(full_text.split())
            metadata.total_images       = sum(len(p.image_paths) for p in pages)
            metadata.extraction_method  = method
            metadata.quality_score      = self._calculate_document_quality(pages)
            metadata.has_scanned_content = any(
                p.quality_score < 0.3 for p in pages
            )

        logger.info(f"✅ PDF processed in {t.elapsed:.2f}s")
        logger.info(f"   Pages     : {metadata.total_pages}")
        logger.info(f"   Characters: {metadata.total_characters:,}")
        logger.info(f"   Words     : {metadata.total_words:,}")
        logger.info(f"   Images    : {metadata.total_images}")
        logger.info(f"   Quality   : {metadata.quality_score:.2f}")
        logger.info(f"   Method    : {metadata.extraction_method}")

        return ProcessedDocument(
            content  = full_text.strip(),
            metadata = metadata,
            pages    = pages,
        )

    # ── Extraction Fallback Chain ────────────────────────────

    def _extract_with_fallback(
        self, pdf_path: str, metadata: DocumentMetadata
    ) -> Tuple[List[PageContent], str]:
        """Try extraction methods in order; return first success."""

        # Method 1: PyMuPDF (best quality + image support)
        if PYMUPDF_AVAILABLE:
            try:
                pages  = self._extract_pymupdf(pdf_path)
                quality = self._calculate_document_quality(pages)
                if quality >= 0.2:
                    logger.info("   Extraction via: PyMuPDF ✅")
                    return pages, "PyMuPDF"
            except Exception as e:
                logger.warning(f"PyMuPDF failed: {e}")

        # Method 2: pdfplumber (great for tables)
        if PDFPLUMBER_AVAILABLE:
            try:
                pages  = self._extract_pdfplumber(pdf_path)
                quality = self._calculate_document_quality(pages)
                if quality >= 0.2:
                    logger.info("   Extraction via: pdfplumber ✅")
                    return pages, "pdfplumber"
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")

        # Method 3: PyPDF2 (fallback)
        if PYPDF2_AVAILABLE:
            try:
                pages = self._extract_pypdf2(pdf_path)
                logger.info("   Extraction via: PyPDF2 ✅")
                return pages, "PyPDF2"
            except Exception as e:
                logger.warning(f"PyPDF2 failed: {e}")

        # Method 4: OCR (last resort for scanned PDFs)
        if OCR_AVAILABLE and self.ocr_fallback:
            try:
                pages = self._extract_ocr(pdf_path)
                logger.info("   Extraction via: OCR (Tesseract) ✅")
                return pages, "OCR"
            except Exception as e:
                logger.error(f"OCR failed: {e}")

        raise ValueError(f"All extraction methods failed for: {pdf_path}")

    # ── PyMuPDF Extraction ───────────────────────────────────

    def _extract_pymupdf(self, pdf_path: str) -> List[PageContent]:
        """Extract text + images using PyMuPDF (fitz)."""
        pages = []
        doc   = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page      = doc[page_num]
            raw_text  = page.get_text("text")
            cleaned   = self._clean_text(raw_text)

            image_paths = []
            if self.extract_images:
                image_paths = self._extract_page_images_pymupdf(
                    doc, page, page_num
                )

            # Extract tables (basic detection)
            tables = []
            try:
                table_finder = page.find_tables()
                for table in table_finder.tables:
                    tables.append(table.extract())
            except Exception:
                pass

            pages.append(PageContent(
                page_number  = page_num + 1,
                raw_text     = raw_text,
                cleaned_text = cleaned,
                char_count   = len(cleaned),
                word_count   = len(cleaned.split()),
                quality_score = self._assess_text_quality(cleaned),
                has_images   = len(image_paths) > 0,
                has_tables   = len(tables) > 0,
                image_paths  = image_paths,
                table_data   = tables,
            ))

        doc.close()
        return pages

    def _extract_page_images_pymupdf(
        self, doc, page, page_num: int
    ) -> List[str]:
        """Extract and save images from a PDF page."""
        saved_paths = []
        image_list  = page.get_images(full=True)

        for img_idx, img_ref in enumerate(image_list):
            try:
                xref       = img_ref[0]
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                img_ext    = base_image["ext"]

                img_path = (
                    self.image_output_dir
                    / f"page{page_num + 1}_img{img_idx + 1}.{img_ext}"
                )
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                saved_paths.append(str(img_path))

                # Run OCR on extracted image if it looks like a chart/diagram
                if OCR_AVAILABLE:
                    try:
                        pil_img  = Image.open(io.BytesIO(img_bytes))
                        ocr_text = pytesseract.image_to_string(pil_img).strip()
                        if ocr_text:
                            logger.debug(
                                f"   Image OCR [{img_path.name}]: "
                                f"{truncate_text(ocr_text, 80)}"
                            )
                    except Exception:
                        pass

            except Exception as e:
                logger.debug(f"Image extraction error page {page_num}: {e}")

        return saved_paths

    # ── pdfplumber Extraction ────────────────────────────────

    def _extract_pdfplumber(self, pdf_path: str) -> List[PageContent]:
        """Extract text + tables using pdfplumber."""
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                raw_text = page.extract_text() or ""
                cleaned  = self._clean_text(raw_text)

                tables = []
                try:
                    for tbl in page.extract_tables():
                        if tbl:
                            tables.append(tbl)
                except Exception:
                    pass

                pages.append(PageContent(
                    page_number  = page_num + 1,
                    raw_text     = raw_text,
                    cleaned_text = cleaned,
                    char_count   = len(cleaned),
                    word_count   = len(cleaned.split()),
                    quality_score = self._assess_text_quality(cleaned),
                    has_tables   = len(tables) > 0,
                    table_data   = tables,
                ))
        return pages

    # ── PyPDF2 Extraction ────────────────────────────────────

    def _extract_pypdf2(self, pdf_path: str) -> List[PageContent]:
        """Fallback extraction using PyPDF2."""
        pages = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                try:
                    raw_text = page.extract_text() or ""
                except Exception:
                    raw_text = ""

                cleaned = self._clean_text(raw_text)
                pages.append(PageContent(
                    page_number  = page_num + 1,
                    raw_text     = raw_text,
                    cleaned_text = cleaned,
                    char_count   = len(cleaned),
                    word_count   = len(cleaned.split()),
                    quality_score = self._assess_text_quality(cleaned),
                ))
        return pages

    # ── OCR Extraction ───────────────────────────────────────

    def _extract_ocr(self, pdf_path: str) -> List[PageContent]:
        """Last-resort: convert PDF pages to images and run OCR."""
        import pdf2image
        pages    = []
        pil_imgs = pdf2image.convert_from_path(pdf_path, dpi=300)

        for page_num, pil_img in enumerate(pil_imgs):
            try:
                raw_text = pytesseract.image_to_string(
                    pil_img, config="--psm 6"
                )
            except Exception:
                raw_text = ""

            cleaned = self._clean_text(raw_text)
            pages.append(PageContent(
                page_number  = page_num + 1,
                raw_text     = raw_text,
                cleaned_text = cleaned,
                char_count   = len(cleaned),
                word_count   = len(cleaned.split()),
                quality_score = self._assess_text_quality(cleaned),
            ))
        return pages

    # ── Text Cleaning ────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove null bytes and non-printable control chars (keep newlines)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Remove PDF artifacts: bullets, replacement chars
        artifacts = ["\ufffd", "\x00", "\uf0b7", "\u2022\u2022"]
        for a in artifacts:
            text = text.replace(a, "")

        # Normalize quotes & dashes
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2013", "-").replace("\u2014", "-")

        # Collapse multiple spaces/newlines
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # ── Quality Assessment ───────────────────────────────────

    def _assess_text_quality(self, text: str) -> float:
        """
        Score text quality from 0.0 to 1.0.
        Penalizes: very short text, bad word/char ratio,
                   excessive special chars (OCR artifacts).
        """
        if not text:
            return 0.0

        char_count = len(text)
        word_count = len(text.split())

        # Length score
        length_score = min(char_count / self.min_chars_per_page, 1.0)

        # Word/character ratio (normal text ~5 chars/word)
        avg_word_len     = char_count / max(word_count, 1)
        word_ratio_score = 1.0 if 3 <= avg_word_len <= 10 else 0.5

        # Special character ratio (high = OCR noise)
        special_chars    = sum(
            1 for c in text
            if not c.isalnum() and c not in " .,!?;:-\"'\n\t()"
        )
        special_ratio    = special_chars / max(char_count, 1)
        special_score    = 1.0 - min(special_ratio * 2, 0.5)

        return (length_score + word_ratio_score + special_score) / 3

    def _calculate_document_quality(self, pages: List[PageContent]) -> float:
        """Average quality score across all pages."""
        if not pages:
            return 0.0
        scores = [p.quality_score for p in pages]
        return sum(scores) / len(scores)
