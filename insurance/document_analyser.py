"""
document_analyser.py — Bill & Document Image Analyser
=======================================================
Uses the Ollama vision model to extract structured data from uploaded
hospital bills, discharge summaries, or rejection letters.
"""

import json
from typing import Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Extraction Prompts ────────────────────────────────────────────────────
BILL_EXTRACTION_PROMPT = """\
You are an expert medical bill analyst. Extract the key structured details \
from the provided hospital bill image. 

Return the data in the following JSON format ONLY:
{
  "hospital_name": "Name of hospital",
  "patient_name": "Name of patient",
  "admission_date": "YYYY-MM-DD",
  "discharge_date": "YYYY-MM-DD",
  "total_amount": "Total bill amount with currency",
  "diagnosis": "Main diagnosis or procedure mentioned",
  "breakdown": [
    {"category": "Room Charges", "amount": "..."},
    {"category": "Surgery", "amount": "..."}
  ]
}

If any field is missing or unclear, use "Not Found".
Respond with valid JSON ONLY.
"""


class DocumentAnalyser:
    """
    Analyses uploaded insurance documents (bills, discharge summaries)
    using the vision LLM.
    """

    def __init__(self, vision_llm_client):
        self.vision_llm = vision_llm_client

    def analyse_bill(self, image_path: str) -> Dict[str, Any]:
        """
        Extract structured details from a hospital bill image.
        """
        if not self.vision_llm:
            return {"error": "Vision LLM is not configured."}

        logger.info(f"📄 Analysing medical bill: {image_path}")

        try:
            result = self.vision_llm.generate(
                prompt=BILL_EXTRACTION_PROMPT,
                images=[image_path],
                max_tokens=500
            )

            response_text = result.get("answer", "")
            
            # Try to parse JSON from the response (in case the model wraps it in markdown blocks)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            
            try:
                data = json.loads(clean_text)
                logger.info("✅ Bill details extracted successfully.")
                return data
            except json.JSONDecodeError:
                logger.warning("Could not parse vision model response as JSON. Returning raw text.")
                return {
                    "raw_text": response_text,
                    "model_used": result.get("model_used")
                }

        except Exception as e:
            logger.error(f"Bill analysis failed: {e}")
            return {"error": str(e)}

