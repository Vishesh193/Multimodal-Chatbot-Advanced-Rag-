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
Analyze the provided hospital final bill image with extreme precision. Extract the following details:

1. **Hospital Name**: Look for the large header or logo (e.g., "TGH Onco-Life Cancer Centre").
2. **Patient Name**: Find the text directly to the right of "PATIENT NAME" (e.g., "MRS. ZAMBRE CHAAYA SHRIKRUSHNA").
3. **DOA (Admission Date)**: Find the date next to "DOA" (e.g., "12-09-2023").
4. **Diagnosis**: Extract the text next to "DAIGNOSIS" (spelled with an I) or "DIAGNOSIS" (e.g., "CA RIGHT BREAST").
5. **Total Amount**: Look for the "Gross Total Amount" or "TOTAL AMOUNT" figure at the bottom (e.g., "26527").
6. **Breakdown**: Create a list of all "PARTICULARS" line items and their corresponding "TOTAL AMOUNT" values.

Return the data in the following JSON format ONLY:
{
  "hospital_name": "...",
  "patient_name": "...",
  "admission_date": "...",
  "discharge_date": "...",
  "total_amount": "...",
  "diagnosis": "...",
  "breakdown": [
    {"category": "Item/Particular Name", "amount": "Cost"}
  ]
}

- For any field not visible, use "Not Found".
- DO NOT guess or hallucinate numbers not on the bill.
- Respond with EXCLUSIVELY valid JSON.
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
                max_tokens=1000  # Increased for potentially longer breakdown
            )

            response_text = result.get("answer", "")
            if response_text.startswith("Error:"):
                return {"error": response_text}

            # More robust JSON extraction
            try:
                # Find the first { and last }
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx+1]
                else:
                    json_str = response_text
                
                data = json.loads(json_str)
                logger.info("✅ Bill details extracted successfully.")
                
                # Check for "John Doe" or other signs of hallucination
                if data.get("patient_name") == "John Doe" or data.get("hospital_name") == "Name of hospital":
                    logger.warning("Extraction contains template defaults. Vision model may not have analyzed the image.")
                    return {
                        "error": "Extraction failed. The model returned a template answer. Please ensure you have a vision-capable model (like llava) installed.",
                        "raw_extracted": data
                    }
                
                return data
                
            except (json.JSONDecodeError, ValueError):
                logger.warning("Could not parse vision model response as JSON. Returning raw text.")
                return {
                    "raw_text": response_text,
                    "model_used": result.get("model_used")
                }

        except Exception as e:
            logger.error(f"Bill analysis failed: {e}")
            return {"error": str(e)}

