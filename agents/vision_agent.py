"""Vision analysis agent using Gemini multimodal capabilities.

Accepts a raw image upload and returns a structured text description
of any MATLAB errors, code snippets, or Simulink block configurations
visible in the image. No OCR libraries are used.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import google.generativeai as genai
from PIL import Image

from config.settings import get_settings

logger: logging.Logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "vision_prompt.txt"
_VISION_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")


def analyze_image(image_bytes: bytes) -> str:
    """Analyze a MATLAB screenshot or Simulink diagram using Gemini Vision.

    Args:
        image_bytes: Raw image bytes from the user's file upload.

    Returns:
        Structured text description of all MATLAB-relevant content in the image.

    Raises:
        ValueError: If ``image_bytes`` is empty.
        RuntimeError: If the Gemini Vision API call fails.
    """
    if not image_bytes:
        raise ValueError("image_bytes must not be empty.")

    cfg = get_settings()
    genai.configure(api_key=cfg.gemini_api_key)
    model = genai.GenerativeModel(cfg.gemini_model)

    try:
        image = Image.open(io.BytesIO(image_bytes))
        response = model.generate_content([_VISION_PROMPT, image])
        vision_text = response.text.strip()
        logger.info("Vision analysis complete — %d characters extracted.", len(vision_text))
        return vision_text

    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Vision analysis failed: %s", exc)
        raise RuntimeError(f"Vision analysis failed: {exc}") from exc
