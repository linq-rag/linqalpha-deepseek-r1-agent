import os
import sys
import logging


# -----------------------------
# Environment and Logging Setup
# -----------------------------
from dotenv import load_dotenv
load_dotenv('.env')

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not FIREWORKS_API_KEY:
    raise ValueError("Missing FIREWORKS_API_KEY environment variable")
if not SERPAPI_API_KEY:
    raise ValueError("Missing SERPAPI_API_KEY environment variable")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def safe_log(message, prefix=None):
    """Log messages with consistent formatting and optional prefix."""
    try:
        formatted_message = ""
        if prefix:
            formatted_message = f"[{prefix}] "
        formatted_message += str(message)
        logger.info(formatted_message)
    except Exception as e:
        print(f"Logging failed: {e} - original message: {message}", file=sys.stderr)
