"""
Configuration module for Gemini AI integration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini AI configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-flash"
GEMINI_TEMPERATURE = 0.7
GEMINI_MAX_OUTPUT_TOKENS = 2048

# Define analysis parameters
DEFAULT_SIMILARITY_THRESHOLD = 0.7
MAX_ITERATIONS = 5
MIN_IMPROVEMENT_THRESHOLD = 0.05  # 5% improvement required to continue

# Configure logging
ENABLE_DEBUG_LOGGING = False