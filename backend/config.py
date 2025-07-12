"""
Configuration module for Gemini AI integration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Google AI Studio API Key - directly configured
GOOGLE_AI_STUDIO_API_KEY = "AIzaSyCErvD_C7fM99E8Q4Sze789p3q8JCYpyWA"

# Gemini AI configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", GOOGLE_AI_STUDIO_API_KEY)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", GOOGLE_AI_STUDIO_API_KEY)
GEMINI_MODEL = "gemini-1.5-flash"  # Updated to latest model
GEMINI_TEMPERATURE = 0.7
GEMINI_MAX_OUTPUT_TOKENS = 4096  # Increased for better responses

# Define analysis parameters
DEFAULT_SIMILARITY_THRESHOLD = 0.7
MAX_ITERATIONS = 5
MIN_IMPROVEMENT_THRESHOLD = 0.05  # 5% improvement required to continue

# Configure logging
ENABLE_DEBUG_LOGGING = True  # Enable for debugging AI modes

class Config:
    """Configuration class for TrackMyPDB"""
    
    # Google AI Studio API Key
    GOOGLE_AI_STUDIO_API_KEY = GOOGLE_AI_STUDIO_API_KEY
    
    # Gemini AI configuration with fallback hierarchy
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", GOOGLE_AI_STUDIO_API_KEY)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", GOOGLE_AI_STUDIO_API_KEY)
    API_KEY = GEMINI_API_KEY or GOOGLE_API_KEY or GOOGLE_AI_STUDIO_API_KEY
    
    GEMINI_MODEL = "gemini-1.5-flash"  # Latest stable model
    GEMINI_TEMPERATURE = 0.7
    GEMINI_MAX_OUTPUT_TOKENS = 4096
    
    # Analysis parameters
    DEFAULT_SIMILARITY_THRESHOLD = 0.7
    MAX_ITERATIONS = 5
    MIN_IMPROVEMENT_THRESHOLD = 0.05
    
    # AI Mode Configuration
    AI_ENABLED = True
    AI_MODE_TIMEOUT = 30  # seconds
    AI_RETRY_ATTEMPTS = 3
    
    # Logging configuration
    ENABLE_DEBUG_LOGGING = True
    
    @classmethod
    def get_api_key(cls):
        """Get the API key with proper fallback"""
        return cls.API_KEY
    
    @classmethod
    def validate_ai_setup(cls):
        """Validate AI configuration"""
        api_key = cls.get_api_key()
        if not api_key:
            return False, "No API key configured"
        if len(api_key) < 20:
            return False, "API key appears invalid"
        return True, "AI configuration valid"